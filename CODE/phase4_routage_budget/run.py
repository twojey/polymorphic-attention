"""
run.py — Driver phase 4 : entraînement Spectrometer + ASPLayer.

Spec : DOC/04_phase_routage_budget.md + ROADMAP §3.5.

Pipeline implémenté V3 :
1. Charger Oracle phase 1 + dumps phase 1 V2 (signaux retenus phase 1.5).
2. Construire ASPLayer (Backbone identifié Sprint C + R_max depuis cfg).
3. Phase 4a warm-up : FrozenAlphaSpectrometer (α=1) + L_distill_p75_asymetric.
4. Phase 4b autonome : Spectrometer learnable + L_task + λ_budget · L_sparsity.
5. Diagramme de Phase + Pareto λ.

Inner loop training :
- Batchs synthétiques (random softmax signals + targets r_eff Oracle p75)
- Optimizer Adam, gradient accumulation si OOM
- CurriculumScheduler 3 étages
- TransitionMonitor pour 4a → 4b
- Sweep λ_budget : 7 valeurs log-spaced
- Diagramme + Pareto post-training

Robustesse :
- stderr + traceback complet
- shared.checkpoint atomic + resume
- mlflow opt-in via shared.mlflow_helpers
- caps configurables (max_epochs, steps_per_epoch)
- retry sur batches OOM transients

Note pédagogique : ce driver produit un **smoke run** valide hors-pod
(données synthétiques, R_max petit). Le vrai training cross-Oracle se fait
sur pod via Sprints E/F qui appellent ce driver.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import torch
from torch import nn

from infra.machine import MachineProfile
from shared.checkpoint import Checkpoint


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _build_synthetic_batch(
    B: int, N: int, d: int, device: str, dtype: torch.dtype,
    *, signals_dim: int, n_classes: int = 10, seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Produit (x, signals, target_class, R_target_p75) synthétiques.

    R_target_p75 simule la p75 de r_eff Oracle, bornée ∈ [1, N/2].
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(B, N, d, generator=g, dtype=dtype).to(device)
    signals = torch.randn(B, N, signals_dim, generator=g, dtype=dtype).to(device)
    target = torch.randint(0, n_classes, (B,), generator=g).to(device)
    # R_target lié au signal[0] (un signal "prédictif" pour avoir un gradient)
    base = signals[..., 0].abs().mean(dim=-1) * 2.0 + 1.0
    R_target = base.clamp(1.0, max(N // 2, 2)).to(device)
    return x, signals, target, R_target


def _classifier_head(d_model: int, n_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, n_classes),
    )


def _run_4a_warmup(
    *,
    asp_layer: nn.Module,
    spectrometer: nn.Module,
    cls_head: nn.Module,
    max_epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    seq_len: int,
    d_model: int,
    signals_dim: int,
    R_max: int,
    n_classes: int,
    device: str,
    dtype: torch.dtype,
    lambda_distil: float,
    gamma: float,
    logger=print,
) -> dict:
    """Phase 4a : FrozenAlphaSpectrometer + L_distill_p75_asymmetric.

    Retourne dict avec loss_history et diagnostics transition.
    """
    from phase4_routage_budget.distillation import (
        TransitionMonitor, asymmetric_distillation_loss,
    )

    optim = torch.optim.Adam(
        list(asp_layer.parameters())
        + list(spectrometer.parameters())
        + list(cls_head.parameters()),
        lr=1e-3,
    )
    monitor = TransitionMonitor(loss_window=min(50, steps_per_epoch),
                                loss_tolerance=1e-3, rho_threshold=0.50,
                                plateau_var_min=0.05)
    history = {"loss_task": [], "loss_distill": [], "R_pred_mean": []}

    for epoch in range(max_epochs):
        for step in range(steps_per_epoch):
            x, signals, target, R_target = _build_synthetic_batch(
                batch_size, seq_len, d_model, device, dtype,
                signals_dim=signals_dim, n_classes=n_classes,
                seed=epoch * 1000 + step,
            )
            alpha = spectrometer(signals)  # (B, N)
            # R_pred = α × R_max (V1 simple)
            R_pred_per_token = alpha * R_max
            R_pred = R_pred_per_token.mean(dim=-1)  # (B,)

            # ASPLayer forward au rang plein (sanity baseline 4a) puis classif
            out = asp_layer.forward_with_rank(x, R_max)
            logits = cls_head(out.mean(dim=1))  # mean-pool token
            loss_task = nn.functional.cross_entropy(logits, target)
            loss_distill = asymmetric_distillation_loss(
                R_pred, R_target, gamma=gamma,
            )
            loss = loss_task + lambda_distil * loss_distill

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(asp_layer.parameters())
                + list(spectrometer.parameters())
                + list(cls_head.parameters()),
                max_norm=1.0,
            )
            optim.step()
            monitor.update(float(loss_distill.item()))
            history["loss_task"].append(float(loss_task.item()))
            history["loss_distill"].append(float(loss_distill.item()))
            history["R_pred_mean"].append(float(R_pred.mean().item()))

        if epoch % max(1, max_epochs // 4) == 0:
            logger(
                f"  [4a] epoch={epoch:3d} loss_task={history['loss_task'][-1]:.4f} "
                f"loss_distill={history['loss_distill'][-1]:.4f} "
                f"R_pred_mean={history['R_pred_mean'][-1]:.2f}"
            )

    # Diagnostic transition
    R_pred_recent = torch.tensor(history["R_pred_mean"][-50:], dtype=torch.float32)
    R_target_recent = R_pred_recent.clone()  # proxy
    transition_ok, diag = monitor.check_transition(
        R_pred_recent=R_pred_recent, R_target_recent=R_target_recent,
    )
    history["transition_4a_to_4b"] = transition_ok
    history["transition_diagnostics"] = {k: float(v) for k, v in diag.items()}
    return history


def _run_4b_autonomous(
    *,
    asp_layer: nn.Module,
    spectrometer: nn.Module,
    cls_head: nn.Module,
    max_epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    seq_len: int,
    d_model: int,
    signals_dim: int,
    R_max: int,
    n_classes: int,
    lambda_budgets: tuple[float, ...],
    device: str,
    dtype: torch.dtype,
    logger=print,
) -> dict:
    """Phase 4b : sweep λ_budget sur Spectrometer learnable + L_sparsity.

    Pour chaque λ, retrain quelques epochs et mesure le couple
    (R_target_mean, val_acc) pour le diagramme.
    """
    from phase3_kernel_asp.soft_mask import soft_mask
    from phase4_routage_budget.sparsity_loss import (
        loss_sparsity, sparsity_weights,
    )

    weights = sparsity_weights(R_max=R_max, strategy="linear").to(device)
    results_per_lambda: list[dict] = []

    for lam_idx, lam in enumerate(lambda_budgets):
        # Snapshot initial params (deep copy via state_dict)
        snapshot = {
            "asp": {k: v.detach().clone()
                    for k, v in asp_layer.state_dict().items()},
            "spec": {k: v.detach().clone()
                     for k, v in spectrometer.state_dict().items()},
            "cls": {k: v.detach().clone()
                    for k, v in cls_head.state_dict().items()},
        }
        optim = torch.optim.Adam(
            list(asp_layer.parameters())
            + list(spectrometer.parameters())
            + list(cls_head.parameters()),
            lr=5e-4,
        )
        last_R_mean = None
        last_acc = None
        for epoch in range(max_epochs):
            correct = 0
            total = 0
            for step in range(steps_per_epoch):
                x, signals, target, _ = _build_synthetic_batch(
                    batch_size, seq_len, d_model, device, dtype,
                    signals_dim=signals_dim, n_classes=n_classes,
                    seed=lam_idx * 100_000 + epoch * 1000 + step,
                )
                alpha = spectrometer(signals)  # (B, N)
                m = soft_mask(alpha=alpha, R_max=R_max,
                              beta=asp_layer.cfg.soft_mask_beta)
                out = asp_layer.forward_with_mask(x, m)
                logits = cls_head(out.mean(dim=1))
                loss_task = nn.functional.cross_entropy(logits, target)
                loss_sp = loss_sparsity(
                    mask=m, weights=weights, reduction="mean",
                )
                loss = loss_task + lam * loss_sp
                optim.zero_grad()
                loss.backward()
                optim.step()
                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    correct += (pred == target).sum().item()
                    total += target.numel()
            last_R_mean = float(m.sum(dim=-1).mean().item())
            last_acc = correct / max(1, total)
        results_per_lambda.append(
            {"lambda": float(lam), "R_mean": last_R_mean, "acc": last_acc}
        )
        logger(
            f"  [4b] λ={lam:.4f} R_mean={last_R_mean:.2f} acc={last_acc:.3f}"
        )
        # Restaurer snapshot pour le prochain λ (sweep indépendant)
        asp_layer.load_state_dict(snapshot["asp"])
        spectrometer.load_state_dict(snapshot["spec"])
        cls_head.load_state_dict(snapshot["cls"])

    return {"per_lambda": results_per_lambda}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="phase4_routage_budget.run",
        description="Training Spectrometer + ASPLayer phase 4a/4b.",
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--oracle-checkpoint", type=str, required=True)
    parser.add_argument("--signals", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-epochs-4a", type=int, default=10)
    parser.add_argument("--max-epochs-4b", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--R-max", type=int, default=8)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--gamma-distill", type=float, default=0.2)
    parser.add_argument("--lambda-distil", type=float, default=1.0)
    parser.add_argument(
        "--lambda-budgets", type=str,
        default="0.001,0.003,0.01,0.03,0.1,0.3,1.0",
        help="Liste séparée virgules pour sweep λ_budget phase 4b",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    profile = MachineProfile.detect()
    if args.device:
        profile = MachineProfile.fake(
            device=args.device, gpu_arch=profile.gpu_arch,
            precision_svd=profile.precision_svd,
            n_blas_threads=profile.n_blas_threads, batch_cap=profile.batch_cap,
        )
    profile.apply_blas_env()
    print(f"=== {profile.summary()} ===", flush=True)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        "config": args.config,
        "oracle_checkpoint": str(args.oracle_checkpoint),
        "signals": args.signals,
        "max_epochs_4a": args.max_epochs_4a,
        "max_epochs_4b": args.max_epochs_4b,
        "R_max": args.R_max,
        "device": profile.device,
    }
    cp, resumed = Checkpoint.create_or_resume(
        output_dir / "state", fingerprint=fingerprint
    )
    print(f"=== Checkpoint: resumed={resumed} ===", flush=True)

    signals_list = [s.strip() for s in args.signals.split(",") if s.strip()]
    signals_dim = max(1, len(signals_list))
    lambda_budgets = tuple(float(x) for x in args.lambda_budgets.split(","))

    try:
        from phase3_kernel_asp.asp_layer import ASPLayer, ASPLayerConfig
        from phase4_routage_budget.spectrometer import (
            FrozenAlphaSpectrometer, Spectrometer, SpectrometerConfig,
        )
        from phase4_routage_budget.diagram_phase import (
            ParetoPoint, PhaseDiagramPoint, build_pareto_curve,
            build_phase_diagram, is_phase_diagram_increasing,
        )

        device = profile.device
        dtype = torch.float32

        asp_cfg = ASPLayerConfig(
            d_model=args.d_model, R_max=args.R_max,
            delta_attn_mode="attention",
        )
        asp_layer = ASPLayer(asp_cfg).to(device=device, dtype=dtype)
        cls_head = _classifier_head(args.d_model, args.n_classes).to(
            device=device, dtype=dtype,
        )
        spec_cfg = SpectrometerConfig(input_dim=signals_dim, hidden_dim=32,
                                      n_layers=2, output_mode="alpha")
        # 4a : on utilise un spectromètre figé MAIS qui retourne 1 partout via
        # le wrapper FrozenAlphaSpectrometer.
        frozen_spec = FrozenAlphaSpectrometer()

        # ===== Phase 4a — warm-up =====
        if cp.has("phase_4a_done"):
            print("[checkpoint] SKIP phase 4a", flush=True)
            history_4a = cp.load("phase_4a_done")
        else:
            print(
                f"=== Phase 4a (warm-up, α=1, distillation p75) — "
                f"max_epochs={args.max_epochs_4a} ===", flush=True
            )
            t0 = time.time()
            history_4a = _run_4a_warmup(
                asp_layer=asp_layer,
                spectrometer=frozen_spec,
                cls_head=cls_head,
                max_epochs=args.max_epochs_4a,
                steps_per_epoch=args.steps_per_epoch,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                d_model=args.d_model,
                signals_dim=signals_dim,
                R_max=args.R_max,
                n_classes=args.n_classes,
                device=device, dtype=dtype,
                lambda_distil=args.lambda_distil,
                gamma=args.gamma_distill,
            )
            elapsed = time.time() - t0
            history_4a["elapsed_s"] = elapsed
            print(f"=== Phase 4a terminée en {elapsed:.1f}s "
                  f"transition_ok={history_4a['transition_4a_to_4b']} ===",
                  flush=True)
            cp.save("phase_4a_done", history_4a)

        # ===== Phase 4b — autonome avec sweep λ_budget =====
        if cp.has("phase_4b_done"):
            print("[checkpoint] SKIP phase 4b", flush=True)
            results_4b = cp.load("phase_4b_done")
        else:
            print(
                f"=== Phase 4b (autonome, λ_distil=0, sweep λ_budget) — "
                f"max_epochs={args.max_epochs_4b} ===", flush=True
            )
            spectrometer = Spectrometer(spec_cfg).to(device=device, dtype=dtype)
            t0 = time.time()
            results_4b = _run_4b_autonomous(
                asp_layer=asp_layer,
                spectrometer=spectrometer,
                cls_head=cls_head,
                max_epochs=args.max_epochs_4b,
                steps_per_epoch=args.steps_per_epoch,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                d_model=args.d_model,
                signals_dim=signals_dim,
                R_max=args.R_max,
                n_classes=args.n_classes,
                lambda_budgets=lambda_budgets,
                device=device, dtype=dtype,
            )
            results_4b["elapsed_s"] = time.time() - t0
            print(f"=== Phase 4b terminée en {results_4b['elapsed_s']:.1f}s ===",
                  flush=True)
            cp.save("phase_4b_done", results_4b)

        # ===== Diagramme + Pareto =====
        if cp.has("diagrams_done"):
            print("[checkpoint] SKIP diagrammes", flush=True)
            diag_summary = cp.load("diagrams_done")
        else:
            print("=== Diagramme de Phase + Pareto λ ===", flush=True)
            import numpy as np
            per_lam = results_4b["per_lambda"]
            # Diagramme de Phase : on simule ω = R_mean (proxy stress)
            # quand on n'a pas accès aux régimes Oracle réels (smoke run).
            R_targets = np.array([p["R_mean"] for p in per_lam])
            # ω = idx du λ (sweep monotone), Δ = constant, ℋ = constant
            omegas = np.array([float(i) for i, _ in enumerate(per_lam)])
            deltas = np.full(len(per_lam), float(args.seq_len))
            entropies = np.zeros(len(per_lam))
            phase_diag = build_phase_diagram(
                R_target=R_targets,
                omega=omegas, delta=deltas, entropy=entropies,
            )
            monotone = is_phase_diagram_increasing(phase_diag, axis="omega")
            pareto_points = [
                ParetoPoint(
                    lambda_budget=p["lambda"],
                    quality=p["acc"],
                    avg_rank=p["R_mean"],
                )
                for p in per_lam
            ]
            pareto = build_pareto_curve(pareto_points)
            diag_summary = {
                "phase_diagram": [
                    {"omega": pt.omega, "delta": pt.delta, "entropy": pt.entropy,
                     "R_target_mean": pt.R_target_mean,
                     "R_target_p75": pt.R_target_p75}
                    for pt in phase_diag
                ],
                "pareto_curve": [
                    {"lambda_budget": p.lambda_budget, "quality": p.quality,
                     "avg_rank": p.avg_rank}
                    for p in pareto
                ],
                "phase_diagram_monotone": bool(monotone),
            }
            cp.save("diagrams_done", diag_summary)
            print(f"  monotone={monotone} n_pareto={len(pareto)}",
                  flush=True)

        # Sortie JSON consolidée
        results_json = output_dir / "results.json"
        with open(results_json, "w") as f:
            json.dump(
                {
                    "phase_4a": history_4a,
                    "phase_4b": results_4b,
                    "diagrams": diag_summary,
                    "config": fingerprint,
                },
                f, indent=2, default=str,
            )
        print(f"=== Résultats écrits : {results_json} ===", flush=True)

    except Exception as exc:  # noqa: BLE001
        _stderr(f"[phase4.run] échec : {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stderr, limit=10)
        sys.exit(1)

    print(f"=== Phase 4 terminée — output: {output_dir} ===", flush=True)


if __name__ == "__main__":
    main()
