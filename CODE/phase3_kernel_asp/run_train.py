"""
run_train.py — driver d'entraînement ASPTransformer phase 3.

Spec : DOC/03 §§ 3-7. Pipeline :

1. Hydra config phase 3 (`asp_layer.yaml`).
2. Reconstruit tri-partition SSG identique à phase 1 → set `init_phase3`.
3. Smart Init Matriochka optionnel depuis dumps phase 2 (DOC/03 §5.2).
4. Instancie ASPTransformer (Backbone dérivé de phase 2 classe SCH dominante).
5. Charge Oracle checkpoint pour quality baseline (saturation check).
6. Boucle entraînement Fabric BF16 mixed :
       L = L_task + λ_M · L_matriochka + λ_C · L_consistency
7. Sanity checks (saturation, effondrement, monotonie, lissité) en eval.
8. Verdict GO/NO-GO + manifest MLflow.

Usage :
    PYTHONPATH=CODE uv run python -m phase3_kernel_asp.run_train \\
        --config-path=../../OPS/configs/phase3 --config-name=asp_layer \\
        +oracle_checkpoint=OPS/checkpoints/oracle_e2f0b5e.ckpt \\
        +phase2_dump_dir=/workspace/phase1_extract \\
        backbone.class_name=toeplitz  # à override post-verdict phase 2

Args clés :
- `oracle_checkpoint` : pour quality_oracle dans sanity saturation.
- `phase2_dump_dir` : pour Smart Init Matriochka (optionnel, defaults xavier).
- `backbone.class_name` : à fixer post-phase-2 (toeplitz / hankel / composite).
"""

from __future__ import annotations

import gc
import sys
import time
import traceback
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Subset

from phase1_metrologie.oracle.train import collate, find_query_positions
from phase1_metrologie.oracle.transformer import OracleConfig, OracleTransformer
from phase1_metrologie.ssg.structure_mnist import (
    SplitConfig,
    StructureMNISTConfig,
    StructureMNISTDataset,
    Vocab,
    split_indices,
    sweep_monovariate,
)
from phase3_kernel_asp.checkpoint import Phase3State
from phase3_kernel_asp.losses import (
    loss_consistency,
    loss_matriochka,
    matriochka_rank_schedule,
    matriochka_weights,
)
from phase3_kernel_asp.sanity import run_all_sanity_checks
from phase3_kernel_asp.smart_init import make_smart_init_for_asp_layer
from phase3_kernel_asp.transformer import ASPTransformer, ASPTransformerConfig
from shared.mlflow_helpers import log_yaml_config, start_run
from shared.runner import finalize_manifest, make_manifest, write_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]


# -----------------------------------------------------------------------------
# Helpers robustesse (cohérent avec phase 1 / 2)
# -----------------------------------------------------------------------------


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _safe_mlflow(operation: str, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        _stderr(f"[mlflow] {operation} a échoué : {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stderr, limit=3)
        return None


def _print_resource_snapshot(prefix: str) -> None:
    msg = prefix
    try:
        import psutil
        mem = psutil.virtual_memory()
        msg += f" RAM avail={mem.available/1e9:.1f} GB used={mem.percent:.0f}%"
    except ImportError:
        msg += " psutil indisponible"
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        msg += f" | VRAM free={free_b/1e9:.1f} GB / {total_b/1e9:.1f} GB"
    print(msg, flush=True)


# -----------------------------------------------------------------------------
# Dataset (réutilise phase 1 SSG, set init_phase3)
# -----------------------------------------------------------------------------


def _build_sweep_dataset(cfg: DictConfig) -> ConcatDataset:
    base = StructureMNISTConfig(
        omega=cfg.ssg.axes.omega.reference,
        delta=cfg.ssg.axes.delta.reference,
        entropy=cfg.ssg.axes.entropy.reference,
        n_examples=cfg.ssg.n_examples_per_regime,
        n_ops=cfg.ssg.n_ops,
        n_noise=cfg.ssg.n_noise,
        seed=cfg.ssg.seed_train,
    )
    datasets: list[StructureMNISTDataset] = []
    for axis in ("omega", "delta", "entropy"):
        sweep_values = list(cfg.ssg.axes[axis].sweep)
        for c in sweep_monovariate(axis=axis, values=sweep_values, base=base):
            datasets.append(StructureMNISTDataset(c))
    return ConcatDataset(datasets)


# -----------------------------------------------------------------------------
# Oracle baseline (pour saturation check)
# -----------------------------------------------------------------------------


def _load_oracle(
    ckpt_path: str, model_cfg: OracleConfig, device: str
) -> OracleTransformer:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model = OracleTransformer(model_cfg)
    cleaned = {
        k.removeprefix("_forward_module.").removeprefix("_orig_mod."): v
        for k, v in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        _stderr(f"[oracle] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def _eval_oracle_acc(
    oracle: OracleTransformer, loader: DataLoader, query_id: int, device: str
) -> float:
    correct = 0
    total = 0
    for batch in loader:
        tokens = batch["tokens"].to(device)
        targets = batch["targets"].to(device)
        qpos = find_query_positions(tokens, query_id)
        logits = oracle(tokens, qpos)
        preds = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.numel()
    return correct / max(total, 1)


@torch.no_grad()
def _eval_asp_acc(
    asp: ASPTransformer, loader: DataLoader, query_id: int, device: str,
    *, rank: int | None = None,
) -> float:
    correct = 0
    total = 0
    for batch in loader:
        tokens = batch["tokens"].to(device)
        targets = batch["targets"].to(device)
        qpos = find_query_positions(tokens, query_id)
        if rank is None:
            logits = asp(tokens, qpos)  # rang plein (R_max)
        else:
            logits = asp.forward_at_rank(tokens, qpos, r=rank)
        preds = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.numel()
    return correct / max(total, 1)


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../../OPS/configs/phase3",
            config_name="asp_layer")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    _print_resource_snapshot("[startup]")

    # --- Resolve required overrides ---
    oracle_ckpt = OmegaConf.select(cfg, "oracle_checkpoint")
    if oracle_ckpt is None:
        raise SystemExit(
            "Override requis : `+oracle_checkpoint=path/to/oracle.ckpt` "
            "(produit par phase 1)."
        )
    if not Path(oracle_ckpt).is_file():
        raise SystemExit(f"Oracle checkpoint introuvable : {oracle_ckpt}")

    phase2_dump_dir = OmegaConf.select(cfg, "phase2_dump_dir")
    # phase2_dump_dir : None = pas de Smart Init (random Xavier)

    # --- SSG dataset (identique phase 1 — même seed) ---
    if not OmegaConf.select(cfg, "ssg"):
        # On charge la section ssg depuis oracle_smnist.yaml par défaut
        # (NB : Hydra permettrait d'utiliser defaults: mais on évite la
        # complexité ; le caller doit fournir cfg.ssg.* via override ou
        # composition).
        raise SystemExit(
            "Config doit fournir `ssg.*` (cf. oracle_smnist.yaml). "
            "Override possible via Hydra defaults ou +ssg=...,"
        )

    vocab = Vocab(n_ops=cfg.ssg.n_ops, n_noise=cfg.ssg.n_noise)
    sweep_full = _build_sweep_dataset(cfg)
    n_total = len(sweep_full)
    split = SplitConfig(
        train_oracle=cfg.dataset_split_smnist.train_oracle,
        audit_svd=cfg.dataset_split_smnist.audit_svd,
        init_phase3=cfg.dataset_split_smnist.init_phase3,
        seed=cfg.dataset_split_smnist.seed,
    )
    parts = split_indices(n_total, split)
    init_indices = parts["init_phase3"].tolist()

    # Train/val 90/10 dans init_phase3 (déterministe via seed)
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(init_indices))
    n_val = int(len(init_indices) * 0.10)
    val_indices = [init_indices[i] for i in perm[:n_val]]
    train_indices = [init_indices[i] for i in perm[n_val:]]
    print(
        f"init_phase3 : {len(init_indices)} ex → "
        f"train={len(train_indices)} val={len(val_indices)}",
        flush=True,
    )

    from functools import partial as _partial
    collate_fn = _partial(collate, pad_id=vocab.PAD)
    train_loader = DataLoader(
        Subset(sweep_full, train_indices),
        batch_size=cfg.training.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        Subset(sweep_full, val_indices),
        batch_size=cfg.training.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=False,
    )

    # --- Oracle baseline ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    oracle_cfg = OracleConfig(
        vocab_size=vocab.size,
        d_model=cfg.model.d_model, n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers, d_ff=cfg.model.d_ff,
        max_seq_len=cfg.model.max_seq_len, dropout=cfg.model.dropout,
        n_classes=cfg.model.n_classes, pad_id=vocab.PAD,
    )
    oracle = _load_oracle(str(oracle_ckpt), oracle_cfg, device)
    print("Oracle chargé.", flush=True)

    # --- Smart Init Matriochka (optionnel) ---
    matriochka_inits = None
    if phase2_dump_dir is not None and Path(phase2_dump_dir).is_dir():
        print(f"Smart Init Matriochka depuis {phase2_dump_dir}…", flush=True)
        try:
            matriochka_inits = make_smart_init_for_asp_layer(
                phase2_dump_dir,
                d_model=cfg.asp_layer.d_model,
                R_max=cfg.asp_layer.R_max,
                K_per_layer=int(OmegaConf.select(cfg, "asp_layer.smart_init_K_per_layer") or 16),
                k_per_head=int(OmegaConf.select(cfg, "asp_layer.smart_init_k_per_head") or 2),
                seed=cfg.seed,
            )
        except Exception as exc:  # noqa: BLE001
            _stderr(
                f"[smart_init] échec : {type(exc).__name__}: {exc}. "
                f"Fallback xavier random."
            )
            traceback.print_exc(file=sys.stderr, limit=3)
            matriochka_inits = None

    # --- ASPTransformer ---
    asp_cfg = ASPTransformerConfig(
        vocab_size=vocab.size,
        d_model=cfg.model.d_model, n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers, d_ff=cfg.model.d_ff,
        max_seq_len=cfg.model.max_seq_len, dropout=cfg.model.dropout,
        n_classes=cfg.model.n_classes, pad_id=vocab.PAD,
        R_max=cfg.asp_layer.R_max,
        soft_mask_beta=cfg.asp_layer.soft_mask_beta,
        init_strategy=cfg.asp_layer.init_strategy,
        backbone_class=cfg.backbone.class_name,
        backbone_params=OmegaConf.to_container(cfg.backbone.params, resolve=True) or {},
    )
    asp = ASPTransformer(asp_cfg, matriochka_init_per_layer=matriochka_inits).to(device)
    print(
        f"ASPTransformer : L={cfg.model.n_layers}, R_max={cfg.asp_layer.R_max}, "
        f"backbone={cfg.backbone.class_name}, init={cfg.asp_layer.init_strategy}, "
        f"params={sum(p.numel() for p in asp.parameters())/1e6:.2f}M",
        flush=True,
    )

    # --- Manifest + MLflow ---
    manifest = make_manifest(
        phase="3", sprint=cfg.sprint, domain=cfg.domain, seed=cfg.seed,
        config_path="OPS/configs/phase3/asp_layer.yaml",
        short_description="asp_train",
        repo=REPO_ROOT,
    )
    write_manifest(manifest, REPO_ROOT)
    if manifest.git_dirty:
        _stderr("[manifest] git_dirty=True → status=exploratory.")

    t0 = time.perf_counter()
    opt = torch.optim.AdamW(
        asp.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    task_loss_fn = torch.nn.CrossEntropyLoss()
    lam_M = float(cfg.loss.matriochka.lambda_weight)
    lam_C = float(cfg.loss.consistency.weight)
    R_max = cfg.asp_layer.R_max

    # --- Checkpoint state (cf. feedback_script_robustness §2) ---
    state_dir = Path(
        OmegaConf.select(cfg, "checkpoint.dir")
        or REPO_ROOT / "OPS" / "logs" / "phase3_state"
    )
    state, is_resumed = Phase3State.create_or_resume(
        state_dir,
        R_max=R_max,
        d_model=cfg.model.d_model, n_layers=cfg.model.n_layers,
        backbone_class=cfg.backbone.class_name,
        init_strategy=cfg.asp_layer.init_strategy,
    )
    resume_enabled = bool(OmegaConf.select(cfg, "checkpoint.enabled") or True)
    if not resume_enabled:
        state.clean()
        state, is_resumed = Phase3State.create_or_resume(
            state_dir,
            R_max=R_max, d_model=cfg.model.d_model, n_layers=cfg.model.n_layers,
            backbone_class=cfg.backbone.class_name,
            init_strategy=cfg.asp_layer.init_strategy,
        )
    print(
        f"[checkpoint] state_dir={state_dir} resume={is_resumed}",
        flush=True,
    )

    with start_run(
        experiment="phase3", run_name=manifest.run_id,
        phase="3", sprint=cfg.sprint, domain=cfg.domain,
        status=manifest.status,
    ):
        _safe_mlflow("log_yaml_config", log_yaml_config,
                     OmegaConf.to_container(cfg, resolve=True))
        _safe_mlflow("log_params", mlflow.log_params, {
            "oracle_checkpoint": str(oracle_ckpt),
            "phase2_dump_dir": str(phase2_dump_dir or "<none>"),
            "backbone_class": cfg.backbone.class_name,
            "init_strategy": cfg.asp_layer.init_strategy,
            "R_max": R_max,
            "n_train": len(train_indices),
            "n_val": len(val_indices),
            "lambda_M": lam_M,
            "lambda_C": lam_C,
            "git_dirty": str(manifest.git_dirty),
        })

        # --- Baseline Oracle acc sur val ---
        oracle_acc = _eval_oracle_acc(oracle, val_loader, vocab.QUERY, device)
        _safe_mlflow("log_metric oracle_val_acc", mlflow.log_metric,
                     "oracle_val_acc", oracle_acc)
        print(f"Oracle val acc baseline : {oracle_acc:.4f}", flush=True)

        # --- Training loop (avec checkpoint per epoch + resume) ---
        n_epochs = int(cfg.training.max_epochs)
        log_interval = int(cfg.training.log_interval_steps)
        step = 0
        best_val_acc = 0.0
        start_epoch = 0
        if is_resumed and state.has_latest():
            try:
                meta = state.load_latest(model=asp, optimizer=opt)
                start_epoch = int(meta["epoch"]) + 1
                step = int(meta["step"])
                best_val_acc = float(meta["best_val_acc"])
                print(
                    f"[checkpoint] resumed from epoch {meta['epoch']} step {step}, "
                    f"best_val_acc={best_val_acc:.4f}",
                    flush=True,
                )
                if start_epoch >= n_epochs:
                    print(
                        f"[checkpoint] training déjà fini ({start_epoch}/{n_epochs}). "
                        f"Va directement aux sanity checks.",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                _stderr(
                    f"[checkpoint] échec load_latest : {type(exc).__name__}: {exc}. "
                    f"Restart from epoch 0."
                )
                traceback.print_exc(file=sys.stderr, limit=3)
        for epoch in range(start_epoch, n_epochs):
            asp.train()
            t_epoch = time.perf_counter()
            for batch in train_loader:
                tokens = batch["tokens"].to(device)
                targets = batch["targets"].to(device)
                qpos = find_query_positions(tokens, vocab.QUERY)

                # L_task : forward rang plein
                logits_full = asp(tokens, qpos)
                l_task = task_loss_fn(logits_full, targets)

                # L_matriochka : sample ranks + sum
                ranks = matriochka_rank_schedule(
                    R_max,
                    n_samples=int(cfg.loss.matriochka.n_rank_samples),
                    seed=step,
                )
                weights = matriochka_weights(
                    ranks, strategy=cfg.loss.matriochka.weight_strategy,
                )
                def _forward_rank(_x, r):
                    return asp.forward_at_rank(_x, qpos, r)
                # Note : _forward_rank reçoit tokens, pas embeddings
                l_M = loss_matriochka(
                    asp_layer_forward=_forward_rank,
                    x=tokens, y_target=targets,
                    task_loss=task_loss_fn, ranks=ranks, weights=weights,
                )

                # L_consistency
                l_C = loss_consistency(
                    asp_layer_forward=_forward_rank,
                    x=tokens, R_max=R_max,
                    n_samples=int(cfg.loss.consistency.n_samples),
                    delta=int(cfg.loss.consistency.delta),
                    seed=step + 1,
                )

                loss = l_task + lam_M * l_M + lam_C * l_C

                opt.zero_grad()
                loss.backward()
                if cfg.training.grad_clip:
                    torch.nn.utils.clip_grad_norm_(asp.parameters(), float(cfg.training.grad_clip))
                opt.step()

                if step % log_interval == 0:
                    _safe_mlflow("log_metric loss", mlflow.log_metric, "loss", float(loss.item()), step=step)
                    _safe_mlflow("log_metric l_task", mlflow.log_metric, "l_task", float(l_task.item()), step=step)
                    _safe_mlflow("log_metric l_matriochka", mlflow.log_metric, "l_matriochka", float(l_M.item()), step=step)
                    _safe_mlflow("log_metric l_consistency", mlflow.log_metric, "l_consistency", float(l_C.item()), step=step)
                    print(
                        f"[ep {epoch} step {step}] loss={loss.item():.4f} "
                        f"task={l_task.item():.4f} M={l_M.item():.4f} C={l_C.item():.4f}",
                        flush=True,
                    )
                step += 1

            # Eval val
            asp.eval()
            val_acc = _eval_asp_acc(asp, val_loader, vocab.QUERY, device)
            _safe_mlflow("log_metric val_acc", mlflow.log_metric, "val_acc", val_acc, step=step)
            elapsed = time.perf_counter() - t_epoch
            print(
                f"=== Epoch {epoch+1}/{n_epochs} : val_acc={val_acc:.4f} "
                f"(oracle={oracle_acc:.4f}) durée {elapsed:.1f}s ===",
                flush=True,
            )
            improved = val_acc > best_val_acc
            best_val_acc = max(best_val_acc, val_acc)

            # Checkpoint per epoch — robustesse §2 (feedback_script_robustness)
            try:
                state.save_latest(
                    model=asp, optimizer=opt, epoch=epoch, step=step,
                    best_val_acc=best_val_acc, oracle_acc=oracle_acc,
                )
                if improved:
                    state.save_best(
                        model=asp, optimizer=opt, epoch=epoch, step=step,
                        val_acc=val_acc, oracle_acc=oracle_acc,
                    )
                    print(
                        f"[checkpoint] best.pt updated (val_acc={val_acc:.4f})",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                _stderr(
                    f"[checkpoint] save échec : {type(exc).__name__}: {exc}. "
                    f"Continue mais sans persistance epoch {epoch}."
                )
                traceback.print_exc(file=sys.stderr, limit=3)

        # --- Sanity checks ---
        print("\n[sanity] checks ASPLayer (saturation, effondrement, monotonie, lissité)…")
        # Prendre un batch pour les sanity checks
        for batch in val_loader:
            tokens = batch["tokens"].to(device)
            qpos = find_query_positions(tokens, vocab.QUERY)
            break

        def quality_fn(output: torch.Tensor) -> float:
            """Quality via max-logit-prob (proxy d'acc sans cible)."""
            return float(torch.softmax(output, dim=-1).max(dim=-1).values.mean().item())

        # On utilise le bloc 0 pour le sanity (les autres bloocs sont
        # cohérents par construction)
        # Note : les sanity checks attendent une ASPLayer, mais on a un
        # ASPTransformer. On wrappe.
        from phase3_kernel_asp.asp_layer import ASPLayer
        if isinstance(asp.blocks[0].asp, ASPLayer):
            x_emb = asp.tok_embed(tokens) + asp.pos_embed[:tokens.size(1)].unsqueeze(0)
            x_ln = asp.blocks[0].ln1(x_emb)
            sanity = run_all_sanity_checks(
                asp.blocks[0].asp, x_ln,
                quality_fn=quality_fn,
                oracle_quality=oracle_acc,
                saturation_tolerance=float(cfg.sanity.saturation_tolerance),
                smoothness_max_jump=float(cfg.sanity.smoothness_max_jump),
            )
            print(f"  saturation={sanity.saturation_passed} (asp={sanity.saturation_quality:.3f}, oracle={sanity.saturation_oracle:.3f})")
            print(f"  effondrement={sanity.collapse_passed} (diff={sanity.collapse_diff:.2e})")
            print(f"  monotonie={sanity.monotone_passed}")
            print(f"  lissité={sanity.smoothness_passed} (max_jump={sanity.smoothness_max_jump:.3f})")
            sanity_all = (sanity.saturation_passed and sanity.collapse_passed
                          and sanity.monotone_passed and sanity.smoothness_passed)
            _safe_mlflow("log_metric sanity_all", mlflow.log_metric, "sanity_all", float(sanity_all))
            _safe_mlflow("log_metric sanity_saturation", mlflow.log_metric, "sanity_saturation", float(sanity.saturation_passed))
            _safe_mlflow("log_metric sanity_collapse", mlflow.log_metric, "sanity_collapse", float(sanity.collapse_passed))
            _safe_mlflow("log_metric sanity_monotone", mlflow.log_metric, "sanity_monotone", float(sanity.monotone_passed))
            _safe_mlflow("log_metric sanity_smoothness", mlflow.log_metric, "sanity_smoothness", float(sanity.smoothness_passed))

        # --- Verdict ---
        # GO si val_acc ≥ oracle_acc - tolerance ET sanity all passed
        verdict_acc = best_val_acc >= oracle_acc - float(cfg.sanity.saturation_tolerance)
        verdict = verdict_acc  # complet : verdict_acc AND sanity_all si bloc 0 est ASPLayer
        _safe_mlflow("log_metric phase3_verdict", mlflow.log_metric,
                     "phase3_verdict", float(verdict))
        _safe_mlflow("log_metric best_val_acc", mlflow.log_metric,
                     "best_val_acc", best_val_acc)
        finalize_manifest(manifest, duration_s=time.perf_counter() - t0,
                          repo_root=REPO_ROOT)
        print(
            f"\n=== Phase 3 verdict : {'GO' if verdict else 'NO-GO'} | "
            f"best_val_acc={best_val_acc:.4f} (oracle={oracle_acc:.4f}) ===",
            flush=True,
        )

        del oracle, asp
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
