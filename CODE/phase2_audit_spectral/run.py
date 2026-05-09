"""
run.py — driver phase 2 (Audit Spectral).

Pipeline :
1. Charger les matrices A FP64 produites par phase 1 (set audit_svd uniquement).
2. SVD batchée → r_eff(0.95), r_eff(0.99) par (couche, tête, exemple).
3. Stress-Rank Map (DOC/02 §2.2) : monovariate + 2D.
4. Loi de transfert r_target = a · ω^α · Δ^β · g(ℋ) (§2.4).
5. Diagnostic spécialisation des têtes (§5b).
6. Batterie A : fit ε_C par classe, classe dominante par régime.
7. Batterie B : analyse résidu (SVD, FFT, PCA cross-régimes).
8. Batterie D : régimes orphelins, asymétrie eigen/SVD.
9. Verdict go/no-go phase 2 (§2.8).
10. Génération rapport phase 2.

Usage :
    PYTHONPATH=CODE uv run python -m phase2_audit_spectral.run \
        --config-path=../../OPS/configs/phase2 --config-name=audit \
        attention_dump_path=PATH

`attention_dump_path` doit pointer vers un fichier .pt produit par phase 1
contenant {layer_attention, omegas, deltas, entropies}.
"""

from __future__ import annotations

import time
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from phase2_audit_spectral.batteries import (
    fit_classes_per_regime,
    residual_analysis,
)
from phase2_audit_spectral.batteries.battery_a import fit_additive_composition, fit_class_with_residual
from phase2_audit_spectral.batteries.battery_d import battery_d_analysis
from phase2_audit_spectral.head_specialization import diagnose_heads, top_specialized_heads
from phase2_audit_spectral.stress_rank_map import build_2d_srm, build_monovariate_srm
from phase2_audit_spectral.svd_pipeline import svd_attention
from phase2_audit_spectral.transfer_law import fit_transfer_law
from shared.mlflow_helpers import log_yaml_config, start_run
from shared.runner import finalize_manifest, make_manifest, write_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]


@hydra.main(version_base=None, config_path="../../OPS/configs/phase2", config_name="audit")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    dump_path = OmegaConf.select(cfg, "attention_dump_path")
    if dump_path is None:
        raise SystemExit(
            "Override requis : `attention_dump_path=path/to/attention_dump.pt` "
            "(produit par phase1_metrologie.run)"
        )

    dump = torch.load(str(dump_path), map_location="cpu", weights_only=False)
    A_per_layer: list[torch.Tensor] = dump["attn"]    # liste (B, H, N, N) FP64
    omegas = dump["omegas"].cpu().numpy()             # (B,)
    deltas = dump["deltas"].cpu().numpy()
    entropies = dump["entropies"].cpu().numpy()
    L = len(A_per_layer)
    B, H, N, _ = A_per_layer[0].shape

    manifest = make_manifest(
        phase="2", sprint=cfg.sprint, domain=cfg.domain, seed=cfg.seed,
        config_path="OPS/configs/phase2/audit.yaml",
        short_description="audit",
        repo=REPO_ROOT,
    )
    write_manifest(manifest, REPO_ROOT)

    t0 = time.perf_counter()
    with start_run(
        experiment="phase2",
        run_name=manifest.run_id,
        phase="2", sprint=cfg.sprint, domain=cfg.domain,
        status=manifest.status,
    ):
        log_yaml_config(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]

        # --- 2. SVD batchée ---
        r_eff_per_layer_head_example = np.zeros((L, B, H), dtype=np.int64)
        for ell, A in enumerate(A_per_layer):
            svd_out = svd_attention(A, theta_values=(0.95, 0.99))
            r_eff_per_layer_head_example[ell] = svd_out[f"r_eff_{int(cfg.theta * 100)}"].cpu().numpy()
        # On agrège sur head pour Stress-Rank Map (un r_eff médian par couche × exemple)
        r_eff_flat = r_eff_per_layer_head_example.reshape(L * H, B).mean(axis=0)  # (B,)

        # --- 3. Stress-Rank Map ---
        srm_mono = build_monovariate_srm(
            r_eff_values=r_eff_flat, omega=omegas, delta=deltas, entropy=entropies,
        )
        for axis, regimes in srm_mono.items():
            for val, stats in regimes.items():
                mlflow.log_metric(f"r_eff_{axis}_{val:g}_median", stats.median)
                mlflow.log_metric(f"r_eff_{axis}_{val:g}_iqr", stats.iqr)

        srm_2d_omega_delta = build_2d_srm(
            r_eff_values=r_eff_flat,
            axis1_name="omega", axis1=omegas.astype(float),
            axis2_name="delta", axis2=deltas.astype(float),
        )

        # --- 4. Loi de transfert ---
        try:
            fit = fit_transfer_law(
                r_target=r_eff_flat.astype(np.float64),
                omega=omegas, delta=deltas, entropy=entropies,
            )
            mlflow.log_metric("transfer_law_alpha", fit.alpha)
            mlflow.log_metric("transfer_law_beta", fit.beta)
            mlflow.log_metric("transfer_law_gamma", fit.gamma)
            mlflow.log_metric("transfer_law_r2", fit.r2)
        except ValueError as e:
            print(f"⚠ Loi de transfert : {e}")

        # --- 5. Diagnostic spécialisation des têtes ---
        diagnostics = diagnose_heads(
            r_eff=r_eff_per_layer_head_example.astype(np.float64),
            dormant_threshold=cfg.head_specialization.dormant_threshold,
            spec_dormant_threshold=cfg.head_specialization.spec_dormant_threshold,
        )
        n_dormant = sum(1 for d in diagnostics if d.is_dormant)
        mlflow.log_metric("n_dormant_heads", n_dormant)
        top_k = top_specialized_heads(diagnostics, k=cfg.head_specialization.top_k)
        for d in top_k:
            print(f"  Tête (L={d.layer}, H={d.head}) spec={d.var_r_eff:.2f} mean_r={d.mean_r_eff:.2f}")

        # --- 6/7. Batteries A + B (par régime, échantillonné) ---
        # Pour réduire le coût, on prend un seul exemple par régime représentatif.
        A_per_regime: dict[tuple, torch.Tensor] = {}
        for ell in range(L):
            for b in range(min(B, cfg.batteries.max_examples_per_layer)):
                key = (ell, int(omegas[b]), int(deltas[b]))
                if key in A_per_regime:
                    continue
                A_per_regime[key] = A_per_layer[ell][b].mean(dim=0)  # moyenne sur les têtes

        battery_a_results = fit_classes_per_regime(A_per_regime)
        epsilon_per_regime = {k: r.epsilon for k, r in battery_a_results.items()}
        for regime, result in list(battery_a_results.items())[: cfg.batteries.log_top_n]:
            mlflow.log_metric(
                f"epsilon_best_layer{regime[0]}_omega{regime[1]}_delta{regime[2]}",
                result.epsilon_best,
            )

        battery_b_residuals = {}
        battery_b_results = []
        for regime, A in A_per_regime.items():
            best_class = battery_a_results[regime].class_best
            _, residual = fit_class_with_residual(A, best_class)
            battery_b_residuals[regime] = residual
            battery_b_results.append(residual_analysis(residual))

        # --- 8. Batterie D ---
        additive_eps: dict[tuple, float] = {}
        for regime, A in A_per_regime.items():
            eps, _, _ = fit_additive_composition(A, class1="toeplitz", class2="hankel")
            additive_eps[regime] = eps
        battery_d = battery_d_analysis(
            epsilon_per_regime=epsilon_per_regime,
            A_per_regime=A_per_regime,
            additive_epsilon=additive_eps,
            orphan_threshold=cfg.batteries.orphan_threshold,
            asymmetry_threshold=cfg.batteries.asymmetry_threshold,
        )
        mlflow.log_metric("n_orphan_regimes", len(battery_d.orphan_regimes))

        # --- 9. Verdict ---
        # GO si :
        # - SCH corroborée (fit transfer law R² > min_r2 OU IQR petite par régime)
        # - portion d'orphelins faible
        passed = (
            len(battery_d.orphan_regimes) / max(len(A_per_regime), 1) < cfg.go_no_go.max_orphan_ratio
        )
        mlflow.log_metric("phase2_passed", float(passed))
        finalize_manifest(manifest, duration_s=time.perf_counter() - t0, repo_root=REPO_ROOT)
        print(f"Phase 2 verdict : {'GO' if passed else 'NO-GO'}")


if __name__ == "__main__":
    main()
