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

Usage (deux modes, mutuellement exclusifs) :

    # Mode A : un seul dump (legacy, dump V1 ou bucket unique)
    PYTHONPATH=CODE uv run python -m phase2_audit_spectral.run \
        --config-path=../../OPS/configs/phase2 --config-name=audit \
        attention_dump_path=PATH

    # Mode B : un dossier de dumps multi-bucket (produit par run_extract.py V2 phase 1)
    PYTHONPATH=CODE uv run python -m phase2_audit_spectral.run \
        --config-path=../../OPS/configs/phase2 --config-name=audit \
        attention_dump_dir=PATH/TO/DIR

Format dump (un fichier .pt) : {"attn": list[(B, H, N, N)], "omegas", "deltas",
"entropies", "tokens"}. En mode dossier, on charge tous les fichiers matchant
`dumps_glob` (défaut `audit_dump_seq*.pt`) et on concatène les exemples dans
l'axe batch ; chaque fichier garde son seq_len propre (les SVD/batteries
n'utilisent jamais N comme dim alignée cross-bucket).
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


def _load_dumps(
    dump_path: str | None,
    dump_dir: str | None,
    dumps_glob: str = "audit_dump_seq*.pt",
) -> list[dict]:
    """Charge un ou plusieurs dumps d'attention.

    Exactement un de `dump_path` / `dump_dir` doit être fourni. Le retour est
    toujours une liste de dicts ; mode single-file = liste à 1 élément. Les
    dumps sont triés (par seq_len si dossier, garantit déterminisme MLflow).
    """
    if dump_path and dump_dir:
        raise SystemExit(
            "Conflit : spécifier soit `attention_dump_path`, soit "
            "`attention_dump_dir`, pas les deux."
        )
    if dump_path:
        return [torch.load(str(dump_path), map_location="cpu", weights_only=False)]
    if dump_dir:
        files = list(Path(dump_dir).glob(dumps_glob))
        if not files:
            raise SystemExit(
                f"Aucun dump matchant `{dumps_glob}` dans {dump_dir}"
            )
        # Tri numérique sur le seq_len extrait du nom (`audit_dump_seq{N}.pt`)
        # → ordre humain naturel et déterminisme MLflow. Fallback tri lex si
        # le pattern ne matche pas.
        import re
        def _seq_key(p: Path) -> tuple[int, str]:
            m = re.search(r"seq(\d+)", p.name)
            return (int(m.group(1)) if m else 10**9, p.name)
        files.sort(key=_seq_key)
        return [
            torch.load(str(f), map_location="cpu", weights_only=False)
            for f in files
        ]
    raise SystemExit(
        "Override requis : `attention_dump_path=path/to/attention_dump.pt` "
        "OU `attention_dump_dir=path/to/dir` (le second pour les dumps "
        "multi-bucket produits par run_extract.py V2 phase 1)."
    )


def _validate_dumps(dumps: list[dict]) -> tuple[int, int]:
    """Vérifie l'invariant cross-dumps (même L, même H). Retourne (L, H)."""
    L = len(dumps[0]["attn"])
    H = dumps[0]["attn"][0].size(1)
    for i, d in enumerate(dumps):
        if len(d["attn"]) != L:
            raise SystemExit(
                f"Dump {i} a {len(d['attn'])} couches, attendu {L} "
                f"(les dumps doivent provenir du même Oracle)"
            )
        h_i = d["attn"][0].size(1)
        if h_i != H:
            raise SystemExit(
                f"Dump {i} a {h_i} têtes, attendu {H} "
                f"(les dumps doivent provenir du même Oracle)"
            )
    return L, H


@hydra.main(version_base=None, config_path="../../OPS/configs/phase2", config_name="audit")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    dumps = _load_dumps(
        dump_path=OmegaConf.select(cfg, "attention_dump_path"),
        dump_dir=OmegaConf.select(cfg, "attention_dump_dir"),
        dumps_glob=OmegaConf.select(cfg, "dumps_glob") or "audit_dump_seq*.pt",
    )
    L, H = _validate_dumps(dumps)
    seq_lens = [d["attn"][0].size(2) for d in dumps]
    print(
        f"Phase 2 : {len(dumps)} dump(s), L={L} couches, H={H} têtes, "
        f"seq_lens={seq_lens}",
        flush=True,
    )

    # Concaténation des axes scalaires (omegas/deltas/entropies) dans la dim batch.
    omegas = np.concatenate([d["omegas"].cpu().numpy() for d in dumps])
    deltas = np.concatenate([d["deltas"].cpu().numpy() for d in dumps])
    entropies = np.concatenate([d["entropies"].cpu().numpy() for d in dumps])
    B = len(omegas)

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
        mlflow.log_params({
            "n_dumps": len(dumps),
            "seq_lens": str(seq_lens),
            "n_examples_total": B,
        })

        # --- 2. SVD batchée (bucket par bucket pour respecter N variable) ---
        r_eff_per_layer_head_example = np.zeros((L, B, H), dtype=np.int64)
        theta_pct = int(cfg.theta * 100)
        b_offset = 0
        for d in dumps:
            B_d = d["attn"][0].size(0)
            for ell, A in enumerate(d["attn"]):
                svd_out = svd_attention(A, theta_values=(0.95, 0.99))
                r_eff_per_layer_head_example[ell, b_offset:b_offset + B_d] = (
                    svd_out[f"r_eff_{theta_pct}"].cpu().numpy()
                )
            b_offset += B_d
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
        # Pour réduire le coût, un seul exemple par régime (ω, Δ) × couche.
        # En multi-bucket on itère sur tous les dumps : un même régime n'existe
        # que dans un seul bucket (les seq_len sont disjoints par construction).
        A_per_regime: dict[tuple, torch.Tensor] = {}
        for d in dumps:
            B_d = d["attn"][0].size(0)
            od = d["omegas"]
            dd = d["deltas"]
            for ell in range(L):
                for b in range(min(B_d, cfg.batteries.max_examples_per_layer)):
                    key = (ell, int(od[b]), int(dd[b]))
                    if key in A_per_regime:
                        continue
                    # moyenne sur les têtes : (H, N, N) → (N, N)
                    A_per_regime[key] = d["attn"][ell][b].mean(dim=0)

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
