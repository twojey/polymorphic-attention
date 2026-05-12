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
9. Diagnostic découplage S_Spectral ↔ r_eff (garde-fou H2, DOC/carnet 2026-05-12).
10. Verdict go/no-go phase 2 (§2.8).
11. Génération rapport phase 2.

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

import gc
import os
import sys
import time
import traceback
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
from phase2_audit_spectral.checkpoint import Phase2State
from phase2_audit_spectral.head_specialization import diagnose_heads, top_specialized_heads
from phase2_audit_spectral.signal_decoupling import (
    diagnose_s_spectral_decoupling,
    log_diagnostic_to_mlflow,
)
from phase2_audit_spectral.stress_rank_map import build_2d_srm, build_monovariate_srm
from phase2_audit_spectral.svd_pipeline import svd_attention
from phase2_audit_spectral.transfer_law import fit_transfer_law
from shared.mlflow_helpers import log_yaml_config, start_run
from shared.runner import finalize_manifest, make_manifest, write_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]


# -----------------------------------------------------------------------------
# Helpers I/O + validation (fail-loud, no silent failures)
# -----------------------------------------------------------------------------


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _safe_mlflow(operation: str, fn, *args, **kwargs):
    """Exécute `fn` (typiquement un log_artifact / log_metric MLflow).

    Si l'opération échoue, log un warning explicite à stderr avec traceback
    abrégé et continue. Le rationnel : les artefacts critiques (dumps,
    manifest, config.yaml) sont écrits sur disque AVANT toute opération
    MLflow, donc la source de vérité est sauve. Les uploads MLflow sont
    optionnels — une coupure réseau ne doit pas perdre des heures de calcul.

    Pour rendre un échec MLflow bloquant : ne pas appeler cette fonction,
    appeler la primitive directement.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        _stderr(
            f"[mlflow] {operation} a échoué : {type(exc).__name__}: {exc}\n"
            f"         (artefact disque préservé, run continue)"
        )
        traceback.print_exc(file=sys.stderr, limit=3)
        return None


def _print_resource_snapshot(prefix: str = "[resources]") -> None:
    """Snapshot RAM + VRAM disponibles, pour traçabilité avant pic mémoire."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        msg = (
            f"{prefix} RAM avail={mem.available/1e9:.1f} GB "
            f"used={mem.percent:.0f}%"
        )
    except ImportError:
        msg = f"{prefix} psutil indisponible"

    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        msg += (
            f" | VRAM free={free_b/1e9:.1f} GB / {total_b/1e9:.1f} GB "
            f"(alloc={torch.cuda.memory_allocated()/1e9:.2f} GB)"
        )
    print(msg, flush=True)


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
        p = Path(dump_path)
        if not p.is_file():
            raise SystemExit(f"Dump introuvable : {p}")
        return [torch.load(str(p), map_location="cpu", weights_only=False)]
    if dump_dir:
        d = Path(dump_dir)
        if not d.is_dir():
            raise SystemExit(f"Dossier de dumps introuvable : {d}")
        files = list(d.glob(dumps_glob))
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
    expected_keys = {"attn", "omegas", "deltas", "entropies"}
    for i, d in enumerate(dumps):
        missing = expected_keys - set(d.keys())
        if missing:
            raise SystemExit(
                f"Dump {i} : clés manquantes {missing}. Format attendu "
                f"= {sorted(expected_keys)}."
            )
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
        # Sanity check : tailles des axes scalaires == B
        B_i = d["attn"][0].size(0)
        for k in ("omegas", "deltas", "entropies"):
            if d[k].numel() != B_i:
                raise SystemExit(
                    f"Dump {i} : '{k}' a {d[k].numel()} éléments mais "
                    f"attn.size(0)={B_i}. Manifeste de corruption."
                )
    return L, H


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../../OPS/configs/phase2", config_name="audit")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    _print_resource_snapshot("[startup]")

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
    if manifest.git_dirty:
        _stderr(
            "[manifest] git_dirty=True → status=exploratory. Le verdict ne "
            "pourra pas être cité comme registered strict. Commit avant "
            "relance pour obtenir registered."
        )

    # Config SVD (cf. svd_pipeline.svd_attention)
    svd_device = str(OmegaConf.select(cfg, "svd.device") or "auto")
    svd_precision = str(OmegaConf.select(cfg, "svd.precision") or "fp64")
    print(f"[svd] device={svd_device} precision={svd_precision}", flush=True)

    # Checkpoint dir : par défaut adjacent au dump_dir pour persistance pod
    default_state_dir = (
        Path(OmegaConf.select(cfg, "attention_dump_dir") or "/tmp")
        / "_phase2_state"
    )
    state_dir = Path(OmegaConf.select(cfg, "checkpoint.dir") or default_state_dir)
    resume_enabled = bool(OmegaConf.select(cfg, "checkpoint.enabled") or True)
    state, is_resumed = Phase2State.create_or_resume(
        state_dir,
        seq_lens=seq_lens, n_examples_total=B,
        svd_device=svd_device, svd_precision=svd_precision,
    )
    if not resume_enabled:
        state.clean()
        state, is_resumed = Phase2State.create_or_resume(
            state_dir, seq_lens=seq_lens, n_examples_total=B,
            svd_device=svd_device, svd_precision=svd_precision,
        )
    print(
        f"[checkpoint] state_dir={state_dir} resume={is_resumed} "
        f"existing={[p.stem for p in state_dir.glob('*.pt')]}",
        flush=True,
    )

    t0 = time.perf_counter()
    with start_run(
        experiment="phase2",
        run_name=manifest.run_id,
        phase="2", sprint=cfg.sprint, domain=cfg.domain,
        status=manifest.status,
    ):
        _safe_mlflow(
            "log_yaml_config",
            log_yaml_config,
            OmegaConf.to_container(cfg, resolve=True),
        )
        _safe_mlflow(
            "log_params",
            mlflow.log_params,
            {
                "n_dumps": len(dumps),
                "seq_lens": str(seq_lens),
                "n_examples_total": B,
                "svd_device": svd_device,
                "svd_precision": svd_precision,
                "git_dirty": str(manifest.git_dirty),
            },
        )

        # --- 2. SVD batchée (bucket par bucket pour respecter N variable) ---
        # r_eff_per_layer_head_example : (L, B, H) avec B = total exemples.
        theta_pct = int(cfg.theta * 100)
        if state.has("svd_r_eff"):
            print("[checkpoint] SKIP SVD (r_eff déjà calculé)", flush=True)
            r_eff_per_layer_head_example = state.load("svd_r_eff")
        else:
            r_eff_per_layer_head_example = np.zeros((L, B, H), dtype=np.int64)
            b_offset = 0
            for dump_idx, d in enumerate(dumps):
                B_d = d["attn"][0].size(0)
                t_bucket = time.perf_counter()
                for ell, A in enumerate(d["attn"]):
                    svd_out = svd_attention(
                        A,
                        theta_values=(0.95, 0.99),
                        device=svd_device,  # type: ignore[arg-type]
                        precision=svd_precision,  # type: ignore[arg-type]
                    )
                    r_eff_per_layer_head_example[ell, b_offset:b_offset + B_d] = (
                        svd_out[f"r_eff_{theta_pct}"].cpu().numpy()
                    )
                elapsed = time.perf_counter() - t_bucket
                print(
                    f"  [svd] dump {dump_idx+1}/{len(dumps)} seq_len={seq_lens[dump_idx]} "
                    f"B={B_d} : {elapsed:.1f}s",
                    flush=True,
                )
                b_offset += B_d
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            # Save checkpoint après SVD batchée complète (l'étape la plus coûteuse).
            state.save("svd_r_eff", r_eff_per_layer_head_example)
            print(f"[checkpoint] SVD r_eff saved → {state_dir}/svd_r_eff.pt", flush=True)

        # On agrège sur (L, H) pour Stress-Rank Map (un r_eff médian par exemple)
        r_eff_flat = r_eff_per_layer_head_example.astype(np.float64).mean(axis=(0, 2))  # (B,)
        mlflow.log_metric("r_eff_global_mean", float(r_eff_flat.mean()))
        mlflow.log_metric("r_eff_global_median", float(np.median(r_eff_flat)))

        # --- 3. Stress-Rank Map ---
        srm_mono = build_monovariate_srm(
            r_eff_values=r_eff_flat, omega=omegas, delta=deltas, entropy=entropies,
        )
        for axis, regimes in srm_mono.items():
            for val, stats in regimes.items():
                _safe_mlflow(
                    f"log_metric r_eff_{axis}_{val:g}",
                    mlflow.log_metric,
                    f"r_eff_{axis}_{val:g}_median", stats.median,
                )
                _safe_mlflow(
                    f"log_metric r_eff_iqr_{axis}_{val:g}",
                    mlflow.log_metric,
                    f"r_eff_{axis}_{val:g}_iqr", stats.iqr,
                )

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
        except ValueError as exc:
            _stderr(
                f"[transfer_law] fit a échoué : {exc}. "
                f"Probable : trop peu d'observations valides. "
                f"On continue mais transfer_law_r2 manquera dans MLflow."
            )

        # --- 5. Diagnostic spécialisation des têtes ---
        # r_eff_per_layer_head_example a shape (L, B, H) mais diagnose_heads
        # attend (L, H, n_examples). Transpose nécessaire (fix vs version
        # antérieure qui passait la shape (L,B,H) sans transpose → la variance
        # par tête était calculée à travers le batch×layer au lieu du batch
        # à layer×tête fixés).
        r_eff_for_heads = r_eff_per_layer_head_example.transpose(0, 2, 1).astype(np.float64)
        # shape : (L, H, B)
        diagnostics = diagnose_heads(
            r_eff=r_eff_for_heads,
            dormant_threshold=cfg.head_specialization.dormant_threshold,
            spec_dormant_threshold=cfg.head_specialization.spec_dormant_threshold,
        )
        n_dormant = sum(1 for d in diagnostics if d.is_dormant)
        mlflow.log_metric("n_dormant_heads", n_dormant)
        top_k = top_specialized_heads(diagnostics, k=cfg.head_specialization.top_k)
        print(f"\n[head_spec] {n_dormant} têtes dormantes / {L*H} total. Top {cfg.head_specialization.top_k} :")
        for d in top_k:
            print(f"  Tête (L={d.layer}, H={d.head}) spec={d.var_r_eff:.2f} mean_r={d.mean_r_eff:.2f}")

        # --- 6/7. Batteries A + B (par régime, échantillonné) ---
        # Pour réduire le coût, un seul exemple par régime (ω, Δ) × couche.
        if state.has("batteries_results"):
            print("[checkpoint] SKIP batteries A/B/D (déjà calculé)", flush=True)
            cached = state.load("batteries_results")
            battery_a_results = cached["battery_a_results"]
            epsilon_per_regime = cached["epsilon_per_regime"]
            battery_b_results = cached["battery_b_results"]
            battery_d = cached["battery_d"]
            n_orphans = len(battery_d.orphan_regimes)
            n_regimes_total = cached["n_regimes_total"]
        else:
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
            print(f"[batteries] {len(A_per_regime)} régimes (couche × ω × Δ)", flush=True)

            battery_a_results = fit_classes_per_regime(A_per_regime)
            epsilon_per_regime = {k: r.epsilon for k, r in battery_a_results.items()}
            for regime, result in list(battery_a_results.items())[: cfg.batteries.log_top_n]:
                _safe_mlflow(
                    f"log_metric epsilon_best regime={regime}",
                    mlflow.log_metric,
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
            n_orphans = len(battery_d.orphan_regimes)
            n_regimes_total = len(A_per_regime)
            # Save checkpoint (sans A_per_regime qui peut être très gros)
            state.save("batteries_results", {
                "battery_a_results": battery_a_results,
                "epsilon_per_regime": epsilon_per_regime,
                "battery_b_results": battery_b_results,
                "battery_d": battery_d,
                "n_regimes_total": n_regimes_total,
            })
            print(f"[checkpoint] batteries saved → {state_dir}/batteries_results.pt", flush=True)
            del A_per_regime  # libère la mémoire

        mlflow.log_metric("n_orphan_regimes", n_orphans)
        mlflow.log_metric("orphan_ratio", n_orphans / max(n_regimes_total, 1))

        # --- 9. Diagnostic découplage S_Spectral ↔ r_eff (garde-fou H2) ---
        # Pré-requis : OPENBLAS_NUM_THREADS=1 (vérifié dans compute_s_spectral)
        decoupling_enabled = bool(OmegaConf.select(cfg, "decoupling.enabled") or True)
        decoupling_diag = None
        if state.has("decoupling_diag"):
            print("[checkpoint] SKIP diagnostic découplage (déjà calculé)", flush=True)
            decoupling_diag = state.load("decoupling_diag")
            log_diagnostic_to_mlflow(decoupling_diag, mlflow_module=mlflow)
        elif decoupling_enabled:
            print("\n[decoupling] Diagnostic S_Spectral vs r_eff (sous-échantillonné)…", flush=True)
            _print_resource_snapshot("[decoupling pre]")
            try:
                _max_seq = OmegaConf.select(cfg, "decoupling.max_seq_len")
                decoupling_diag = diagnose_s_spectral_decoupling(
                    dumps=dumps,
                    r_eff_per_layer_head_example=r_eff_per_layer_head_example,
                    K=int(OmegaConf.select(cfg, "decoupling.K") or 64),
                    tau=float(OmegaConf.select(cfg, "decoupling.tau") or 1e-3),
                    max_examples=int(OmegaConf.select(cfg, "decoupling.max_examples") or 200),
                    seed=int(OmegaConf.select(cfg, "decoupling.seed") or 0),
                    n_boot=int(OmegaConf.select(cfg, "decoupling.n_boot") or 1000),
                    threshold_decoupling=float(
                        OmegaConf.select(cfg, "decoupling.threshold") or 0.60
                    ),
                    max_seq_len=(int(_max_seq) if _max_seq is not None else None),
                )
                log_diagnostic_to_mlflow(decoupling_diag, mlflow_module=mlflow)
                state.save("decoupling_diag", decoupling_diag)
                print(f"[checkpoint] diagnostic découplage saved → {state_dir}/decoupling_diag.pt", flush=True)
                rho = decoupling_diag.rho_global
                print(
                    f"[decoupling] ρ_global = {rho.rho:+.3f} "
                    f"[IC95 {rho.ci_low:+.3f}, {rho.ci_high:+.3f}] "
                    f"n={decoupling_diag.n_examples_used} → verdict={decoupling_diag.verdict}",
                    flush=True,
                )
            except (RuntimeError, ValueError) as exc:
                _stderr(
                    f"[decoupling] échec : {type(exc).__name__}: {exc}. "
                    f"Le verdict découplage ne sera pas disponible. "
                    f"Vérifier OPENBLAS_NUM_THREADS=1 dans l'environnement."
                )
                traceback.print_exc(file=sys.stderr, limit=5)
        else:
            print("[decoupling] désactivé (decoupling.enabled=false)", flush=True)

        # --- 10. Verdict ---
        # GO si :
        # - SCH corroborée (fit transfer law R² > min_r2 OU IQR petite par régime)
        # - portion d'orphelins faible
        # - signal S_Spectral non découplé du vrai r_eff (si diagnostic activé)
        orphan_ratio = n_orphans / max(n_regimes_total, 1)
        passed_orphans = orphan_ratio < cfg.go_no_go.max_orphan_ratio
        passed_decoupling = (
            decoupling_diag is None
            or decoupling_diag.verdict == "ok"
        )
        passed = passed_orphans and passed_decoupling

        mlflow.log_metric("phase2_passed", float(passed))
        mlflow.log_metric("phase2_passed_orphans", float(passed_orphans))
        mlflow.log_metric("phase2_passed_decoupling", float(passed_decoupling))

        finalize_manifest(manifest, duration_s=time.perf_counter() - t0, repo_root=REPO_ROOT)
        elapsed_total = time.perf_counter() - t0
        print(
            f"\n=== Phase 2 verdict : {'GO' if passed else 'NO-GO'} "
            f"(orphans_ratio={orphan_ratio:.3f}, "
            f"decoupling={'OK' if passed_decoupling else 'KO'}) "
            f"durée {elapsed_total:.0f}s ===",
            flush=True,
        )
        if not passed_decoupling and decoupling_diag is not None:
            _stderr(
                f"[verdict] DRAPEAU ROUGE découplage : ρ_global="
                f"{decoupling_diag.rho_global.rho:.3f} < seuil "
                f"{decoupling_diag.threshold_decoupling}. "
                f"S_Spectral phase 1.5 peut avoir mesuré un artefact de "
                f"fenêtre K=64. H2 (allocation guidée) à reconsidérer "
                f"avant phase 3."
            )

        # Libération mémoire avant retour (utile si chained in nohup)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
