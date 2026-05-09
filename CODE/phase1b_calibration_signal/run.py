"""
run.py — driver Hydra phase 1.5 (Identification Gate).

Spec : DOC/01b. Pipeline :

1. Charger l'Oracle phase 1 (poids fournis via config ou MLflow artifact).
2. Construire le banc hybride (50% SSG variant + 50% bruit).
3. Calibrer le baseline KL global sur séquences-bruit (DOC/01b §8 piège 1).
4. Extraire matrices A sur tout le banc.
5. Calculer S_KL, S_Grad, S_Spectral (DOC/01b §1).
6. Agréger Max-Pool tête + Concat deep-stack (§3).
7. Sous-échantillonnage tokens (§8 piège 3).
8. Spearman + bootstrap IC95 (§4, §8 piège 4).
9. Critères 0.70 / 0.20 → liste signaux retenus.
10. Verdict go/no-go phase 1.5 (§7).

Usage :
    PYTHONPATH=CODE uv run python -m phase1b_calibration_signal.run \
        --config-path=../../OPS/configs/phase1b --config-name=signals \
        oracle_checkpoint=PATH_OR_MLFLOW_URI
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from phase1_metrologie.oracle.extract import AttentionExtractor
from phase1_metrologie.oracle.train import collate, find_query_positions
from phase1_metrologie.oracle.transformer import OracleConfig, OracleTransformer
from phase1_metrologie.ssg.structure_mnist import (
    StructureMNISTConfig,
    StructureMNISTDataset,
    Vocab,
)
from phase1b_calibration_signal.bench.hybrid import HybridBenchConfig, build_hybrid_bench
from phase1b_calibration_signal.bench.spearman import (
    passes_phase1b_criteria,
    signal_correlations,
)
from phase1b_calibration_signal.signals.aggregation import aggregate_signal_per_token
from phase1b_calibration_signal.signals.s_kl import GlobalKLBaseline, compute_s_kl
from phase1b_calibration_signal.signals.s_spectral import compute_s_spectral
from shared.mlflow_helpers import log_yaml_config, start_run
from shared.runner import finalize_manifest, make_manifest, write_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_oracle(checkpoint_path: str, vocab: Vocab) -> OracleTransformer:
    """Charge l'Oracle. Le checkpoint contient le state_dict + cfg sérialisée."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Reconstitue OracleConfig depuis le state ; en V1 on suppose le caller
    # passe les bons hyperparamètres via config Hydra.
    raise NotImplementedError(
        "TODO sur le pod : charger l'Oracle via MLflow artifact ou state_dict checkpoint."
    )


def _calibrate_kl_baseline(
    oracle: OracleTransformer, vocab: Vocab, *, n_examples: int, seed: int, batch_size: int = 16
) -> GlobalKLBaseline:
    """Calibre le baseline KL sur séquences sans structure (ω=0, Δ=0, ℋ étalée)."""
    cfg = StructureMNISTConfig(
        omega=0, delta=0, entropy=0.5, n_examples=n_examples,
        n_ops=vocab.n_ops, n_noise=vocab.n_noise, seed=seed,
    )
    ds = StructureMNISTDataset(cfg)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate)

    extractor = AttentionExtractor(oracle)
    accumulators: list[torch.Tensor] | None = None  # par couche
    n_seen = 0
    for batch in loader:
        tokens = batch["tokens"]
        qpos = find_query_positions(tokens, vocab.QUERY)
        dump = extractor.extract(
            tokens, qpos, batch["targets"], batch["omegas"], batch["deltas"], batch["entropies"]
        )
        if accumulators is None:
            accumulators = [a.clone() for a in dump.attn]
        else:
            for i, a in enumerate(dump.attn):
                # taille variable possible si seq_len change ; on assume seq_len fixe
                accumulators[i] += a
        n_seen += 1
    assert accumulators is not None
    averaged = [a / max(n_seen, 1) for a in accumulators]
    return GlobalKLBaseline.from_attention_dumps(averaged)


def _compute_signals_on_bench(
    *,
    oracle: OracleTransformer,
    bench_loader: DataLoader,
    vocab: Vocab,
    s_spectral_K: int,
    s_spectral_tau: float,
    baseline: GlobalKLBaseline,
) -> dict[str, np.ndarray]:
    """Pour chaque token du banc, calcule (S_KL, S_Spectral) agrégés (B, L, N)
    puis aplatit en vecteur 1D + collecte stress (omega, delta, entropy)."""
    extractor = AttentionExtractor(oracle)
    s_kl_all: list[np.ndarray] = []
    s_spectral_all: list[np.ndarray] = []
    omega_all: list[np.ndarray] = []
    delta_all: list[np.ndarray] = []
    entropy_all: list[np.ndarray] = []

    for batch in bench_loader:
        tokens = batch["tokens"]
        qpos = find_query_positions(tokens, vocab.QUERY)
        dump = extractor.extract(
            tokens, qpos, batch["targets"], batch["omegas"], batch["deltas"], batch["entropies"]
        )
        skl = compute_s_kl(dump.attn, baseline)         # (L, B, H, N)
        sspec = compute_s_spectral(dump.attn, K=s_spectral_K, tau=s_spectral_tau)
        skl_agg = aggregate_signal_per_token(skl)       # (B, L, N)
        sspec_agg = aggregate_signal_per_token(sspec)
        # Pour Spearman : flatten (B, L, N) -> (B*L*N,) avec stress par token
        B, L, N = skl_agg.shape
        # On utilise le max sur L pour réduire à 1 valeur par token
        skl_flat = skl_agg.max(dim=1).values.cpu().numpy().reshape(-1)
        sspec_flat = sspec_agg.max(dim=1).values.cpu().numpy().reshape(-1)
        # Stress : broadcast par token (chaque token de l'exemple b a le même stress)
        omega_token = batch["omegas"].cpu().numpy()[:, None].repeat(N, axis=1).reshape(-1)
        delta_token = batch["deltas"].cpu().numpy()[:, None].repeat(N, axis=1).reshape(-1)
        entropy_token = batch["entropies"].cpu().numpy()[:, None].repeat(N, axis=1).reshape(-1)
        s_kl_all.append(skl_flat)
        s_spectral_all.append(sspec_flat)
        omega_all.append(omega_token)
        delta_all.append(delta_token)
        entropy_all.append(entropy_token)

    return {
        "S_KL": np.concatenate(s_kl_all),
        "S_Spectral": np.concatenate(s_spectral_all),
        "omega": np.concatenate(omega_all),
        "delta": np.concatenate(delta_all),
        "entropy": np.concatenate(entropy_all),
    }


def _subsample(arr: np.ndarray, every_k: int, seed: int) -> np.ndarray:
    """Sous-échantillonne tous les k tokens (DOC/01b §8 piège 3)."""
    rng = np.random.default_rng(seed)
    n = arr.size
    idx = np.arange(0, n, every_k)
    if idx.size == 0:
        return arr
    return arr[idx]


@hydra.main(version_base=None, config_path="../../OPS/configs/phase1b", config_name="signals")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    vocab = Vocab(n_ops=4, n_noise=8)

    # --- Manifest + run ---
    manifest = make_manifest(
        phase="1.5", sprint=cfg.sprint, domain=cfg.domain, seed=cfg.bench.seed,
        config_path="OPS/configs/phase1b/signals.yaml",
        short_description="signals",
        repo=REPO_ROOT,
    )
    write_manifest(manifest, REPO_ROOT)

    t0 = time.perf_counter()
    with start_run(
        experiment="phase1.5",
        run_name=manifest.run_id,
        phase="1.5", sprint=cfg.sprint, domain=cfg.domain,
        status=manifest.status,
    ):
        log_yaml_config(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]

        # --- Oracle ---
        # NB sur le pod : le chemin du checkpoint est passé via override Hydra
        # ou récupéré depuis MLflow. Driver V1 : passer `oracle_checkpoint=...`
        oracle_checkpoint = OmegaConf.select(cfg, "oracle_checkpoint")
        if oracle_checkpoint is None:
            raise SystemExit(
                "Oracle checkpoint requis. Override Hydra : "
                "`oracle_checkpoint=path/to/ckpt.pt`"
            )
        oracle = _load_oracle(str(oracle_checkpoint), vocab)
        oracle.eval()

        # --- Calibration baseline ---
        baseline = _calibrate_kl_baseline(
            oracle, vocab,
            n_examples=cfg.s_kl.baseline.n_calibration_examples,
            seed=cfg.s_kl.baseline.seed,
        )

        # --- Banc hybride ---
        bench_cfg = HybridBenchConfig(
            n_examples=cfg.bench.n_examples,
            seed=cfg.bench.seed,
            seq_len=cfg.bench.seq_len,
            structured_omegas=tuple(cfg.bench.structured_omegas),
            structured_deltas=tuple(cfg.bench.structured_deltas),
            structured_entropy_max=cfg.bench.structured_entropy_max,
            noise_entropies=tuple(cfg.bench.noise_entropies),
        )
        bench_ds = build_hybrid_bench(bench_cfg)
        bench_loader = DataLoader(bench_ds, batch_size=8, collate_fn=collate)

        # --- Signaux + Spearman ---
        results = _compute_signals_on_bench(
            oracle=oracle, bench_loader=bench_loader, vocab=vocab,
            s_spectral_K=cfg.s_spectral.K, s_spectral_tau=cfg.s_spectral.tau,
            baseline=baseline,
        )
        every_k = cfg.subsample.every_k_tokens
        seed = cfg.subsample.seed
        signals = {
            "S_KL": _subsample(results["S_KL"], every_k, seed),
            "S_Spectral": _subsample(results["S_Spectral"], every_k, seed),
        }
        stress = {
            "omega": _subsample(results["omega"], every_k, seed),
            "delta": _subsample(results["delta"], every_k, seed),
            "entropy": _subsample(results["entropy"], every_k, seed),
        }
        correlations = signal_correlations(
            signals=signals, stress=stress,
            n_boot=cfg.bootstrap.n_boot, seed=cfg.bench.seed,
        )

        for (s_name, axis), result in correlations.items():
            mlflow.log_metric(f"rho_{s_name}_{axis}", result.rho)
            mlflow.log_metric(f"rho_{s_name}_{axis}_ci_low", result.ci_low)
            mlflow.log_metric(f"rho_{s_name}_{axis}_ci_high", result.ci_high)

        verdict = passes_phase1b_criteria(
            correlations,
            threshold_structural=cfg.thresholds_phase1b.selection.threshold_structural,
            threshold_noise=cfg.thresholds_phase1b.selection.threshold_noise,
        )
        retained = [s for s, v in verdict.items() if v["passed"]]
        mlflow.log_param("retained_signals", ",".join(retained) or "NONE")
        mlflow.log_metric("phase1b_passed", float(len(retained) > 0))

        finalize_manifest(manifest, duration_s=time.perf_counter() - t0, repo_root=REPO_ROOT)

        if retained:
            print(f"GO. Signaux retenus : {retained}")
        else:
            print("NO-GO. Aucun signal ne passe les critères. Arrêt du protocole.")


if __name__ == "__main__":
    main()
