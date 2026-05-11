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


def _make_padded_collate(pad_id: int):
    """Retourne un collate_fn fixé sur le pad_id du Vocab."""
    from functools import partial
    return partial(collate, pad_id=pad_id)


def _load_oracle(
    checkpoint_path: str,
    vocab: Vocab,
    *,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
    dropout: float,
    max_seq_len: int,
    n_classes: int,
    device: str = "cuda",
) -> OracleTransformer:
    """Charge l'Oracle depuis un checkpoint Fabric (state_dict + meta).

    Le checkpoint a la forme {"model": state_dict, "epoch": int, "best_val_loss": float}
    (cf. phase1_metrologie.oracle.train.train_oracle). Les hyperparamètres
    du modèle ne sont pas dans le checkpoint — ils sont fournis via la config
    Hydra et doivent correspondre EXACTEMENT à ceux utilisés à l'entraînement
    sinon load_state_dict échoue.
    """
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state

    cfg = OracleConfig(
        vocab_size=vocab.size,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
        dropout=dropout, max_seq_len=max_seq_len, n_classes=n_classes,
        pad_id=vocab.PAD,
    )
    model = OracleTransformer(cfg)
    # state_dict peut avoir des préfixes : "_forward_module." (Fabric) et/ou
    # "_orig_mod." (torch.compile). On strip les deux si présents.
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k.removeprefix("_forward_module.").removeprefix("_orig_mod.")
        cleaned[new_k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        print(f"AVERT load_state_dict : missing={len(missing)} keys, unexpected={len(unexpected)} keys")
        if missing[:3]:
            print(f"  premiers missing : {missing[:3]}")
        if unexpected[:3]:
            print(f"  premiers unexpected : {unexpected[:3]}")
    model = model.to(device)
    return model


def _calibrate_kl_baseline(
    oracle: OracleTransformer, vocab: Vocab, *, n_examples: int, seed: int,
    omega_max: int = 0, delta_max: int = 0, batch_size: int = 16,
) -> GlobalKLBaseline:
    """Calibre le baseline KL sur séquences-bruit.

    Si bench seq_len variable (omega_max/delta_max > 0), on génère le bruit
    à seq_len max (formule SSG : 1 + (2+2δ)·ω + 1 + δ + 1) avec entropy=1.0
    (bruit pur, ℋ saturée). À l'usage, compute_s_kl slice la baseline au
    seq_len du batch + renormalise. Cf. carnet 2026-05-11 décision option C.

    Si omega_max=0 et delta_max=0 (défaut historique), on reste sur ω=0, Δ=0,
    entropy=0.5 (comportement V1 pour bench à seq_len fixe).
    """
    if omega_max > 0 or delta_max > 0:
        cfg = StructureMNISTConfig(
            omega=omega_max, delta=delta_max, entropy=1.0, n_examples=n_examples,
            n_ops=vocab.n_ops, n_noise=vocab.n_noise, seed=seed,
        )
    else:
        cfg = StructureMNISTConfig(
            omega=0, delta=0, entropy=0.5, n_examples=n_examples,
            n_ops=vocab.n_ops, n_noise=vocab.n_noise, seed=seed,
        )
    ds = StructureMNISTDataset(cfg)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=_make_padded_collate(vocab.PAD))

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
    baseline: GlobalKLBaseline | None,
    enable_s_kl: bool = True,
    enable_s_spectral: bool = True,
) -> dict[str, np.ndarray]:
    """Pour chaque token du banc, calcule les signaux activés agrégés (B, L, N)
    puis aplatit en vecteur 1D + collecte stress (omega, delta, entropy).

    NB V1 : S_KL nécessite que la calibration baseline et le bench utilisent
    le même seq_len, sinon les distributions ne broadcast pas. Si seq_len
    variable dans le bench, désactiver S_KL via `enable_s_kl=False`.
    """
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
        skl_flat = None
        if enable_s_kl and baseline is not None:
            skl = compute_s_kl(dump.attn, baseline)         # (L, B, H, N)
            skl_agg = aggregate_signal_per_token(skl)       # (B, L, N)
            B, L, N_skl = skl_agg.shape
            skl_flat = skl_agg.max(dim=1).values.cpu().numpy().reshape(-1)

        sspec_flat = None
        if enable_s_spectral:
            sspec = compute_s_spectral(dump.attn, K=s_spectral_K, tau=s_spectral_tau)
            sspec_agg = aggregate_signal_per_token(sspec)
            B, L, N = sspec_agg.shape
            sspec_flat = sspec_agg.max(dim=1).values.cpu().numpy().reshape(-1)
        else:
            B, _, _, N = dump.attn[0].shape

        omega_token = batch["omegas"].cpu().numpy()[:, None].repeat(N, axis=1).reshape(-1)
        delta_token = batch["deltas"].cpu().numpy()[:, None].repeat(N, axis=1).reshape(-1)
        entropy_token = batch["entropies"].cpu().numpy()[:, None].repeat(N, axis=1).reshape(-1)
        if skl_flat is not None:
            s_kl_all.append(skl_flat)
        if sspec_flat is not None:
            s_spectral_all.append(sspec_flat)
        omega_all.append(omega_token)
        delta_all.append(delta_token)
        entropy_all.append(entropy_token)

    out: dict[str, np.ndarray] = {
        "omega": np.concatenate(omega_all),
        "delta": np.concatenate(delta_all),
        "entropy": np.concatenate(entropy_all),
    }
    if s_kl_all:
        out["S_KL"] = np.concatenate(s_kl_all)
    if s_spectral_all:
        out["S_Spectral"] = np.concatenate(s_spectral_all)
    return out


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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        oracle = _load_oracle(
            str(oracle_checkpoint), vocab,
            d_model=cfg.model.d_model, n_heads=cfg.model.n_heads,
            n_layers=cfg.model.n_layers, d_ff=cfg.model.d_ff,
            dropout=cfg.model.dropout, max_seq_len=cfg.model.max_seq_len,
            n_classes=cfg.model.n_classes, device=device,
        )
        oracle.eval()
        print(f"Oracle chargé depuis {oracle_checkpoint} sur device {device}", flush=True)

        # --- Calibration baseline (uniquement si S_KL activé) ---
        # Adaptation seq_len variable (carnet 2026-05-11 option C) : la baseline
        # est calibrée à seq_len max (ω_max, δ_max du bench, entropy=1.0 = bruit
        # pur). compute_s_kl slice + renormalise à seq_len du batch.
        enable_s_kl = bool(OmegaConf.select(cfg, "s_kl.enabled", default=False))
        enable_s_spectral = bool(OmegaConf.select(cfg, "s_spectral.enabled", default=True))
        baseline = None
        if enable_s_kl:
            omega_max = max(cfg.bench.structured_omegas) if cfg.bench.structured_omegas else 0
            delta_max = max(cfg.bench.structured_deltas) if cfg.bench.structured_deltas else 0
            baseline = _calibrate_kl_baseline(
                oracle, vocab,
                n_examples=cfg.s_kl.baseline.n_calibration_examples,
                seed=cfg.s_kl.baseline.seed,
                omega_max=omega_max, delta_max=delta_max,
            )
            print(
                f"Baseline KL calibrée sur {cfg.s_kl.baseline.n_calibration_examples} exemples "
                f"(ω_max={omega_max}, δ_max={delta_max}, entropy=1.0)", flush=True
            )
        else:
            print("S_KL désactivé pour ce run (s_kl.enabled=false).", flush=True)

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
        # batch_size petit car capture_attn=True matérialise A en N² × layers × heads
        # peak mémoire ≈ batch × 6 × 8 × N² × 4 bytes ; pour N=4100, batch=2 → ~6.5 GB ok
        bench_batch = OmegaConf.select(cfg, "bench.batch_size", default=2)
        bench_loader = DataLoader(bench_ds, batch_size=bench_batch, collate_fn=_make_padded_collate(vocab.PAD))

        # --- Signaux + Spearman ---
        results = _compute_signals_on_bench(
            oracle=oracle, bench_loader=bench_loader, vocab=vocab,
            s_spectral_K=cfg.s_spectral.K, s_spectral_tau=cfg.s_spectral.tau,
            baseline=baseline,
            enable_s_kl=enable_s_kl, enable_s_spectral=enable_s_spectral,
        )
        every_k = cfg.subsample.every_k_tokens
        seed = cfg.subsample.seed
        signals = {}
        for sname in ("S_KL", "S_Spectral"):
            if sname in results:
                signals[sname] = _subsample(results[sname], every_k, seed)
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
