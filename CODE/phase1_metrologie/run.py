"""
run.py — driver Hydra end-to-end pour Phase 1 (RCP Structure-MNIST).

Usage :
    PYTHONPATH=CODE uv run python -m phase1_metrologie.run \
        --config-path=../../OPS/configs/phase1 --config-name=oracle_smnist

Exécute :
1. Construction du SSG (tri-partition train_oracle / audit_svd / init_phase3)
2. Entraînement Oracle (BF16, plateau-based stopping)
3. Extraction matrices A en FP64 sur le set audit_svd
4. Métriques : rang Hankel + entropie spectrale par régime
5. Log MLflow + manifest YAML
"""

from __future__ import annotations

import time
from pathlib import Path

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, ConcatDataset

from phase1_metrologie.metrics.hankel import hankel_rank_numerical
from phase1_metrologie.metrics.spectral import spectral_entropy
from phase1_metrologie.oracle.extract import AttentionExtractor
from phase1_metrologie.oracle.train import TrainConfig, collate, find_query_positions, train_oracle
from phase1_metrologie.oracle.transformer import OracleConfig
from phase1_metrologie.ssg.structure_mnist import (
    SplitConfig,
    StructureMNISTConfig,
    StructureMNISTDataset,
    Vocab,
    split_indices,
    sweep_monovariate,
)
from shared.mlflow_helpers import log_yaml_config, start_run
from shared.runner import finalize_manifest, make_manifest, write_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_sweep_dataset(cfg: DictConfig, vocab: Vocab) -> tuple[ConcatDataset, dict[str, list[StructureMNISTDataset]]]:
    """Construit un ConcatDataset balayant chaque axe ω/Δ/ℋ.

    Retourne aussi un dict {axis_name -> [datasets par valeur]} pour analyse aval.
    """
    base = StructureMNISTConfig(
        omega=cfg.ssg.axes.omega.reference,
        delta=cfg.ssg.axes.delta.reference,
        entropy=cfg.ssg.axes.entropy.reference,
        n_examples=cfg.ssg.n_examples_per_regime,
        n_ops=cfg.ssg.n_ops,
        n_noise=cfg.ssg.n_noise,
        seed=cfg.ssg.seed_train,
    )
    per_axis: dict[str, list[StructureMNISTDataset]] = {}
    all_datasets: list[StructureMNISTDataset] = []
    for axis in ("omega", "delta", "entropy"):
        sweep_values = list(cfg.ssg.axes[axis].sweep)
        per_axis[axis] = []
        for c in sweep_monovariate(axis=axis, values=sweep_values, base=base):
            ds = StructureMNISTDataset(c)
            per_axis[axis].append(ds)
            all_datasets.append(ds)
    return ConcatDataset(all_datasets), per_axis


@hydra.main(version_base=None, config_path="../../OPS/configs/phase1", config_name="oracle_smnist")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    vocab = Vocab(n_ops=cfg.ssg.n_ops, n_noise=cfg.ssg.n_noise)

    # --- Datasets : train, val, audit_svd ---
    sweep_full, _per_axis_train = _build_sweep_dataset(cfg, vocab)

    val_cfg = StructureMNISTConfig(
        omega=cfg.ssg.axes.omega.reference,
        delta=cfg.ssg.axes.delta.reference,
        entropy=cfg.ssg.axes.entropy.reference,
        n_examples=cfg.ssg.n_examples_per_regime // 4,
        n_ops=cfg.ssg.n_ops, n_noise=cfg.ssg.n_noise,
        seed=cfg.ssg.seed_val,
    )
    val_ds = StructureMNISTDataset(val_cfg)

    # tri-partition globale (audit_svd = pour extraction matrices)
    n_total = len(sweep_full)
    split = SplitConfig(
        train_oracle=cfg.dataset_split_smnist.train_oracle,
        audit_svd=cfg.dataset_split_smnist.audit_svd,
        init_phase3=cfg.dataset_split_smnist.init_phase3,
        seed=cfg.dataset_split_smnist.seed,
    )
    parts = split_indices(n_total, split)
    print(f"Tri-partition : train={len(parts['train_oracle'])} audit_svd={len(parts['audit_svd'])} init_phase3={len(parts['init_phase3'])}")

    # --- Modèle ---
    model_cfg = OracleConfig(
        vocab_size=vocab.size,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        d_ff=cfg.model.d_ff,
        max_seq_len=cfg.model.max_seq_len,
        dropout=cfg.model.dropout,
        n_classes=cfg.model.n_classes,
        pad_id=vocab.PAD,
    )
    train_cfg = TrainConfig(
        batch_size=cfg.training.batch_size,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        max_epochs=cfg.training.max_epochs,
        patience=cfg.training.patience,
        plateau_tolerance=cfg.training.plateau_tolerance,
        grad_clip=cfg.training.grad_clip,
        precision=cfg.training.precision,
        num_workers=cfg.training.get("num_workers", 4),
        compile=cfg.training.get("compile", True),
    )

    # --- Manifest + MLflow run ---
    manifest = make_manifest(
        phase="1", sprint=cfg.sprint, domain=cfg.domain, seed=cfg.ssg.seed_train,
        config_path="OPS/configs/phase1/oracle_smnist.yaml",
        short_description="oracle",
        repo=REPO_ROOT,
    )
    write_manifest(manifest, REPO_ROOT)
    print(f"Manifest : OPS/logs/manifests/{manifest.run_id}.yaml — status={manifest.status}")

    t0 = time.perf_counter()
    with start_run(
        experiment="phase1",
        run_name=manifest.run_id,
        phase="1", sprint=cfg.sprint, domain=cfg.domain,
        status=manifest.status,
    ):
        log_yaml_config(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
        mlflow.log_params({"d_model": cfg.model.d_model, "n_heads": cfg.model.n_heads,
                           "n_layers": cfg.model.n_layers, "lr": cfg.training.lr,
                           "batch_size": cfg.training.batch_size})

        from torch.utils.data import Subset
        train_subset = Subset(sweep_full, parts["train_oracle"].tolist())

        def _cb(epoch: int, train_loss: float, val_metrics: dict) -> None:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_metrics["val_loss"], step=epoch)
            mlflow.log_metric("val_acc", val_metrics["val_acc"], step=epoch)

        model, last_metrics = train_oracle(
            model_cfg=model_cfg, train_ds=train_subset, val_ds=val_ds,
            train_cfg=train_cfg, query_id=vocab.QUERY, metric_callback=_cb,
        )

        # --- Extraction sur audit_svd ---
        extractor = AttentionExtractor(model)
        audit_subset = Subset(sweep_full, parts["audit_svd"][: cfg.extraction.batch_size].tolist())
        from functools import partial as _partial
        loader = DataLoader(audit_subset, batch_size=cfg.extraction.batch_size,
                            collate_fn=_partial(collate, pad_id=vocab.PAD),
                            num_workers=2, pin_memory=True)
        for batch in loader:
            tokens = batch["tokens"]
            qpos = find_query_positions(tokens, vocab.QUERY)
            dump = extractor.extract(
                tokens, qpos, batch["targets"], batch["omegas"], batch["deltas"], batch["entropies"]
            )
            break  # un seul batch d'extraction pour le V1 driver

        # --- Métriques ---
        for ell, A_layer in enumerate(dump.attn):
            mlflow.log_metric(f"hankel_rank_layer{ell}_mean",
                              float(hankel_rank_numerical(A_layer, tau=cfg.thresholds_phase1.hankel.tau).item()),
                              step=0)
            mlflow.log_metric(f"spectral_entropy_layer{ell}_mean",
                              float(spectral_entropy(A_layer).mean().item()),
                              step=0)

        finalize_manifest(manifest, duration_s=time.perf_counter() - t0, repo_root=REPO_ROOT)
        print(f"Verdict val_loss={last_metrics.get('val_loss', 'NaN'):.4f} val_acc={last_metrics.get('val_acc', 'NaN'):.3f}")


if __name__ == "__main__":
    main()
