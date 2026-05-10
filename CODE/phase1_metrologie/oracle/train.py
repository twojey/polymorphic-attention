"""
train.py — boucle d'entraînement de l'Oracle (Lightning Fabric).

Critère d'arrêt : plateau de loss validation (cf. DOC/01 §8.3). Pour V1
on simplifie : un patience N époques sans amélioration > tolérance arrête.

Précision : BF16 mixed-precision à l'entraînement (cf. DOC/01 §8.4).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from lightning.fabric import Fabric
from torch import nn
from torch.utils.data import DataLoader

from phase1_metrologie.oracle.transformer import OracleConfig, OracleTransformer
from phase1_metrologie.ssg.structure_mnist import StructureMNISTDataset, StructureMNISTSample


@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.0
    max_epochs: int = 200
    patience: int = 8                      # plateau detection
    plateau_tolerance: float = 1e-3
    grad_clip: float = 1.0
    log_interval_steps: int = 50
    precision: str = "bf16-mixed"
    num_workers: int = 4                   # DataLoader CPU workers
    compile: bool = True                   # torch.compile pour kernel fusion


def collate(samples: list[StructureMNISTSample], pad_id: int = 0) -> dict[str, torch.Tensor]:
    # Pad to max length in batch — sweep ConcatDataset peut mélanger seq_len
    # variables (ω/Δ/ℋ produisent des longueurs différentes).
    from torch.nn.utils.rnn import pad_sequence
    tokens = pad_sequence([s.tokens for s in samples], batch_first=True, padding_value=pad_id)
    targets = torch.tensor([s.target for s in samples], dtype=torch.int64)
    omegas = torch.tensor([s.omega for s in samples], dtype=torch.int64)
    deltas = torch.tensor([s.delta for s in samples], dtype=torch.int64)
    entropies = torch.tensor([s.entropy for s in samples], dtype=torch.float32)
    return {
        "tokens": tokens,
        "targets": targets,
        "omegas": omegas,
        "deltas": deltas,
        "entropies": entropies,
    }


def find_query_positions(tokens: torch.Tensor, query_id: int) -> torch.Tensor:
    """Retourne (B,) int64 : position du token QUERY dans chaque séquence."""
    matches = tokens == query_id
    # premier index où c'est vrai
    idx = matches.float().argmax(dim=-1)
    return idx.to(torch.int64)


@torch.no_grad()
def evaluate(
    fabric: Fabric,
    model: OracleTransformer,
    loader: DataLoader,
    query_id: int,
) -> dict[str, float]:
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for batch in loader:
        tokens = batch["tokens"]
        targets = batch["targets"]
        qpos = find_query_positions(tokens, query_id)
        logits = model(tokens, qpos, capture_attn=False)
        loss = nn.functional.cross_entropy(logits, targets, reduction="sum")
        total_loss += loss.item()
        total_correct += (logits.argmax(dim=-1) == targets).sum().item()
        total_n += targets.numel()
    return {
        "val_loss": total_loss / max(total_n, 1),
        "val_acc": total_correct / max(total_n, 1),
        "val_n": total_n,
    }


def train_oracle(
    *,
    model_cfg: OracleConfig,
    train_ds: StructureMNISTDataset,
    val_ds: StructureMNISTDataset,
    train_cfg: TrainConfig,
    query_id: int,
    fabric: Fabric | None = None,
    metric_callback: Any | None = None,
    checkpoint_path: Path | None = None,
) -> tuple[OracleTransformer, dict[str, float]]:
    """Entraîne l'Oracle. Retourne (modèle entraîné, dernier set de métriques validation).

    `metric_callback(epoch, train_loss, val_metrics)` est appelé à chaque
    époque si fourni — branche MLflow ici.
    """
    # TF32 high precision pour matmul auxiliaires (init, lstsq batterie 2)
    torch.set_float32_matmul_precision("high")

    fabric = fabric or Fabric(precision=train_cfg.precision)
    fabric.launch()

    model = OracleTransformer(model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    model, optimizer = fabric.setup(model, optimizer)

    # Kernel fusion via torch.compile (dynamic=True pour gérer les seq_len variables
    # du sweep ConcatDataset sans recompilation par shape).
    if train_cfg.compile:
        model = torch.compile(model, dynamic=True)

    from functools import partial as _partial
    collate_fn = _partial(collate, pad_id=model_cfg.pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True,
        num_workers=train_cfg.num_workers, pin_memory=True,
        persistent_workers=train_cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=max(train_cfg.num_workers // 2, 1), pin_memory=True,
        persistent_workers=train_cfg.num_workers > 0,
    )
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    best_val_loss = float("inf")
    plateau_counter = 0
    val_metrics: dict[str, float] = {}

    for epoch in range(train_cfg.max_epochs):
        model.train()
        epoch_loss, epoch_n = 0.0, 0
        t0 = time.perf_counter()
        for step, batch in enumerate(train_loader):
            tokens = batch["tokens"]
            targets = batch["targets"]
            qpos = find_query_positions(tokens, query_id)
            optimizer.zero_grad()
            logits = model(tokens, qpos, capture_attn=False)
            loss = nn.functional.cross_entropy(logits, targets)
            fabric.backward(loss)
            fabric.clip_gradients(model, optimizer, max_norm=train_cfg.grad_clip)
            optimizer.step()
            epoch_loss += loss.item() * targets.numel()
            epoch_n += targets.numel()

        train_loss = epoch_loss / max(epoch_n, 1)
        val_metrics = evaluate(fabric, model, val_loader, query_id)
        if metric_callback is not None:
            metric_callback(epoch, train_loss, val_metrics)

        # plateau detection
        improvement = best_val_loss - val_metrics["val_loss"]
        if improvement > train_cfg.plateau_tolerance:
            best_val_loss = val_metrics["val_loss"]
            plateau_counter = 0
            if checkpoint_path is not None:
                state = {"model": model.state_dict(), "epoch": epoch, "best_val_loss": best_val_loss}
                fabric.save(str(checkpoint_path), state)
        else:
            plateau_counter += 1

        fabric.print(
            f"epoch {epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['val_loss']:.4f} val_acc={val_metrics['val_acc']:.3f} "
            f"plateau={plateau_counter}/{train_cfg.patience} "
            f"dur={time.perf_counter() - t0:.1f}s"
        )

        if plateau_counter >= train_cfg.patience:
            fabric.print(f"Plateau atteint à l'époque {epoch}. Arrêt.")
            break

    return model, val_metrics
