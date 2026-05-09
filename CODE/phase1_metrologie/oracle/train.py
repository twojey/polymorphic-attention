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


def collate(samples: list[StructureMNISTSample]) -> dict[str, torch.Tensor]:
    tokens = torch.stack([s.tokens for s in samples], dim=0)
    targets = torch.tensor([s.target for s in samples], dtype=torch.int64)
    omegas = torch.tensor([s.omega for s in samples], dtype=torch.int64)
    deltas = torch.tensor([s.delta for s in samples], dtype=torch.int64)
    entropies = torch.tensor([s.entropy for s in samples], dtype=torch.float32)
    # query_pos : indice du dernier token QUERY (par construction = avant le PAD final)
    # Méthode : dernier index où token == QUERY ID (à passer en argument séparé en pratique).
    # Ici on assume que la position QUERY est la même pour tous les exemples du batch
    # (vrai dans un sweep monovarié à ω/Δ fixés).
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
    fabric = fabric or Fabric(precision=train_cfg.precision)
    fabric.launch()

    model = OracleTransformer(model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    model, optimizer = fabric.setup(model, optimizer)

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate
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
