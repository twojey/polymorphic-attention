"""
checkpoint.py — wrapper léger sur shared.checkpoint.Checkpoint pour phase 3.

Maintenu pour backward compat. Le stockage est délégué à
`shared.checkpoint.Checkpoint` ; les méthodes save_latest / save_best /
load_latest construisent les dicts checkpoint adaptés au training
ASPTransformer (model.state_dict + optimizer.state_dict + meta).

État sauvé après chaque epoch :
- model.state_dict()
- optimizer.state_dict()
- epoch + global step
- best_val_acc + oracle_acc baseline

Fingerprint : R_max, d_model, n_layers, backbone_class, init_strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from shared.checkpoint import Checkpoint


@dataclass
class Phase3State:
    """État reprenable de l'entraînement phase 3 (compose Checkpoint)."""

    checkpoint: Checkpoint

    @property
    def state_dir(self) -> Path:
        return self.checkpoint.state_dir

    @property
    def fingerprint(self) -> dict[str, Any]:
        return self.checkpoint.fingerprint

    @classmethod
    def create_or_resume(
        cls,
        state_dir: Path | str,
        *,
        R_max: int,
        d_model: int,
        n_layers: int,
        backbone_class: str,
        init_strategy: str,
    ) -> tuple["Phase3State", bool]:
        fp: dict[str, Any] = {
            "R_max": R_max, "d_model": d_model, "n_layers": n_layers,
            "backbone_class": backbone_class, "init_strategy": init_strategy,
        }
        cp, resumed = Checkpoint.create_or_resume(state_dir, fingerprint=fp)
        return cls(checkpoint=cp), resumed

    def has_latest(self) -> bool:
        return self.checkpoint.has("latest")

    def save_latest(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        best_val_acc: float,
        oracle_acc: float,
    ) -> None:
        self.checkpoint.save("latest", {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch, "step": step,
            "best_val_acc": best_val_acc, "oracle_acc": oracle_acc,
        })

    def save_best(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        val_acc: float,
        oracle_acc: float,
    ) -> None:
        self.checkpoint.save("best", {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch, "step": step,
            "val_acc": val_acc, "oracle_acc": oracle_acc,
        })

    def load_latest(
        self, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        ckpt = self.checkpoint.load("latest")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        return {
            "epoch": ckpt["epoch"], "step": ckpt["step"],
            "best_val_acc": ckpt["best_val_acc"],
            "oracle_acc": ckpt["oracle_acc"],
        }

    def clean(self) -> None:
        self.checkpoint.clean()
