"""
checkpoint.py — sauvegarde/restore du training ASPTransformer phase 3.

Cohérent avec feedback_script_robustness §2 (resume après crash obligatoire
pour toute étape > 5 min). Pour phase 3, l'entraînement peut prendre des
heures sur grand R_max — un crash en milieu d'epoch perdrait des minutes
voire des heures.

État sauvé après chaque epoch :
- model.state_dict()
- optimizer.state_dict()
- epoch courant + global step
- best_val_acc + oracle_acc baseline

Format : `<output_dir>/_phase3_state/`
- `fingerprint.pkl` : config critique (R_max, d_model, n_layers, backbone_class)
- `latest.pt`       : checkpoint le plus récent (epoch fini)
- `best.pt`         : checkpoint avec best val_acc à ce jour

Atomic save via `torch.save(obj, tmp)` + `os.replace(tmp, target)`.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class Phase3State:
    """État reprenable de l'entraînement phase 3."""
    state_dir: Path
    fingerprint: dict[str, Any] = field(default_factory=dict)

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
        """Charge l'état si compatible, sinon en crée un.

        Retourne (state, is_resumed). Si fingerprint diff → raise.
        """
        state_dir = Path(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        fp_path = state_dir / "fingerprint.pkl"
        new_fp = {
            "R_max": R_max, "d_model": d_model, "n_layers": n_layers,
            "backbone_class": backbone_class, "init_strategy": init_strategy,
        }
        if fp_path.exists():
            with open(fp_path, "rb") as f:
                old_fp = pickle.load(f)
            if old_fp != new_fp:
                raise RuntimeError(
                    f"État phase 3 incompatible : {old_fp} != {new_fp}. "
                    f"Supprimer {state_dir} pour repartir."
                )
            return cls(state_dir=state_dir, fingerprint=new_fp), True
        with open(fp_path, "wb") as f:
            pickle.dump(new_fp, f)
        return cls(state_dir=state_dir, fingerprint=new_fp), False

    def has_latest(self) -> bool:
        return (self.state_dir / "latest.pt").is_file()

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
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch, "step": step,
            "best_val_acc": best_val_acc, "oracle_acc": oracle_acc,
        }
        target = self.state_dir / "latest.pt"
        tmp = self.state_dir / "latest.pt.tmp"
        torch.save(ckpt, tmp)
        os.replace(tmp, target)

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
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch, "step": step,
            "val_acc": val_acc, "oracle_acc": oracle_acc,
        }
        target = self.state_dir / "best.pt"
        tmp = self.state_dir / "best.pt.tmp"
        torch.save(ckpt, tmp)
        os.replace(tmp, target)

    def load_latest(
        self, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        """Restaure model + optimizer en place. Retourne metadata (epoch, step, ...)."""
        target = self.state_dir / "latest.pt"
        if not target.is_file():
            raise FileNotFoundError(f"Pas de latest.pt dans {self.state_dir}")
        ckpt = torch.load(str(target), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        return {
            "epoch": ckpt["epoch"], "step": ckpt["step"],
            "best_val_acc": ckpt["best_val_acc"],
            "oracle_acc": ckpt["oracle_acc"],
        }

    def clean(self) -> None:
        for f in self.state_dir.glob("*.pt"):
            f.unlink()
        fp = self.state_dir / "fingerprint.pkl"
        if fp.is_file():
            fp.unlink()
