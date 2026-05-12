"""
checkpoint.py — Checkpoint générique pour étapes longues du pipeline ASP.

Cohérent avec feedback_script_robustness §2 : toute étape > 5 min DOIT
avoir un checkpoint atomique pour resume après crash (SIGKILL, OOM, pod
fermé, etc.).

Pattern unifié pour phase 1/2/3 et catalog/ futurs :
- Fingerprint : dict de params critiques sérialisable (pickle)
- État : un fichier .pt par étape, atomic write via os.replace
- Resume : check fingerprint compat, load .pt, skip étape

Remplace les Phase2State (104 l) et Phase3State (141 l) ainsi que les
futurs BatteryState. Une seule implémentation, typed wrappers à côté si
besoin.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class Checkpoint:
    """Checkpoint générique — fingerprint + atomic save par étape nommée."""

    state_dir: Path
    fingerprint: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create_or_resume(
        cls,
        state_dir: Path | str,
        *,
        fingerprint: dict[str, Any],
    ) -> tuple["Checkpoint", bool]:
        """Crée ou reprend un checkpoint.

        Retourne (checkpoint, is_resumed). is_resumed=True si état existant
        compatible (fingerprint match). Lève RuntimeError si fingerprint
        incompatible — l'utilisateur doit explicitement clean pour repartir.
        """
        state_dir = Path(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        fp_path = state_dir / "fingerprint.pkl"

        if fp_path.exists():
            with open(fp_path, "rb") as f:
                old_fp = pickle.load(f)
            if old_fp != fingerprint:
                raise RuntimeError(
                    f"Checkpoint state incompatible : {old_fp} != {fingerprint}. "
                    f"Supprimer {state_dir} pour repartir de zéro."
                )
            return cls(state_dir=state_dir, fingerprint=fingerprint), True

        with open(fp_path, "wb") as f:
            pickle.dump(fingerprint, f)
        return cls(state_dir=state_dir, fingerprint=fingerprint), False

    def has(self, key: str) -> bool:
        """True si l'étape `key` a déjà été calculée et sauvegardée."""
        return (self.state_dir / f"{key}.pt").is_file()

    def save(self, key: str, obj: Any) -> None:
        """Sauvegarde atomique via tmp + rename. Robuste à SIGKILL en plein write."""
        target = self.state_dir / f"{key}.pt"
        tmp = self.state_dir / f"{key}.pt.tmp"
        torch.save(obj, tmp)
        os.replace(tmp, target)

    def load(self, key: str) -> Any:
        target = self.state_dir / f"{key}.pt"
        if not target.is_file():
            raise FileNotFoundError(f"Pas d'état pour key='{key}' dans {self.state_dir}")
        return torch.load(str(target), map_location="cpu", weights_only=False)

    def clean(self) -> None:
        """Supprime tous les checkpoints (force re-run from scratch)."""
        for f in self.state_dir.glob("*.pt"):
            f.unlink()
        fp = self.state_dir / "fingerprint.pkl"
        if fp.is_file():
            fp.unlink()

    def keys(self) -> list[str]:
        """Liste les étapes déjà sauvegardées."""
        return sorted(f.stem for f in self.state_dir.glob("*.pt"))
