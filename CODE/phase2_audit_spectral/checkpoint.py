"""
checkpoint.py — sauvegarde/restore d'état intermédiaire phase 2.

Permet de relancer le driver après crash (OOM, SIGKILL, etc.) sans refaire
les étapes coûteuses déjà calculées. État sauvé après chaque étape majeure :

1. svd_done   → r_eff_per_layer_head_example.npy
2. srm_done   → srm_monovariate.pkl
3. transfer_law_done
4. head_spec_done
5. batteries_a_done
6. batteries_b_done
7. batteries_d_done
8. decoupling_done

Le caller vérifie `state.has(key)` avant de relancer une étape. Si oui, il
restaure et passe à l'étape suivante.

Format : un dossier `<output_dir>/_phase2_state/` avec un fichier .pt par
key. Atomic write via `torch.save` + rename.

NB : seq_lens et n_examples_total sont sauvegardés pour vérifier que le
resume est cohérent (mêmes dumps, sinon abort).
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class Phase2State:
    """État reprenable phase 2."""
    state_dir: Path
    fingerprint: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create_or_resume(
        cls,
        state_dir: Path | str,
        *,
        seq_lens: list[int],
        n_examples_total: int,
        svd_device: str,
        svd_precision: str,
    ) -> tuple["Phase2State", bool]:
        """Charge l'état existant si présent ET cohérent, sinon en crée un.

        Retourne (state, is_resumed). is_resumed=True si on reprend un état
        compatible. Si l'état existe mais est incompatible (seq_lens diff,
        config diff), on raise pour forcer l'utilisateur à clean le state_dir.
        """
        state_dir = Path(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        fingerprint_path = state_dir / "fingerprint.pkl"
        new_fingerprint = {
            "seq_lens": sorted(seq_lens),
            "n_examples_total": n_examples_total,
            "svd_device": svd_device,
            "svd_precision": svd_precision,
        }
        if fingerprint_path.exists():
            with open(fingerprint_path, "rb") as f:
                old_fingerprint = pickle.load(f)
            if old_fingerprint != new_fingerprint:
                raise RuntimeError(
                    f"État phase 2 existant incompatible : {old_fingerprint} != {new_fingerprint}. "
                    f"Supprimer {state_dir} pour repartir de zéro."
                )
            return cls(state_dir=state_dir, fingerprint=new_fingerprint), True

        with open(fingerprint_path, "wb") as f:
            pickle.dump(new_fingerprint, f)
        return cls(state_dir=state_dir, fingerprint=new_fingerprint), False

    def has(self, key: str) -> bool:
        return (self.state_dir / f"{key}.pt").is_file()

    def save(self, key: str, obj: Any) -> None:
        """Sauvegarde atomique via rename (évite corruption sur SIGKILL en plein write)."""
        target = self.state_dir / f"{key}.pt"
        tmp = self.state_dir / f"{key}.pt.tmp"
        torch.save(obj, tmp)
        os.replace(tmp, target)

    def load(self, key: str) -> Any:
        target = self.state_dir / f"{key}.pt"
        if not target.is_file():
            raise FileNotFoundError(f"Pas d'état pour key='{key}' dans {self.state_dir}")
        return torch.load(target, map_location="cpu", weights_only=False)

    def clean(self) -> None:
        """Supprime tous les checkpoints (force re-run from scratch au prochain start)."""
        for f in self.state_dir.glob("*.pt"):
            f.unlink()
        fp = self.state_dir / "fingerprint.pkl"
        if fp.is_file():
            fp.unlink()
