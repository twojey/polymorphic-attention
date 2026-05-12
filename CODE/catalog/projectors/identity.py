"""
identity.py — Projecteur Identity (diagonale principale).

Conserve uniquement la diagonale, met les off-diagonal à 0. Sert de baseline
trivial pour les batteries — devrait toujours échouer ε ≫ 0 sur des matrices
d'attention denses.
"""

from __future__ import annotations

import torch

from catalog.projectors.base import Projector


class Identity(Projector):
    """Projection sur les matrices diagonales (off-diagonal = 0)."""

    name = "identity"
    family = "B"

    def project(self, A: torch.Tensor) -> torch.Tensor:
        if A.size(-1) != A.size(-2):
            raise ValueError(
                f"Identity attend une matrice carrée (..., N, N), "
                f"reçu shape {tuple(A.shape)}"
            )
        out = torch.zeros_like(A)
        diag = torch.diagonal(A, dim1=-2, dim2=-1)  # (..., N)
        k = diag.size(-1)
        idx = torch.arange(k, device=A.device)
        out[..., idx, idx] = diag
        return out
