"""
hankel.py — Projecteur Hankel (moyenne par anti-diagonale).

Spec : DOC/glossaire §Matrice de Hankel, DOC/00b B2.

H_{ij} = h_{i+j}. Constante sur chaque anti-diagonale. La projection
orthogonale Frobenius P_H(A)_{ij} = moyenne des A_{i'j'} sur i' + j' = i + j.

Le rang de Hankel est lié à l'ordre minimal d'un système LTI qui génère
la matrice (Ho-Kalman, DOC/00b famille P).
"""

from __future__ import annotations

import torch

from catalog.projectors.base import Projector


class Hankel(Projector):
    """Projection sur les matrices Hankel (constantes sur les anti-diagonales)."""

    name = "hankel"
    family = "B"

    def project(self, A: torch.Tensor) -> torch.Tensor:
        if A.ndim < 2:
            raise ValueError(f"A doit être (..., M, N), reçu shape {A.shape}")
        *batch, M, N = A.shape
        out = torch.zeros_like(A)
        for s in range(M + N - 1):
            # antidiag i + j = s
            positions = [(i, s - i) for i in range(max(0, s - N + 1), min(M, s + 1))]
            if not positions:
                continue
            vals = torch.stack([A[..., i, j] for i, j in positions], dim=-1)
            mean = vals.mean(dim=-1, keepdim=True)
            for i, j in positions:
                out[..., i, j] = mean.squeeze(-1)
        return out
