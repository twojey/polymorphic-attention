"""
toeplitz.py — Projecteur Toeplitz (moyenne par diagonale).

Spec : DOC/glossaire §Matrice de Toeplitz, DOC/CATALOGUE B1.

T_{ij} = t_{i−j}. Constante sur chaque diagonale. La projection orthogonale
Frobenius P_T(A)_{ij} = moyenne des A_{i'j'} sur i' − j' = i − j.

Implémentation : itère sur les diagonales offset d ∈ [-(M-1), N), calcule
la moyenne par diagonale, broadcast back. Vectorisé sur les dims batch
arbitraires en préfixe.
"""

from __future__ import annotations

import torch

from catalog.projectors.base import Projector


class Toeplitz(Projector):
    """Projection sur les matrices Toeplitz (constantes sur les diagonales)."""

    name = "toeplitz"
    family = "B"

    def project(self, A: torch.Tensor) -> torch.Tensor:
        if A.ndim < 2:
            raise ValueError(f"A doit être (..., M, N), reçu shape {A.shape}")
        *batch, M, N = A.shape
        out = torch.zeros_like(A)
        for d in range(-(M - 1), N):
            diag = torch.diagonal(A, offset=d, dim1=-2, dim2=-1)  # (..., k)
            if diag.numel() == 0:
                continue
            mean = diag.mean(dim=-1, keepdim=True)  # (..., 1)
            diag_size = diag.size(-1)
            i_start = max(0, -d)
            j_start = max(0, d)
            for k in range(diag_size):
                out[..., i_start + k, j_start + k] = mean.squeeze(-1)
        return out
