"""
block_diagonal.py — Projecteur Block-Diagonal.

Spec : DOC/CATALOGUE B5 (block_diag), DOC/glossaire §H-matrix.

Une matrice block-diagonale a A_{ij} = 0 si (i // block) ≠ (j // block).
La projection orthogonale Frobenius est : garder les blocs sur la diagonale,
zéro ailleurs.

Paramètre : `block_size`. Pour N non divisible par block_size, le dernier
bloc est tronqué.
"""

from __future__ import annotations

import torch

from catalog.projectors.base import Projector


class BlockDiagonal(Projector):
    """Projection sur les matrices block-diagonales de taille de bloc fixée.

    Avec block_size=1 → équivalent à Identity (cf. `Identity()`).
    Avec block_size=N → projecteur trivial (A inchangée).
    """

    name = "block_diagonal"
    family = "B"

    def __init__(self, block_size: int) -> None:
        if block_size < 1:
            raise ValueError(f"block_size doit être ≥ 1, reçu {block_size}")
        self.block_size = block_size

    def project(self, A: torch.Tensor) -> torch.Tensor:
        if A.size(-1) != A.size(-2):
            raise ValueError(
                f"BlockDiagonal attend carrée (..., N, N), reçu {A.shape}"
            )
        N = A.size(-1)
        bs = self.block_size
        out = torch.zeros_like(A)
        for start in range(0, N, bs):
            end = min(start + bs, N)
            out[..., start:end, start:end] = A[..., start:end, start:end]
        return out
