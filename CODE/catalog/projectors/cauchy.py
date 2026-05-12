"""
cauchy.py — Projector sur la classe Cauchy paramétrée par (x, y) donnés.

Définition : matrice de Cauchy C[i,j] = 1/(y_i − x_j) où x = (x_0, ..., x_{N-1})
et y = (y_0, ..., y_{N-1}) sont deux ensembles de poles disjoints.

Pour des poles (x, y) FIXÉS, la "classe Cauchy" est strictement un sous-espace
1-dim (la matrice elle-même) — donc la projection orthogonale Frobenius est
triviale (projection scalaire). Plus utile pratiquement : projection sur le
sous-espace généré par {C × diag(α)} pour α ∈ R^N (scaling de colonnes,
Cauchy-like factorisations).

V1 : projection scalaire (rank-1) sur la classe = projection sur span(C).
   Suffisant pour mesure de distance B3 ε_Cauchy.
"""

from __future__ import annotations

import torch

from catalog.projectors.base import Projector


class Cauchy(Projector):
    """Projection sur la matrice de Cauchy fixée par les poles (x, y).

    Pour poles donnés (x, y), la matrice de Cauchy C(x, y) est unique :
    C[i,j] = 1/(y_i − x_j). La projection orthogonale Frobenius de A sur
    span(C) est <A, C>_F / ‖C‖²_F · C.

    Args
    ----
    x, y : tensors 1D (N,) — poles disjoints (y_i ≠ x_j ∀ i, j)
    """

    name = "cauchy"
    family = "B"

    def __init__(self, x: torch.Tensor, y: torch.Tensor, eps_floor: float = 1e-12) -> None:
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError(f"x, y doivent être 1D, reçu {x.shape}, {y.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x et y doivent même longueur, reçu {x.shape[0]} != {y.shape[0]}")
        self.x = x
        self.y = y
        self.eps_floor = eps_floor

    def _build_cauchy(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Construit la matrice de Cauchy (N, N) pour (x, y)."""
        x = self.x.to(device=device, dtype=dtype)
        y = self.y.to(device=device, dtype=dtype)
        # C[i, j] = 1 / (y_i - x_j)
        diff = y.view(-1, 1) - x.view(1, -1)  # (N, N)
        # Pour éviter divisions par zéro si poles non strictement disjoints
        safe_diff = torch.where(
            diff.abs() < self.eps_floor,
            torch.full_like(diff, self.eps_floor),
            diff,
        )
        return 1.0 / safe_diff

    def project(self, A: torch.Tensor) -> torch.Tensor:
        """Projection orthogonale Frobenius de A sur span(Cauchy(x, y))."""
        if A.shape[-1] != self.x.shape[0] or A.shape[-2] != self.y.shape[0]:
            raise ValueError(
                f"A doit être (..., len(y), len(x)) = (..., {self.y.shape[0]}, {self.x.shape[0]}), "
                f"reçu {A.shape}"
            )
        C = self._build_cauchy(A.device, A.dtype)  # (N, N)
        # Inner product <A, C>_F par batch elem
        inner = (A * C).flatten(start_dim=-2).sum(dim=-1)  # (...,)
        norm_sq = (C ** 2).sum().clamp_min(self.eps_floor)
        # Projection rank-1 : (inner / ‖C‖²) · C
        scale = inner / norm_sq
        proj = scale.unsqueeze(-1).unsqueeze(-1) * C
        return proj
