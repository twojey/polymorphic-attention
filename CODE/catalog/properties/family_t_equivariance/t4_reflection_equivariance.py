"""
t4_reflection_equivariance.py — Property T4 : équivariance par réflexion.

Pour A : reflechir = inverser l'ordre des tokens. Si A est réflexivement
équivariant : A = J A J où J est la matrice échange (J[i, j] = δ(i+j, N-1)).

Mesure : ε_J = ‖A − J A J‖_F / ‖A‖_F.

Attention causale = jamais réflexive (causalité brisée).
Attention bidir + position embedding symétrique = potentiellement réflexive.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class T4ReflectionEquivariance(Property):
    """T4 — distance à l'équivariance par réflexion (échange ordre tokens)."""

    name = "T4_reflection_equivariance"
    family = "T"
    cost_class = 1
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self, eps_floor: float = 1e-30) -> None:
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N != N2:
            return {"skip_reason": "non-square", "n_matrices": int(B * H)}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Réflexion : flip lignes et colonnes
        A_refl = A_work.flip(dims=(-2, -1))
        A_norm = A_work.flatten(start_dim=-2).norm(dim=-1).clamp_min(self.eps_floor)
        diff = (A_work - A_refl).flatten(start_dim=-2).norm(dim=-1)
        eps = (diff / A_norm).float().flatten()

        # Aussi : test équivariance partielle (flip lignes seules ou colonnes seules)
        A_flip_row = A_work.flip(dims=(-2,))
        A_flip_col = A_work.flip(dims=(-1,))
        eps_row = ((A_work - A_flip_row).flatten(start_dim=-2).norm(dim=-1) / A_norm).float().flatten()
        eps_col = ((A_work - A_flip_col).flatten(start_dim=-2).norm(dim=-1) / A_norm).float().flatten()

        return {
            "epsilon_reflection_median": float(eps.median().item()),
            "epsilon_reflection_mean": float(eps.mean().item()),
            "epsilon_flip_row_median": float(eps_row.median().item()),
            "epsilon_flip_col_median": float(eps_col.median().item()),
            "fraction_approx_reflective": float(
                (eps < 0.10).float().mean().item()
            ),
            "n_matrices": int(B * H),
        }
