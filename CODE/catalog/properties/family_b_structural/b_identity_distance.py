"""
b_identity_distance.py — Property B-identity : distance de Frobenius à la classe diagonale.

Baseline trivial. Sur attention dense softmax, on s'attend à ε_I ≈ 1
(les matrices ne sont presque jamais diagonales). Sert de plancher de
comparaison pour Toeplitz/Hankel/Cauchy.

NB : pas une propriété "officielle" du catalogue DOC/CATALOGUE mais utile en
ablation. Identifiant `B0_identity_distance` (avant B1) par convention.
"""

from __future__ import annotations

import torch

from catalog.projectors import Identity
from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class B0IdentityDistance(Property):
    name = "B0_identity_distance"
    family = "B"
    cost_class = 1  # extrêmement rapide (diag extraction)
    requires_fp64 = False
    requires_symmetric = False
    scope = "per_regime"

    def __init__(self) -> None:
        self._projector = Identity()

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        proj = self._projector.project(A_work)

        eps = (
            (A_work - proj).flatten(start_dim=-2).norm(dim=-1)
            / A_work.flatten(start_dim=-2).norm(dim=-1).clamp_min(1e-30)
        )
        eps_flat = eps.float().flatten()

        return {
            "epsilon_median": float(eps_flat.median().item()),
            "epsilon_mean": float(eps_flat.mean().item()),
            "epsilon_min": float(eps_flat.min().item()),
            "epsilon_max": float(eps_flat.max().item()),
            "n_matrices": int(eps_flat.numel()),
        }
