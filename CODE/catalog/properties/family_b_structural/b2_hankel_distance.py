"""
b2_hankel_distance.py — Property B2 : distance de Frobenius à la classe Hankel.

Spec : DOC/00b B2, DOC/glossaire §Matrice de Hankel.

ε_H(A) = ‖A − P_H(A)‖_F / ‖A‖_F

Le rang Hankel d'une matrice est lié à l'ordre minimal du système LTI qui
la génère (Ho-Kalman, DOC/00b famille P). Phase 2 sur SMNIST a montré
~33 % des régimes "less bad" → Hankel-winner mais ε > 0.45 partout.
"""

from __future__ import annotations

import torch

from catalog.projectors import Hankel
from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class B2HankelDistance(Property):
    name = "B2_hankel_distance"
    family = "B"
    cost_class = 2
    requires_fp64 = False
    requires_symmetric = False
    scope = "per_regime"

    def __init__(self) -> None:
        self._projector = Hankel()

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")

        cache_key = ctx.cache_key(
            "projection_hankel", tuple(A.shape), str(A.dtype)
        )

        def _project() -> torch.Tensor:
            A_work = A.to(device=ctx.device, dtype=ctx.dtype)
            return self._projector.project(A_work)

        proj = ctx.get_or_compute(cache_key, _project)
        A_work = A.to(device=ctx.device, dtype=ctx.dtype)

        residual = A_work - proj
        eps = (
            residual.flatten(start_dim=-2).norm(dim=-1)
            / A_work.flatten(start_dim=-2).norm(dim=-1).clamp_min(1e-30)
        )
        eps_flat = eps.float().flatten()

        return {
            "epsilon_median": float(eps_flat.median().item()),
            "epsilon_mean": float(eps_flat.mean().item()),
            "epsilon_min": float(eps_flat.min().item()),
            "epsilon_max": float(eps_flat.max().item()),
            "epsilon_p10": float(eps_flat.quantile(0.10).item()),
            "epsilon_p90": float(eps_flat.quantile(0.90).item()),
            "n_matrices": int(eps_flat.numel()),
            "fraction_below_0p30": float((eps_flat < 0.30).float().mean().item()),
        }
