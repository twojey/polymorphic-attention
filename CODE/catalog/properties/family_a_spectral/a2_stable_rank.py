"""
a2_stable_rank.py — Property A2 : rang stable (Frobenius²/spectral²).

Spec : DOC/CATALOGUE §A2 (étendu V3, fin session).

stable_rank(A) = ‖A‖_F² / ‖A‖_2² = Σ σ_i² / σ_max²

Borne inférieure du rang algébrique, robuste au bruit numérique. Plus
informatif que r_eff(θ) pour les distributions très étalées.

Réutilise cache `svd_singular_values` (mutualisation A1/A3/A4/A5/A6).
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class A2StableRank(Property):
    """A2 — stable rank = ‖A‖_F² / ‖A‖_2²."""

    name = "A2_stable_rank"
    family = "A"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self, eps_floor: float = 1e-30) -> None:
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, _, _ = A.shape

        s = ctx.svdvals_cached(A)  # (B, H, K)
        s_max = s[..., 0].clamp_min(self.eps_floor ** 0.5)
        frob2 = s.pow(2).sum(dim=-1)
        sr = frob2 / s_max.pow(2)  # (B, H)
        sr_flat = sr.float().flatten()

        return {
            "stable_rank_median": float(sr_flat.median().item()),
            "stable_rank_mean": float(sr_flat.mean().item()),
            "stable_rank_max": float(sr_flat.max().item()),
            "fraction_below_3": float((sr_flat < 3.0).float().mean().item()),
            "fraction_below_10": float((sr_flat < 10.0).float().mean().item()),
            "n_matrices": int(B * H),
        }
