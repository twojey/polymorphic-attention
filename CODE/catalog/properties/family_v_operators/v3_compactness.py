"""
v3_compactness.py — Property V3 : compactness (décroissance queue spectrale).

Spec : DOC/CATALOGUE §V3.

Un opérateur compact a un spectre {σ_n} qui décroît vers 0 quand n → ∞.
Pour une matrice finie, on mesure :
- σ_K (queue) / σ_1 → quasi-zero pour compact-like
- nb σ_n > τ · σ_1 : "ordre effectif"

V1 : mesure les ratios σ_K/σ_1, σ_{K/2}/σ_1 pour caractériser la queue.
Réutilise cache svd_singular_values.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class V3Compactness(Property):
    """V3 — caractérisation de la queue spectrale (compactness)."""

    name = "V3_compactness"
    family = "V"
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
        B, H, N, N2 = A.shape

        cache_key = ctx.cache_key(
            "svd_singular_values", tuple(A.shape), str(A.dtype)
        )

        def _svd() -> torch.Tensor:
            return torch.linalg.svdvals(A.to(device=ctx.device, dtype=ctx.dtype))

        s = ctx.get_or_compute(cache_key, _svd)  # (B, H, K)
        K = s.shape[-1]
        sigma_1 = s[..., 0].clamp_min(self.eps_floor)

        tail_idx = K - 1
        sigma_K = s[..., tail_idx]
        sigma_half = s[..., K // 2]
        sigma_quart = s[..., (3 * K) // 4]

        ratio_tail = (sigma_K / sigma_1).float().flatten()
        ratio_half = (sigma_half / sigma_1).float().flatten()
        ratio_three_quarters = (sigma_quart / sigma_1).float().flatten()

        return {
            "tail_ratio_sigmaK_over_sigma1_median": float(ratio_tail.median().item()),
            "mid_ratio_sigma_half_over_sigma1_median": float(ratio_half.median().item()),
            "three_quarters_ratio_median": float(ratio_three_quarters.median().item()),
            "fraction_compact_like_tail_below_0p01": float(
                (ratio_tail < 0.01).float().mean().item()
            ),
            "K": K,
            "n_matrices": int(B * H),
        }
