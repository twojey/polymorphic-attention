"""
a6_participation_ratio.py — Property A6 : Participation Ratio.

Spec : DOC/CATALOGUE §A6.

PR(A) = (Σ_i σ_i²)² / Σ_i σ_i⁴

Mesure le "nombre effectif de modes" excitations. PR = 1 si une seule σ
non-nulle (rank-1). PR = K si toutes σ égales (uniforme sur K modes).
Plus rugueux que r_eff(θ) mais sans seuil.

Réutilise cache `svd_singular_values` posé par A1.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class A6ParticipationRatio(Property):
    """A6 — Participation Ratio = (Σ σ²)² / Σ σ⁴, "nb effectif de modes"."""

    name = "A6_participation_ratio"
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

        cache_key = ctx.cache_key(
            "svd_singular_values", tuple(A.shape), str(A.dtype)
        )

        def _svd() -> torch.Tensor:
            return torch.linalg.svdvals(A.to(device=ctx.device, dtype=ctx.dtype))

        s = ctx.get_or_compute(cache_key, _svd)  # (B, H, K)
        s2 = s.pow(2)
        s4 = s.pow(4)
        num = s2.sum(dim=-1) ** 2
        den = s4.sum(dim=-1).clamp_min(self.eps_floor)
        PR = num / den  # (B, H), ∈ [1, K]

        K = s.shape[-1]
        PR_norm = PR / K  # ∈ [1/K, 1]
        pr_flat = PR.float().flatten()
        n_flat = PR_norm.float().flatten()

        return {
            "participation_ratio_median": float(pr_flat.median().item()),
            "participation_ratio_mean": float(pr_flat.mean().item()),
            "participation_ratio_norm_median": float(n_flat.median().item()),
            "fraction_pr_below_3": float((pr_flat < 3.0).float().mean().item()),
            "fraction_pr_norm_above_0p5": float(
                (n_flat > 0.5).float().mean().item()
            ),
            "K": K,
            "n_matrices": int(pr_flat.numel()),
        }
