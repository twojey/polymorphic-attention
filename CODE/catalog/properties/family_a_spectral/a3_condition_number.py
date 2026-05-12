"""
a3_condition_number.py — Property A3 : nombre de conditionnement κ.

Spec : DOC/CATALOGUE §A3.

κ(A) = σ_max(A) / σ_min(A) (rapport valeur singulière max / min).
Pour matrice singulière, σ_min = 0 → κ = ∞ ; on plafonne via une masque
"finite" et on rapporte le log10(κ) pour stabilité numérique.

Réutilise le cache `svd_singular_values` posé par A1.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class A3ConditionNumber(Property):
    """A3 — log10(κ) = log10(σ_max / σ_min), plus fraction matrices singulières."""

    name = "A3_condition_number"
    family = "A"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self, sigma_min_floor: float = 1e-14) -> None:
        self.sigma_min_floor = sigma_min_floor

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

        s = ctx.get_or_compute(cache_key, _svd)  # (B, H, N), décroissant
        sigma_max = s[..., 0]
        sigma_min = s[..., -1]

        # Singulier strict
        is_singular = sigma_min < self.sigma_min_floor  # (B, H) bool
        # κ "robuste" : clamp σ_min au floor, mais on garde le tag "singular"
        sigma_min_safe = sigma_min.clamp_min(self.sigma_min_floor)
        kappa = sigma_max / sigma_min_safe
        log10_kappa = kappa.log10()

        log_k_flat = log10_kappa.float().flatten()
        sigma_min_flat = sigma_min.float().flatten()

        return {
            "log10_kappa_median": float(log_k_flat.median().item()),
            "log10_kappa_mean": float(log_k_flat.mean().item()),
            "log10_kappa_p10": float(log_k_flat.quantile(0.10).item()),
            "log10_kappa_p90": float(log_k_flat.quantile(0.90).item()),
            "sigma_min_median": float(sigma_min_flat.median().item()),
            "fraction_singular": float(is_singular.float().mean().item()),
            "n_matrices": int(log_k_flat.numel()),
        }
