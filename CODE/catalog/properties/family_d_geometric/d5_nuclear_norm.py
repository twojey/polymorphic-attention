"""
d5_nuclear_norm.py — Property D5 : norme nucléaire ‖A‖_*.

‖A‖_* = Σ σ_i, dual de la norme spectrale. Tight relaxation convexe du rang.
Permet de comparer Oracles : grand ‖A‖_* = beaucoup d'énergie distribuée
sur de nombreux modes ; petit = quelques modes dominants.

Réutilise cache `svd_singular_values` (mutualisation famille A).
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class D5NuclearNorm(Property):
    """D5 — ‖A‖_* = Σ σ_i (norme nucléaire = sum of singular values)."""

    name = "D5_nuclear_norm"
    family = "D"
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
        nuc = s.sum(dim=-1)
        frob = s.pow(2).sum(dim=-1).sqrt()
        spec = s[..., 0].clamp_min(self.eps_floor)
        # ratio (proche 1 si σ_max domine, proche √K si σ uniformes)
        nuc_over_spec = nuc / spec
        nuc_flat = nuc.float().flatten()
        ratio_flat = nuc_over_spec.float().flatten()
        nuc_over_frob = (nuc / frob.clamp_min(self.eps_floor)).float().flatten()

        K = s.shape[-1]
        return {
            "nuclear_norm_median": float(nuc_flat.median().item()),
            "nuclear_norm_mean": float(nuc_flat.mean().item()),
            "nuclear_over_spectral_median": float(ratio_flat.median().item()),
            "nuclear_over_frobenius_median": float(nuc_over_frob.median().item()),
            "K": K,
            "n_matrices": int(B * H),
        }
