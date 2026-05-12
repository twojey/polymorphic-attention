"""
v5_schatten_p_norm.py — Property V5 : normes de Schatten S_p pour p ∈ {1, 2, ∞}.

‖A‖_{S_p} = (Σ σ_i^p)^{1/p}

- p = 1 : norme nucléaire (S_1)
- p = 2 : norme Frobenius (S_2)
- p = ∞ : norme spectrale (S_∞)

Schatten norms forment une échelle entre concentration spectrale (p → ∞,
seul σ_max compte) et dispersion (p → 1, somme totale). Le ratio S_1/S_∞
mesure la dispersion.

Réutilise svdvals_cached.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class V5SchattenPNorm(Property):
    """V5 — Schatten S_p norms for p ∈ {1, 2, ∞} + ratios."""

    name = "V5_schatten_p_norm"
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
        B, H, _, _ = A.shape

        s = ctx.svdvals_cached(A)  # (B, H, K)
        S1 = s.sum(dim=-1)
        S2 = s.pow(2).sum(dim=-1).sqrt()
        Sinf = s[..., 0].clamp_min(self.eps_floor)
        ratio_1_inf = S1 / Sinf  # ∈ [1, K]
        ratio_1_2 = S1 / S2.clamp_min(self.eps_floor)  # ∈ [1, √K]

        s1_flat = S1.float().flatten()
        s2_flat = S2.float().flatten()
        sinf_flat = Sinf.float().flatten()
        r_1_inf = ratio_1_inf.float().flatten()
        r_1_2 = ratio_1_2.float().flatten()

        K = s.shape[-1]
        return {
            "schatten_S1_nuclear_median": float(s1_flat.median().item()),
            "schatten_S2_frobenius_median": float(s2_flat.median().item()),
            "schatten_Sinf_spectral_median": float(sinf_flat.median().item()),
            "ratio_S1_over_Sinf_median": float(r_1_inf.median().item()),
            "ratio_S1_over_Sinf_normalized": float(
                (r_1_inf / K).median().item()
            ),
            "ratio_S1_over_S2_median": float(r_1_2.median().item()),
            "K": K,
            "n_matrices": int(B * H),
        }
