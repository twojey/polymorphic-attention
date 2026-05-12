"""
a7_effective_dimension.py — Property A7 : dimension effective de Rényi-2.

Spec : DOC/CATALOGUE §A7 (étendu V3, fin session).

d_eff_2(A) = (Σ p_i)² / Σ p_i²    avec p_i = σ_i² / Σ σ_j²

Dimension effective au sens Rényi-2 = exp(H_2) où H_2 = -log Σ p_i².
Distincte de A4 (Shannon H_1) et de A6 (PR sur σ²) :
A7 = exp(-log Σ p²) sur les **valeurs propres normalisées**.

Pour Oracles très concentrés, d_eff_2 ≪ A4 (Rényi-2 plus sensible aux pics).
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class A7EffectiveDimension(Property):
    """A7 — d_eff Rényi-2 = exp(-log Σ p²) sur valeurs singulières normalisées."""

    name = "A7_effective_dimension"
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
        s2 = s.pow(2)
        total = s2.sum(dim=-1).clamp_min(self.eps_floor)
        p = s2 / total.unsqueeze(-1)
        # Rényi-2 entropy : H_2 = -log Σ p²
        H2 = -torch.log(p.pow(2).sum(dim=-1).clamp_min(self.eps_floor))
        d_eff = torch.exp(H2)  # (B, H)
        d_eff_flat = d_eff.float().flatten()

        K = s.shape[-1]
        return {
            "d_eff_renyi2_median": float(d_eff_flat.median().item()),
            "d_eff_renyi2_mean": float(d_eff_flat.mean().item()),
            "d_eff_renyi2_norm_median": float(
                (d_eff_flat / max(K, 1)).median().item()
            ),
            "H2_median_nats": float(
                (-torch.log(p.pow(2).sum(-1).clamp_min(self.eps_floor)))
                .float().median().item()
            ),
            "fraction_d_eff_below_3": float(
                (d_eff_flat < 3.0).float().mean().item()
            ),
            "K": K,
            "n_matrices": int(B * H),
        }
