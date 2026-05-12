"""
e4_mdl_proxy.py — Property E4 : proxy de Minimum Description Length.

MDL(A) ≈ rang_eff(A) · (m + n) · ⌈log₂(1/precision)⌉ bits
       + Σ_i ⌈log₂(1 + σ_i)⌉ bits

Approximation : pour une matrice (m, n), une description low-rank coûte
~(m+n)·r bits, plus les σ_i quantifiés. Plus c'est faible, plus A est
compressible au sens MDL.

Réutilise cache `svd_singular_values`.
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class E4MdlProxy(Property):
    """E4 — proxy MDL bits = (m+n)·r_eff + Σ log₂(σ)."""

    name = "E4_mdl_proxy"
    family = "E"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(
        self,
        theta_cumulative: float = 0.99,
        precision_bits: int = 8,
        eps_floor: float = 1e-30,
    ) -> None:
        self.theta = theta_cumulative
        self.precision_bits = precision_bits
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape

        s = ctx.svdvals_cached(A)  # (B, H, K)
        s2 = s.pow(2)
        cumsum = s2.cumsum(dim=-1)
        total = cumsum[..., -1:].clamp_min(self.eps_floor)
        ratio = cumsum / total
        above = ratio >= self.theta
        r_eff = above.float().argmax(dim=-1).clamp_min(0) + 1

        # MDL en bits : (m+n)·r·b + sum log₂(σ_i ∈ top r_eff)
        struct_bits = (N + N2) * r_eff.float() * self.precision_bits
        # Approx Σ log₂(σ_i) : juste sur les top r
        log_s = torch.log2(s.clamp_min(self.eps_floor))  # (B, H, K)
        # Mask top r_eff per (b, h)
        idx = torch.arange(s.shape[-1], device=s.device).expand_as(log_s)
        mask = idx < r_eff.unsqueeze(-1)
        sigma_bits = (log_s * mask.float()).sum(dim=-1).abs()
        mdl = (struct_bits + sigma_bits).float().flatten()

        # MDL relatif : par rapport à description brute (m*n*b)
        raw_bits = N * N2 * self.precision_bits
        mdl_ratio = mdl / max(raw_bits, 1)

        return {
            "mdl_bits_median": float(mdl.median().item()),
            "mdl_bits_mean": float(mdl.mean().item()),
            "mdl_ratio_median": float(mdl_ratio.median().item()),
            "fraction_compressible_mdl_below_0p2": float(
                (mdl_ratio < 0.20).float().mean().item()
            ),
            "raw_bits_per_matrix": int(raw_bits),
            "n_matrices": int(B * H),
        }
