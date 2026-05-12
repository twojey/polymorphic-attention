"""
e2_compressibility.py — Property E2 : compressibilité (proxy spectral + entropie).

Spec : DOC/CATALOGUE §E2 "proxy LZ ou rang spectral".

Pour mesurer "à quel point A est compressible", deux proxies :
1. Rang effectif r_eff / N → compressibilité via décomposition low-rank
2. Entropie des coefficients après quantification grossière (8 bins fixed)
   → proxy "Lempel-Ziv léger" : moins d'entropie = plus compressible

Sortie : ratio_lowrank, ratio_quantized_entropy, score_combined.
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class E2Compressibility(Property):
    """E2 — proxy compressibilité spectral + quantification."""

    name = "E2_compressibility"
    family = "E"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(
        self,
        n_bins: int = 8,
        theta_cumulative: float = 0.99,
        eps_floor: float = 1e-30,
    ) -> None:
        if n_bins < 2:
            raise ValueError(f"n_bins doit être ≥ 2, reçu {n_bins}")
        self.n_bins = n_bins
        self.theta = theta_cumulative
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)

        # --- 1. Compressibilité spectrale via r_eff / N ---
        cache_key = ctx.cache_key(
            "svd_singular_values", tuple(A.shape), str(A.dtype)
        )

        def _svd() -> torch.Tensor:
            return torch.linalg.svdvals(A_work)

        s = ctx.get_or_compute(cache_key, _svd)
        s2 = s.pow(2)
        cumsum = s2.cumsum(dim=-1)
        total = cumsum[..., -1:].clamp_min(self.eps_floor)
        ratio = cumsum / total
        above = ratio >= self.theta
        r_eff = above.float().argmax(dim=-1) + 1  # (B, H)
        K = s.shape[-1]
        compression_lowrank = (r_eff.float() / K).flatten()  # ratio ∈ [1/K, 1]

        # --- 2. Entropie des coefficients quantifiés ---
        flat = A_work.flatten(start_dim=-2)  # (B, H, N²)
        # Min/max par matrice
        mn = flat.min(dim=-1, keepdim=True).values
        mx = flat.max(dim=-1, keepdim=True).values
        rg = (mx - mn).clamp_min(self.eps_floor)
        # Bin index ∈ [0, n_bins-1]
        bin_idx = ((flat - mn) / rg * (self.n_bins - 1)).floor().long().clamp(0, self.n_bins - 1)
        # Compter occurrences par bin (vectorisé via one_hot puis sum)
        one_hot = torch.nn.functional.one_hot(bin_idx, num_classes=self.n_bins).float()
        counts = one_hot.sum(dim=-2)  # (B, H, n_bins)
        probs = counts / counts.sum(dim=-1, keepdim=True).clamp_min(self.eps_floor)
        probs_safe = probs.clamp_min(self.eps_floor)
        H_quant = -(probs * probs_safe.log()).sum(dim=-1)  # nats
        # Normalisée par log(n_bins) (max théorique)
        H_norm = H_quant / math.log(self.n_bins)
        compression_entropy = H_norm.flatten()  # ratio ∈ [0, 1]

        # --- 3. Score combiné (moyenne géométrique) ---
        combined = (compression_lowrank * compression_entropy).sqrt()

        cl_flat = compression_lowrank.float()
        ce_flat = compression_entropy.float()
        cm_flat = combined.float()

        return {
            "compression_lowrank_median": float(cl_flat.median().item()),
            "compression_lowrank_mean": float(cl_flat.mean().item()),
            "compression_quantized_entropy_median": float(ce_flat.median().item()),
            "compression_quantized_entropy_mean": float(ce_flat.mean().item()),
            "compression_combined_median": float(cm_flat.median().item()),
            "compression_combined_mean": float(cm_flat.mean().item()),
            "fraction_highly_compressible_combined_below_0p20": float(
                (cm_flat < 0.20).float().mean().item()
            ),
            "n_bins": self.n_bins,
            "n_matrices": int(B * H),
        }
