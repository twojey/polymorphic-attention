"""
i5_head_redundancy.py — Property I5 : redondance d'attention entre têtes.

Pour chaque paire de têtes (h1, h2), calcule la corrélation de Pearson
des matrices A_{h1} et A_{h2} aplaties. Redondance forte = corr proche 1.

Distinguer de I3 (clustering : groupes) et D3 (subspace angles : géométrie).
I5 est une mesure simple, scalable, vectorisable.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class I5HeadRedundancy(Property):
    """I5 — corrélation Pearson cross-head, aplatie en (N×N) features."""

    name = "I5_head_redundancy"
    family = "I"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self, eps_floor: float = 1e-12) -> None:
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, _ = A.shape
        if H < 2:
            return {"skip_reason": "H<2", "n_matrices": int(B * H)}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Flatten (B, H, N²) puis centrer per head
        flat = A_work.flatten(start_dim=-2)  # (B, H, N²)
        mean = flat.mean(dim=-1, keepdim=True)
        centered = flat - mean
        std = centered.norm(dim=-1, keepdim=True).clamp_min(self.eps_floor)
        normed = centered / std  # (B, H, N²)
        # Corrélation pairwise via Gram matrix
        gram = torch.einsum("bhi,bki->bhk", normed, normed)  # (B, H, H)
        # Off-diagonal
        eye = torch.eye(H, device=A_work.device, dtype=torch.bool)
        off_diag = gram[:, ~eye]  # (B, H*(H-1))
        off_flat = off_diag.float().flatten()

        return {
            "head_corr_median": float(off_flat.median().item()),
            "head_corr_mean": float(off_flat.mean().item()),
            "head_corr_max": float(off_flat.max().item()),
            "head_corr_abs_median": float(off_flat.abs().median().item()),
            "fraction_redundant_pairs_above_0p80": float(
                (off_flat > 0.80).float().mean().item()
            ),
            "fraction_anti_correlated_below_minus_0p20": float(
                (off_flat < -0.20).float().mean().item()
            ),
            "n_head_pairs": int(H * (H - 1)),
            "n_matrices": int(B * H),
        }
