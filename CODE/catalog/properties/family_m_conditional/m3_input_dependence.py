"""
m3_input_dependence.py — Property M3 : dépendance vs entrée intra-batch.

Pour B exemples du même régime, mesure la variance des matrices A entre
exemples (var across batch axis). Si A varie peu cross-batch → A est
quasi data-independent (ex : ALiBi, RoPE pur). Si A varie beaucoup → A
dépend fortement de l'entrée.

variance_per_entry = Var_B(A[b, h, t, t']).mean() (sur h, t, t').
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class M3InputDependence(Property):
    """M3 — variance Inter-batch des matrices d'attention."""

    name = "M3_input_dependence"
    family = "M"
    cost_class = 1
    requires_fp64 = False
    scope = "per_regime"

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, _ = A.shape
        if B < 2:
            return {"skip_reason": "B<2", "n_matrices": int(B * H)}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        var_entry = A_work.var(dim=0, unbiased=False)  # (H, N, N)
        # Mean variance per head
        var_per_head = var_entry.mean(dim=(-2, -1))  # (H,)
        # Mean A magnitude per head (pour ratio)
        mean_magnitude = A_work.mean(dim=0).abs().mean(dim=(-2, -1))  # (H,)
        normalized_var = var_per_head / mean_magnitude.clamp_min(1e-12)

        # Variance row-sum : ce qui se passe sur row distribution
        row_dist_var = A_work.var(dim=0, unbiased=False).flatten()

        var_flat = var_per_head.float().flatten()
        norm_flat = normalized_var.float().flatten()
        return {
            "input_variance_per_head_median": float(var_flat.median().item()),
            "input_variance_per_head_max": float(var_flat.max().item()),
            "normalized_variance_median": float(norm_flat.median().item()),
            "row_dist_var_median": float(row_dist_var.float().median().item()),
            "fraction_data_independent": float(
                (norm_flat < 0.05).float().mean().item()
            ),
            "n_batch": int(B),
            "n_matrices": int(B * H),
        }
