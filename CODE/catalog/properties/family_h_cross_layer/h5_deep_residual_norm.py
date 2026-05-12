"""
h5_deep_residual_norm.py — Property H5 : norme du résidu deep layer-by-layer.

Pour chaque couple (ℓ, ℓ+1), mesure ‖A_{ℓ+1} − A_ℓ‖_F / ‖A_ℓ‖_F. Si proche
de 0 : la couche ne fait quasiment rien (résidu identité). Si élevé :
l'attention évolue fortement entre couches.

Diagnostic du phénomène "deep layer plateau" observé sur les LLMs : les
dernières couches modifient peu l'attention.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class H5DeepResidualNorm(Property):
    """H5 — résidu ‖A_{ℓ+1} − A_ℓ‖ / ‖A_ℓ‖ par paire de couches consécutives."""

    name = "H5_deep_residual_norm"
    family = "H"
    cost_class = 1
    requires_fp64 = False
    scope = "per_regime_layers"

    def __init__(self, eps_floor: float = 1e-30) -> None:
        self.eps_floor = eps_floor

    def compute(
        self, A: list[torch.Tensor], ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if not isinstance(A, (list, tuple)) or len(A) < 2:
            return {
                "skip_reason": "need ≥ 2 layers",
                "n_layers": int(len(A) if isinstance(A, (list, tuple)) else 0),
            }
        L = len(A)
        ratios = []
        for l_idx in range(L - 1):
            X = A[l_idx].to(device=ctx.device, dtype=ctx.dtype)
            Y = A[l_idx + 1].to(device=ctx.device, dtype=ctx.dtype)
            if X.shape != Y.shape:
                continue
            num = (Y - X).flatten(start_dim=-2).norm(dim=-1)
            den = X.flatten(start_dim=-2).norm(dim=-1).clamp_min(self.eps_floor)
            ratios.append((num / den).float().flatten())

        if not ratios:
            return {"skip_reason": "all shape mismatch", "n_layers": int(L)}
        all_r = torch.cat(ratios)
        # Indice de plateau : fraction de paires < 0.05
        plateau = (all_r < 0.05).float().mean().item()
        # Max résidu par paire (pic d'évolution)
        per_pair_max = [r.max().item() for r in ratios]
        argmax_pair = int(max(range(len(per_pair_max)), key=lambda i: per_pair_max[i]))

        return {
            "deep_residual_median": float(all_r.median().item()),
            "deep_residual_mean": float(all_r.mean().item()),
            "deep_residual_max": float(all_r.max().item()),
            "fraction_plateau_below_0p05": float(plateau),
            "argmax_residual_layer_pair": argmax_pair,
            "n_layers": int(L),
            "n_pairs": int(all_r.numel()),
        }
