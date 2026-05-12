"""
h6_attention_sink_score.py — Property H6 : "attention sink" trans-couches.

Le phénomène "attention sink" (Xiao et al. 2024) : sur les LLMs, presque
toutes les têtes attribuent une fraction substantielle de leur attention
au tout premier token (BOS).

Score = fraction moyenne d'attention sur A[..., t, 0] (la colonne 0).
Mesuré per layer pour observer son évolution.

Score > 0.30 = sink fort attendu sur LLM dense entraîné.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class H6AttentionSinkScore(Property):
    """H6 — fraction d'attention sur le premier token, per layer."""

    name = "H6_attention_sink_score"
    family = "H"
    cost_class = 1
    requires_fp64 = False
    scope = "per_regime_layers"

    def __init__(self, sink_position: int = 0) -> None:
        self.sink_position = sink_position

    def compute(
        self, A: list[torch.Tensor], ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if not isinstance(A, (list, tuple)) or len(A) == 0:
            return {"skip_reason": "no layers", "n_layers": 0}
        L = len(A)
        per_layer_sink = []
        for X in A:
            X_work = X.to(device=ctx.device, dtype=ctx.dtype)
            if X_work.shape[-1] <= self.sink_position:
                per_layer_sink.append(0.0)
                continue
            # Mean over (B, H, queries) of A[..., t, sink_position]
            col = X_work[..., :, self.sink_position]  # (B, H, N) for each query
            per_layer_sink.append(float(col.float().mean().item()))

        results: dict[str, float | int | str | bool] = {}
        for ell, v in enumerate(per_layer_sink):
            results[f"sink_score_layer_{ell}"] = v
        if per_layer_sink:
            results["sink_score_max_layer"] = float(max(per_layer_sink))
            results["sink_score_min_layer"] = float(min(per_layer_sink))
            results["sink_score_mean_across_layers"] = float(
                sum(per_layer_sink) / len(per_layer_sink)
            )
            results["fraction_layers_with_strong_sink"] = float(
                sum(1 for v in per_layer_sink if v > 0.30) / len(per_layer_sink)
            )
        results["sink_position"] = int(self.sink_position)
        results["n_layers"] = int(L)
        return results
