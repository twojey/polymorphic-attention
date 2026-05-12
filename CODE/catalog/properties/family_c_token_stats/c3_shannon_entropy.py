"""
c3_shannon_entropy.py — Property C3 : entropie de Shannon par ligne.

Spec : DOC/00b §C3 "H(A[t,:])".

H(p) = − Σ p_i log p_i en nats. Normalisation possible : H/log(N) ∈ [0, 1].

Mesure le "spread" de l'attention par token query. Distribution piquée
(one-hot) → H ≈ 0 ; distribution uniforme → H = log N.
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class C3ShannonEntropy(Property):
    """C3 — entropie de Shannon par ligne, normalisée par log(N)."""

    name = "C3_shannon_entropy"
    family = "C"
    cost_class = 1
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self, eps_floor: float = 1e-30) -> None:
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Renormalise au cas où les lignes ne somment pas exactement à 1
        row_sum = A_work.sum(dim=-1, keepdim=True).clamp_min(self.eps_floor)
        P = A_work / row_sum
        P_safe = P.clamp_min(self.eps_floor)
        # H(p) = -Σ p log p (nats). 0 log 0 = 0 par convention.
        log_P = P_safe.log()
        entropy = -(P_safe * log_P).sum(dim=-1)  # (B, H, N), nats
        # Set entries where P=0 to contribute 0 (clamp already handled magnitude)

        log_N = math.log(N2)
        entropy_norm = entropy / log_N  # (B, H, N) ∈ [0, 1]

        ent_flat = entropy.float().flatten()
        norm_flat = entropy_norm.float().flatten()

        return {
            "shannon_entropy_median": float(ent_flat.median().item()),
            "shannon_entropy_mean": float(ent_flat.mean().item()),
            "shannon_entropy_p10": float(ent_flat.quantile(0.10).item()),
            "shannon_entropy_p90": float(ent_flat.quantile(0.90).item()),
            "shannon_entropy_norm_median": float(norm_flat.median().item()),
            "shannon_entropy_norm_mean": float(norm_flat.mean().item()),
            "fraction_rows_norm_below_0p20": float(
                (norm_flat < 0.20).float().mean().item()
            ),
            "fraction_rows_norm_above_0p80": float(
                (norm_flat > 0.80).float().mean().item()
            ),
            "n_rows": int(ent_flat.numel()),
            "log_N": log_N,
        }
