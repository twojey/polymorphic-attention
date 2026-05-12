"""
e3_redundancy.py — Property E3 : redondance d'information.

R(A) = 1 − H(A) / H_max où H(A) est l'entropie Shannon (ligne par ligne,
moyennée) et H_max = log(N) (entropie uniforme).

R ∈ [0, 1] : 0 = uniforme (aucune redondance), 1 = concentrée (max redondance).
Différent de E2 (LZ proxy) : R mesure la concentration sur les lignes
post-softmax, sans quantification.
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class E3Redundancy(Property):
    """E3 — redondance d'information R = 1 − H/H_max."""

    name = "E3_redundancy"
    family = "E"
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
        B, H, N, _ = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype).clamp_min(self.eps_floor)
        H_row = -(A_work * A_work.log()).sum(dim=-1)  # (B, H, N), nats
        H_max = math.log(N) if N > 1 else 1.0
        R = 1.0 - H_row / H_max  # (B, H, N)
        R_flat = R.float().flatten()

        return {
            "redundancy_median": float(R_flat.median().item()),
            "redundancy_mean": float(R_flat.mean().item()),
            "redundancy_p90": float(R_flat.quantile(0.90).item()),
            "fraction_above_0p5": float((R_flat > 0.5).float().mean().item()),
            "H_max_nats": float(H_max),
            "n_matrices": int(B * H),
        }
