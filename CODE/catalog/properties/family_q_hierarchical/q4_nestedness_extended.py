"""
q4_nestedness_extended.py — Property Q4 : nestedness H²/HSS étendu.

Spec : DOC/CATALOGUE §Q4 "nestedness : la base d'un cluster est
contenue dans la base de son parent dans la partition binaire récursive.
Mesure clé pour le complexity O(N) strict des HSS".

V1 : partition binaire récursive (depth L = log2(N)). À chaque niveau ℓ,
on calcule rang(A[I_p, J_q]) où (I_p, J_q) sont les blocs admissibles
de la partition. Pour chaque parent (I, J), on regarde ses 2 enfants
(I_left, J_right) et calcule :
    nest_ratio(parent → child) = rang(child) / rang(parent)
Nestedness parfaite (HSS strict) → nest_ratio ≤ 1.
Nestedness violée → nest_ratio > 1 (enfants plus complexes que parents).

Étend Q3 par traversée hierarchique multi-niveau au lieu d'un seul split.
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


def _r_eff(A: torch.Tensor, theta: float, eps_floor: float) -> torch.Tensor:
    s = torch.linalg.svdvals(A)
    s2 = s.pow(2)
    cumsum = s2.cumsum(dim=-1)
    total = cumsum[..., -1:].clamp_min(eps_floor)
    ratio = cumsum / total
    return (ratio >= theta).float().argmax(dim=-1) + 1


@register_property
class Q4NestednessExtended(Property):
    """Q4 — nestedness H²/HSS multi-niveau récursive."""

    name = "Q4_nestedness_extended"
    family = "Q"
    cost_class = 4
    requires_fp64 = True
    scope = "per_regime"

    def __init__(
        self,
        max_depth: int = 3,
        theta_cumulative: float = 0.99,
        eps_floor: float = 1e-30,
    ) -> None:
        self.max_depth = max_depth
        self.theta = theta_cumulative
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, _ = A.shape
        log2N = int(math.log2(N)) if N > 0 else 0
        depth = min(self.max_depth, log2N)
        if depth < 1:
            return {"n_matrices": int(B * H), "skip_reason": "N too small for hierarchy"}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)

        # Calcule rang effectif des blocs off-diagonaux à chaque niveau
        ranks_per_level: dict[int, list[float]] = {}
        for level in range(1, depth + 1):
            n_splits = 2 ** level
            if N % n_splits != 0:
                continue
            block_size = N // n_splits
            level_ranks: list[float] = []
            for i in range(n_splits):
                for j in range(n_splits):
                    if abs(i - j) < 2:  # off-diagonal seulement (séparation HSS)
                        continue
                    sub = A_work[..., i * block_size: (i + 1) * block_size,
                                       j * block_size: (j + 1) * block_size]
                    r = _r_eff(sub, self.theta, self.eps_floor).float()
                    level_ranks.append(float(r.mean().item()))
            if level_ranks:
                ranks_per_level[level] = level_ranks

        if len(ranks_per_level) < 2:
            return {"n_matrices": int(B * H), "skip_reason": "not enough levels"}

        # Nestedness : rang moyen à chaque niveau ; comparé au niveau précédent
        results: dict[str, float | int | str | bool] = {
            "n_matrices": int(B * H),
            "seq_len": int(N),
            "depth_used": int(depth),
        }
        prev_avg = None
        violations = 0
        total_compare = 0
        for level, ranks in sorted(ranks_per_level.items()):
            avg_r = sum(ranks) / len(ranks)
            results[f"avg_rank_level{level}"] = avg_r
            if prev_avg is not None:
                nest_ratio = avg_r / max(prev_avg, self.eps_floor)
                results[f"nest_ratio_level{level}"] = nest_ratio
                if nest_ratio > 1.05:  # tolérance
                    violations += 1
                total_compare += 1
            prev_avg = avg_r
        results["nestedness_violations"] = violations
        results["nestedness_violation_rate"] = (
            float(violations) / max(total_compare, 1)
        )
        results["nestedness_is_hss_strict_proxy"] = bool(violations == 0)
        return results
