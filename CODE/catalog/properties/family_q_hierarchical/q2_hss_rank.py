"""
q2_hss_rank.py — Property Q2 : rang Hierarchically Semi-Separable (HSS).

Spec : DOC/CATALOGUE §Q2.

Une matrice HSS a la propriété que pour tout couple (I, J) d'indices
"hierarchiquement séparés" (i.e., pas adjacents dans l'arbre de partition),
le bloc A[I, J] est de rang faible.

V1 : moyenne du rang effectif sur les blocs strictement off-diagonaux (i.e.,
‖i - j‖_∞ ≥ 2 dans la grille de blocs) à un seul niveau de splitting.
Donne une estimation "HSS rank" du système.
"""

from __future__ import annotations

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
class Q2HSSRank(Property):
    """Q2 — rang effectif des blocs strictement off-diagonaux (sépération ≥ 2)."""

    name = "Q2_hss_rank"
    family = "Q"
    cost_class = 3
    requires_fp64 = True
    scope = "per_regime"

    def __init__(
        self,
        n_splits: int = 4,
        separation: int = 2,
        theta_cumulative: float = 0.99,
        eps_floor: float = 1e-30,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits ≥ 2")
        if separation < 1:
            raise ValueError("separation ≥ 1")
        self.n_splits = n_splits
        self.separation = separation
        self.theta = theta_cumulative
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N < self.n_splits or N2 < self.n_splits:
            return {
                "n_matrices": int(B * H),
                "skip_reason": "N too small for n_splits",
            }

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        block_h = N // self.n_splits
        block_w = N2 // self.n_splits

        r_effs: list[torch.Tensor] = []
        n_blocs_separated = 0
        for i in range(self.n_splits):
            for j in range(self.n_splits):
                if abs(i - j) < self.separation:
                    continue
                sub = A_work[..., i * block_h: (i + 1) * block_h,
                                   j * block_w: (j + 1) * block_w]
                r = _r_eff(sub, self.theta, self.eps_floor).float().flatten()
                r_effs.append(r)
                n_blocs_separated += 1

        if not r_effs:
            return {"n_matrices": int(B * H), "skip_reason": "no separated blocs"}

        r_all = torch.cat(r_effs)
        return {
            "hss_rank_median": float(r_all.median().item()),
            "hss_rank_mean": float(r_all.mean().item()),
            "hss_rank_max": float(r_all.max().item()),
            "hss_rank_min": float(r_all.min().item()),
            "fraction_hss_rank_le_3": float((r_all <= 3).float().mean().item()),
            "n_separated_blocs": n_blocs_separated,
            "n_matrices": int(B * H),
            "separation": self.separation,
        }
