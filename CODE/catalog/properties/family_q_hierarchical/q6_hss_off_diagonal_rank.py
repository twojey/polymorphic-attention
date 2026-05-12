"""
q6_hss_off_diagonal_rank.py — Property Q6 : rang des sous-blocs off-diagonaux.

Pour une partition récursive de A en 4 quadrants
    [A11 A12]
    [A21 A22]
on calcule r_eff(A12) et r_eff(A21) (les blocs hors-diagonaux).

HSS strict ⇔ ces rangs sont bornés indépendamment de N. On rapporte
r_eff médian et max sur (B, H).

Complète Q1-Q5 par une mesure directe HSS qui ne dépend pas d'une
factorisation H-matrix complète.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class Q6HssOffDiagonalRank(Property):
    """Q6 — r_eff des blocs off-diagonaux A12, A21 (partition top-down 2x2)."""

    name = "Q6_hss_off_diagonal_rank"
    family = "Q"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self, theta_cumulative: float = 0.99, eps_floor: float = 1e-30) -> None:
        self.theta = theta_cumulative
        self.eps_floor = eps_floor

    def _r_eff(self, X: torch.Tensor) -> torch.Tensor:
        s = torch.linalg.svdvals(X)
        s2 = s.pow(2)
        cumsum = s2.cumsum(dim=-1)
        total = cumsum[..., -1:].clamp_min(self.eps_floor)
        ratio = cumsum / total
        above = ratio >= self.theta
        return (above.float().argmax(dim=-1) + 1).float()

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N != N2 or N < 4:
            return {"skip_reason": "non-square or N<4", "n_matrices": int(B * H)}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        m = N // 2
        A12 = A_work[..., :m, m:]
        A21 = A_work[..., m:, :m]

        r12 = self._r_eff(A12).flatten()
        r21 = self._r_eff(A21).flatten()
        max_off = torch.max(r12, r21)

        return {
            "off_diag_rank_top_right_median": float(r12.median().item()),
            "off_diag_rank_bottom_left_median": float(r21.median().item()),
            "off_diag_rank_max_median": float(max_off.median().item()),
            "off_diag_rank_max_p90": float(max_off.quantile(0.90).item()),
            "fraction_hss_compatible_below_4": float(
                (max_off < 4).float().mean().item()
            ),
            "block_size_each": int(m),
            "n_matrices": int(B * H),
        }
