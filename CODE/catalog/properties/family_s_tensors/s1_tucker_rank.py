"""
s1_tucker_rank.py — Property S1 : rangs Tucker n-mode du tenseur batch.

Spec : DOC/CATALOGUE §S1.

Le batch d'attentions (B, H, N, N) est un tenseur 4-d. Sa décomposition
Tucker est T = G ×_1 U_1 ×_2 U_2 ×_3 U_3 ×_4 U_4, où G est le core tensor
et U_i sont des matrices orthogonales. Le rang Tucker (r_1, r_2, r_3, r_4)
est défini par r_i = rang du i-ème mode-unfolding.

V1 : on calcule r_eff sur chaque unfolding (les 4 modes : batch, head, row, col).
Donne une signature tensorielle structurelle.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


def _r_eff_unfolding(tensor: torch.Tensor, mode: int, theta: float, eps_floor: float) -> int:
    """r_eff sur l'unfolding mode-i du tenseur tensor."""
    # Move axis `mode` to position 0, flatten others
    n_dims = tensor.ndim
    perm = [mode] + [i for i in range(n_dims) if i != mode]
    unfolded = tensor.permute(*perm).reshape(tensor.shape[mode], -1)
    s = torch.linalg.svdvals(unfolded)
    s2 = s.pow(2)
    cumsum = s2.cumsum(dim=-1)
    total = cumsum[-1].clamp_min(eps_floor)
    ratio = cumsum / total
    above = ratio >= theta
    if above.any():
        return int(above.float().argmax().item()) + 1
    return s.shape[-1]


@register_property
class S1TuckerRank(Property):
    """S1 — rang Tucker (r_1, r_2, r_3, r_4) du tenseur batch (B, H, N, N)."""

    name = "S1_tucker_rank"
    family = "S"
    cost_class = 4  # SVD sur 4 unfoldings, dont 2 grands (mode-3, mode-4)
    requires_fp64 = True
    scope = "per_regime"

    def __init__(self, theta_cumulative: float = 0.99, eps_floor: float = 1e-30) -> None:
        self.theta = theta_cumulative
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # 4 modes : batch, head, row, col
        r_batch = _r_eff_unfolding(A_work, mode=0, theta=self.theta, eps_floor=self.eps_floor)
        r_head = _r_eff_unfolding(A_work, mode=1, theta=self.theta, eps_floor=self.eps_floor)
        r_row = _r_eff_unfolding(A_work, mode=2, theta=self.theta, eps_floor=self.eps_floor)
        r_col = _r_eff_unfolding(A_work, mode=3, theta=self.theta, eps_floor=self.eps_floor)

        # Compression ratio : produit(r_i) / produit(N_i) (proxy size)
        sizes = [B, H, N, N2]
        rs = [r_batch, r_head, r_row, r_col]
        compression = 1.0
        for r, s in zip(rs, sizes):
            compression *= (r / s)

        return {
            "tucker_rank_batch": r_batch,
            "tucker_rank_head": r_head,
            "tucker_rank_row": r_row,
            "tucker_rank_col": r_col,
            "tucker_rank_max": max(rs),
            "tucker_rank_min": min(rs),
            "tucker_compression_ratio": float(compression),
            "B": B, "H": H, "N": N, "N2": N2,
            "theta": self.theta,
        }
