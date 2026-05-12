"""
s5_unfolding_rank.py — Property S5 : rangs des unfoldings paires.

Pour le tenseur (B, H, N, N), au-delà des 4 unfoldings simples (cf. S1),
on calcule les rangs des **paires d'unfoldings** :
- mode (0,1) : (BH × NN)
- mode (0,2) : (BN × HN)
- mode (1,2) : (HN × BN)

Cela donne 3 nouveaux rangs structurels. Permet de différencier les
structures par paires : un rang faible sur (1,2) = signature "blocs" sur
la dimension head×row.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


def _r_eff_pair(
    tensor: torch.Tensor,
    modes_left: tuple[int, ...],
    theta: float,
    eps_floor: float,
) -> int:
    n_dims = tensor.ndim
    modes_right = tuple(i for i in range(n_dims) if i not in modes_left)
    perm = list(modes_left) + list(modes_right)
    shape = tensor.shape
    left_size = 1
    for m in modes_left:
        left_size *= shape[m]
    unf = tensor.permute(*perm).reshape(left_size, -1)
    s = torch.linalg.svdvals(unf)
    s2 = s.pow(2)
    cumsum = s2.cumsum(dim=-1)
    total = cumsum[-1].clamp_min(eps_floor)
    ratio = cumsum / total
    above = ratio >= theta
    if above.any():
        return int(above.float().argmax().item()) + 1
    return s.shape[-1]


@register_property
class S5UnfoldingRank(Property):
    """S5 — r_eff sur unfoldings de paires d'axes (0,1), (0,2), (1,2)."""

    name = "S5_unfolding_rank"
    family = "S"
    cost_class = 4
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

        r_bh = _r_eff_pair(A_work, (0, 1), self.theta, self.eps_floor)
        r_br = _r_eff_pair(A_work, (0, 2), self.theta, self.eps_floor)
        r_hr = _r_eff_pair(A_work, (1, 2), self.theta, self.eps_floor)

        # Tailles potentielles
        size_bh = B * H
        size_br = B * N
        size_hr = H * N

        return {
            "rank_unfold_batch_head": int(r_bh),
            "rank_unfold_batch_row": int(r_br),
            "rank_unfold_head_row": int(r_hr),
            "compression_batch_head": float(r_bh / max(size_bh, 1)),
            "compression_batch_row": float(r_br / max(size_br, 1)),
            "compression_head_row": float(r_hr / max(size_hr, 1)),
            "B": B, "H": H, "N": N,
            "theta": self.theta,
        }
