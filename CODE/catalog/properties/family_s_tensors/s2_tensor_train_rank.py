"""
s2_tensor_train_rank.py — Property S2 : rangs Tensor Train (TT) du batch.

Spec : DOC/CATALOGUE §S2.

Pour un tenseur d-dimensionnel T, la décomposition TT factorise comme
T = G_1 ×_2 G_2 ×_2 ... ×_2 G_d où G_i sont des "cores" 3-d. Les rangs
TT (r_1, ..., r_{d-1}) contrôlent la complexité de la décomposition.

Pour T (B, H, N, N) batch d'attention, on a 3 rangs TT possibles à
calculer via SVD itérative TT-SVD. Le rang i est défini par SVD de
l'unfolding "mode-1 to mode-i".

V1 : on calcule via TT-SVD avec θ-cumulative pour la troncature.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class S2TensorTrainRank(Property):
    """S2 — rangs TT (r_1, r_2, r_3) du tenseur 4-d via TT-SVD."""

    name = "S2_tensor_train_rank"
    family = "S"
    cost_class = 4
    requires_fp64 = True
    scope = "per_regime"

    def __init__(self, theta_cumulative: float = 0.99, eps_floor: float = 1e-30) -> None:
        self.theta = theta_cumulative
        self.eps_floor = eps_floor

    def _r_eff(self, M: torch.Tensor) -> int:
        s = torch.linalg.svdvals(M)
        s2 = s.pow(2)
        cumsum = s2.cumsum(dim=-1)
        total = cumsum[-1].clamp_min(self.eps_floor)
        ratio = cumsum / total
        above = ratio >= self.theta
        if above.any():
            return int(above.float().argmax().item()) + 1
        return s.shape[-1]

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # TT-SVD : SVD itératif sur unfoldings (1) | (2,3,4), puis (1,2) | (3,4), etc.
        # r_1 = rank(unfolding mode-1) — déjà = r_batch de Tucker
        unf_1 = A_work.reshape(B, H * N * N2)
        r_1 = self._r_eff(unf_1)

        # r_2 : (B*H, N*N2)
        unf_2 = A_work.reshape(B * H, N * N2)
        r_2 = self._r_eff(unf_2)

        # r_3 : (B*H*N, N2)
        unf_3 = A_work.reshape(B * H * N, N2)
        r_3 = self._r_eff(unf_3)

        # Compression ratio TT vs dense
        # Dense : B·H·N·N. TT : r_1·B + r_1·r_2·H + r_2·r_3·N + r_3·N2 (approx)
        dense_size = B * H * N * N2
        tt_size = r_1 * B + r_1 * r_2 * H + r_2 * r_3 * N + r_3 * N2
        compression = float(tt_size / max(dense_size, 1))

        return {
            "tt_rank_1": r_1,
            "tt_rank_2": r_2,
            "tt_rank_3": r_3,
            "tt_rank_max": max(r_1, r_2, r_3),
            "tt_compression_ratio": compression,
            "B": B, "H": H, "N": N, "N2": N2,
            "theta": self.theta,
        }
