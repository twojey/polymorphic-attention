"""
b9_quasiseparable.py — Property B9 : structure quasiseparable.

Spec : DOC/CATALOGUE §B9 "rangs de déplacement généralisés (générateurs courts)".

Une matrice est r-quasiseparable si les blocs strictement supérieurs (et
strictement inférieurs) ont rang ≤ r. Différence avec Q2 HSS rank : ici
on regarde TOUS les blocs supérieurs (pas seulement à séparation ≥ 2).

V1 : pour chaque k ∈ {1, .., N//2}, on calcule rang(A[0:k, k:]) (bloc
supérieur de coin) et on prend le max. C'est le **quasiseparable rank**
canonique (Eidelman-Gohberg).
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class B9Quasiseparable(Property):
    """B9 — rang quasiseparable max sur tous les coins supérieurs."""

    name = "B9_quasiseparable"
    family = "B"
    cost_class = 3
    requires_fp64 = True
    scope = "per_regime"

    def __init__(
        self,
        theta_cumulative: float = 0.99,
        eps_floor: float = 1e-30,
        n_corners_max: int = 8,
    ) -> None:
        self.theta = theta_cumulative
        self.eps_floor = eps_floor
        self.n_corners_max = n_corners_max

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N < 4:
            return {"n_matrices": int(B * H), "skip_reason": "N too small"}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Subset des coins {1, 2, 4, ..., N/2} jusqu'à n_corners_max
        candidates = [1, 2, 4, 8, 16, 32, N // 4, N // 2, 3 * N // 4]
        corners = sorted(set(k for k in candidates if 1 <= k < N))[: self.n_corners_max]

        upper_ranks: list[float] = []
        lower_ranks: list[float] = []
        for k in corners:
            # Bloc supérieur droit : A[0:k, k:N2]
            upper = A_work[..., :k, k:]
            sigmas_u = torch.linalg.svdvals(upper)
            s2_u = sigmas_u.pow(2)
            cumsum_u = s2_u.cumsum(dim=-1)
            total_u = cumsum_u[..., -1:].clamp_min(self.eps_floor)
            ratio_u = cumsum_u / total_u
            r_u = (ratio_u >= self.theta).float().argmax(dim=-1) + 1

            # Bloc inférieur gauche : A[k:N, 0:k]
            lower = A_work[..., k:, :k]
            sigmas_l = torch.linalg.svdvals(lower)
            s2_l = sigmas_l.pow(2)
            cumsum_l = s2_l.cumsum(dim=-1)
            total_l = cumsum_l[..., -1:].clamp_min(self.eps_floor)
            ratio_l = cumsum_l / total_l
            r_l = (ratio_l >= self.theta).float().argmax(dim=-1) + 1

            upper_ranks.append(float(r_u.float().median().item()))
            lower_ranks.append(float(r_l.float().median().item()))

        return {
            "qs_upper_rank_max_corner": float(max(upper_ranks)),
            "qs_upper_rank_mean_corners": float(sum(upper_ranks) / len(upper_ranks)),
            "qs_lower_rank_max_corner": float(max(lower_ranks)),
            "qs_lower_rank_mean_corners": float(sum(lower_ranks) / len(lower_ranks)),
            "qs_combined_max": float(max(max(upper_ranks), max(lower_ranks))),
            "n_corners_tested": len(corners),
            "n_matrices": int(B * H),
            "theta": self.theta,
        }
