"""
q3_nestedness.py — Property Q3 : nestedness (rang petit bloc / rang parent).

Spec : DOC/CATALOGUE §Q3.

Une matrice hiérarchique a la propriété de **nestedness** : le rang d'un
bloc enfant est ≤ rang du bloc parent. Mesure utile pour valider une
décomposition Hierarchical Tucker / Hierarchical Matrix.

V1 : on découpe A à 2 niveaux (parent N/2, enfant N/4) et on calcule
le ratio rang_enfant / rang_parent moyen.
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
class Q3Nestedness(Property):
    """Q3 — ratio r_eff(enfant) / r_eff(parent), proxy nestedness hiérarchique."""

    name = "Q3_nestedness"
    family = "Q"
    cost_class = 3
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
        if N < 8 or N2 < 8:
            return {"n_matrices": int(B * H), "skip_reason": "N too small"}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Bloc parent (top-left quadrant) : A[0:N/2, 0:N/2]
        # Bloc enfant : A[0:N/4, 0:N/4]
        h_p = N // 2
        w_p = N2 // 2
        h_c = N // 4
        w_c = N2 // 4

        parent = A_work[..., :h_p, :w_p]
        child = A_work[..., :h_c, :w_c]

        r_parent = _r_eff(parent, self.theta, self.eps_floor)
        r_child = _r_eff(child, self.theta, self.eps_floor)

        ratio = r_child.float() / r_parent.float().clamp_min(1.0)
        ratio_flat = ratio.float().flatten()

        # Une vraie hiérarchie : ratio < 1 (rang enfant strictement plus faible)
        # Si ratio = 1 : pas de hiérarchie (rang plat)
        # Si ratio > 1 : artefact (ne devrait pas arriver avec un quadrant strict)
        return {
            "nestedness_ratio_median": float(ratio_flat.median().item()),
            "nestedness_ratio_mean": float(ratio_flat.mean().item()),
            "r_parent_median": float(r_parent.float().median().item()),
            "r_child_median": float(r_child.float().median().item()),
            "fraction_strict_nested_ratio_below_0p7": float(
                (ratio_flat < 0.7).float().mean().item()
            ),
            "n_matrices": int(B * H),
        }
