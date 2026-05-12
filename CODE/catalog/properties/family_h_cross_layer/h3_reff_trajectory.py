"""
h3_reff_trajectory.py — Property H3 : trajectoire r_eff(ℓ) à travers la profondeur.

Spec : DOC/CATALOGUE §H3.

Pour chaque couche ℓ, calcule r_eff(A_ℓ) (cf. A1) et retourne la trajectoire.
Permet de répondre :
- Le rang spectral monte-il avec la profondeur ?
- Stabilité (plateau) ou évolution monotone ?
- Couche où r_eff atteint son max ?

Reuse de la logique A1 — on factorise mathematically le calcul (svdvals
+ ratio cumulatif).
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


def _r_eff(A: torch.Tensor, theta: float) -> torch.Tensor:
    """r_eff(θ) : nb val. singulières nécessaires pour θ % de l'énergie.

    A : (..., M, N). Retourne (...,) entier ≥ 1.
    """
    sigmas = torch.linalg.svdvals(A)
    cumsum = (sigmas ** 2).cumsum(dim=-1)
    total = cumsum[..., -1:].clamp_min(1e-30)
    ratio = cumsum / total
    above = ratio >= theta
    return above.float().argmax(dim=-1) + 1


@register_property
class H3REffTrajectory(Property):
    """H3 — trajectoire de r_eff(ℓ) à travers les couches d'un régime."""

    name = "H3_reff_trajectory"
    family = "H"
    cost_class = 3  # SVD par couche
    requires_fp64 = True
    scope = "per_regime_layers"

    def __init__(self, theta_cumulative: float = 0.99) -> None:
        if not 0.0 < theta_cumulative <= 1.0:
            raise ValueError(f"theta={theta_cumulative} doit être ∈ (0, 1]")
        self.theta = theta_cumulative

    def compute(
        self, attn_layers: list[torch.Tensor], ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if not isinstance(attn_layers, list) or len(attn_layers) < 1:
            raise ValueError(f"H3 requiert au moins 1 couche")

        L = len(attn_layers)
        per_layer_medians: list[float] = []
        per_layer_means: list[float] = []
        per_layer_maxs: list[float] = []

        for ell, A in enumerate(attn_layers):
            A_work = A.to(device=ctx.device, dtype=ctx.dtype)
            r = _r_eff(A_work, theta=self.theta).float().flatten()
            per_layer_medians.append(float(r.median().item()))
            per_layer_means.append(float(r.mean().item()))
            per_layer_maxs.append(float(r.max().item()))

        results: dict[str, float | int | str | bool] = {}
        for ell in range(L):
            results[f"r_eff_layer_{ell}_median"] = per_layer_medians[ell]
            results[f"r_eff_layer_{ell}_mean"] = per_layer_means[ell]
            results[f"r_eff_layer_{ell}_max"] = per_layer_maxs[ell]

        # Stats trajectoire
        results["r_eff_layer_min"] = float(min(per_layer_medians))
        results["r_eff_layer_max"] = float(max(per_layer_medians))
        results["r_eff_layer_argmax"] = int(
            per_layer_medians.index(max(per_layer_medians))
        )
        results["r_eff_layer_argmin"] = int(
            per_layer_medians.index(min(per_layer_medians))
        )
        # Range et trend : monotonie ?
        results["r_eff_layer_range"] = float(
            max(per_layer_medians) - min(per_layer_medians)
        )

        # Trend monotone strict croissant / décroissant
        diffs = [
            per_layer_medians[i + 1] - per_layer_medians[i] for i in range(L - 1)
        ]
        if diffs:
            results["fraction_increasing_layers"] = float(
                sum(1 for d in diffs if d > 0) / len(diffs)
            )
            results["fraction_decreasing_layers"] = float(
                sum(1 for d in diffs if d < 0) / len(diffs)
            )

        results["n_layers"] = L
        results["theta"] = self.theta
        return results
