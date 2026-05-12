"""
h4_layer_convergence.py — Property H4 : convergence layer→∞.

Spec : DOC/CATALOGUE §H4 "distance A_ℓ vs A_∞ extrapolé".

Pour mesurer si l'attention "se stabilise" en profondeur, on extrapole
A_∞ comme la moyenne des dernières couches (par défaut 2 dernières) et
mesure :
- ε_∞(ℓ) = ‖A_ℓ − A_∞‖_F / ‖A_∞‖_F par couche
- Trend monotone (les ε diminuent vers la fin ?)
- Plateau : à partir de quelle couche ε < seuil (0.10)
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class H4LayerConvergence(Property):
    """H4 — convergence vers A_∞ (moyenne des dernières couches)."""

    name = "H4_layer_convergence"
    family = "H"
    cost_class = 1
    requires_fp64 = False
    scope = "per_regime_layers"

    def __init__(
        self,
        n_last_for_infinity: int = 2,
        plateau_threshold: float = 0.10,
        eps_floor: float = 1e-30,
    ) -> None:
        if n_last_for_infinity < 1:
            raise ValueError(f"n_last_for_infinity doit être ≥ 1")
        self.n_last = n_last_for_infinity
        self.threshold = plateau_threshold
        self.eps_floor = eps_floor

    def compute(
        self, attn_layers: list[torch.Tensor], ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if not isinstance(attn_layers, list) or len(attn_layers) < 2:
            raise ValueError(f"H4 requiert ≥ 2 couches")

        L = len(attn_layers)
        n_last = min(self.n_last, L)

        # Moyenne des n_last dernières couches → A_∞
        last_layers = torch.stack([
            attn_layers[i].to(device=ctx.device, dtype=ctx.dtype)
            for i in range(L - n_last, L)
        ], dim=0)  # (n_last, B, H, N, N)
        A_inf = last_layers.mean(dim=0)  # (B, H, N, N)
        A_inf_norm = A_inf.flatten(start_dim=-2).norm(dim=-1).clamp_min(self.eps_floor)

        results: dict[str, float | int | str | bool] = {}
        eps_per_layer: list[float] = []
        for ell in range(L):
            A = attn_layers[ell].to(device=ctx.device, dtype=ctx.dtype)
            diff = (A - A_inf).flatten(start_dim=-2).norm(dim=-1)
            eps = (diff / A_inf_norm).float()
            med = float(eps.median().item())
            eps_per_layer.append(med)
            results[f"eps_to_infinity_layer_{ell}_median"] = med

        # Plateau : première couche où eps < threshold
        plateau_layer = L  # default = pas de plateau atteint
        for ell, e in enumerate(eps_per_layer):
            if e < self.threshold:
                plateau_layer = ell
                break
        results["plateau_layer"] = plateau_layer
        results["plateau_reached"] = bool(plateau_layer < L)

        # Trend monotone décroissant
        diffs = [eps_per_layer[i + 1] - eps_per_layer[i] for i in range(L - 1)]
        decreasing = sum(1 for d in diffs if d < 0)
        results["fraction_decreasing"] = float(decreasing / max(1, len(diffs)))
        results["eps_first_layer"] = eps_per_layer[0]
        results["eps_last_layer"] = eps_per_layer[-1]
        results["eps_range"] = max(eps_per_layer) - min(eps_per_layer)

        results["n_layers"] = L
        results["n_last_for_infinity"] = n_last
        results["threshold"] = self.threshold
        return results
