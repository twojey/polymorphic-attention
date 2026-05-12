"""
h2_layer_composition.py — Property H2 : composition cross-layer.

Spec : DOC/CATALOGUE §H2 "A_ℓ × A_ℓ₊₁ (propagation effective)".

Pour chaque paire (ℓ, ℓ+1), calcule le produit A_ℓ · A_ℓ₊₁ et compare :
- À A_ℓ (rang produit vs rang individuel)
- À A_ℓ₊₁ (idem)

Pour deux matrices stochastiques en lignes, le produit est aussi
stochastique en lignes (composition Markov). On mesure la similarité
cosinus entre produit et chaque facteur, ainsi que le rang effectif du
produit.

Scope per_regime_layers : reçoit toutes les couches d'un régime.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class H2LayerComposition(Property):
    """H2 — composition A_ℓ · A_ℓ₊₁ et son rang effectif."""

    name = "H2_layer_composition"
    family = "H"
    cost_class = 3
    requires_fp64 = True
    scope = "per_regime_layers"

    def __init__(self, theta_cumulative: float = 0.99, eps_floor: float = 1e-30) -> None:
        self.theta = theta_cumulative
        self.eps_floor = eps_floor

    def compute(
        self, attn_layers: list[torch.Tensor], ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if not isinstance(attn_layers, list) or len(attn_layers) < 2:
            raise ValueError(f"H2 requiert ≥ 2 couches")

        L = len(attn_layers)
        results: dict[str, float | int | str | bool] = {}
        r_eff_products: list[float] = []
        cos_with_left: list[float] = []
        cos_with_right: list[float] = []

        for ell in range(L - 1):
            A1 = attn_layers[ell].to(device=ctx.device, dtype=ctx.dtype)
            A2 = attn_layers[ell + 1].to(device=ctx.device, dtype=ctx.dtype)
            if A1.shape != A2.shape:
                continue

            prod = A1 @ A2  # (B, H, N, N)
            # r_eff du produit
            sigmas = torch.linalg.svdvals(prod)
            cumsum = (sigmas ** 2).cumsum(dim=-1)
            total = cumsum[..., -1:].clamp_min(self.eps_floor)
            ratio = cumsum / total
            r_eff = (ratio >= self.theta).float().argmax(dim=-1) + 1  # (B, H)
            r_eff_products.append(float(r_eff.float().median().item()))

            # Cosine sim avec A1 et A2
            f_prod = prod.flatten(start_dim=-2)
            f_A1 = A1.flatten(start_dim=-2)
            f_A2 = A2.flatten(start_dim=-2)
            cos1 = (f_prod * f_A1).sum(dim=-1) / (
                f_prod.norm(dim=-1) * f_A1.norm(dim=-1)
            ).clamp_min(self.eps_floor)
            cos2 = (f_prod * f_A2).sum(dim=-1) / (
                f_prod.norm(dim=-1) * f_A2.norm(dim=-1)
            ).clamp_min(self.eps_floor)
            cos_with_left.append(float(cos1.float().median().item()))
            cos_with_right.append(float(cos2.float().median().item()))

            results[f"r_eff_product_layer_{ell}_{ell+1}_median"] = r_eff_products[-1]
            results[f"cos_prod_with_left_{ell}_median"] = cos_with_left[-1]
            results[f"cos_prod_with_right_{ell}_median"] = cos_with_right[-1]

        if r_eff_products:
            results["r_eff_product_global_median"] = float(
                torch.tensor(r_eff_products).median().item()
            )
            results["cos_prod_left_global_mean"] = float(
                torch.tensor(cos_with_left).mean().item()
            )
            results["cos_prod_right_global_mean"] = float(
                torch.tensor(cos_with_right).mean().item()
            )

        results["n_layer_pairs"] = L - 1
        results["theta"] = self.theta
        return results
