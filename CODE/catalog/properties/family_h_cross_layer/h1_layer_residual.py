"""
h1_layer_residual.py — Property H1 : résidu entre couches successives.

Spec : DOC/CATALOGUE §H1.

ε_H1(ℓ) = ‖A_ℓ − A_ℓ₊₁‖_F / ‖A_ℓ‖_F

Mesure à quel point l'attention change d'une couche à la suivante. Si
ε est petit, les couches sont quasi-identiques (peut-être redondantes
ou en plateau). Si grand, transformation forte.

Sortie : moyenne / médiane sur tous les couples (ℓ, ℓ+1) + delta L1
entre première et dernière couche.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class H1LayerResidual(Property):
    """H1 — distance relative entre couches consécutives."""

    name = "H1_layer_residual"
    family = "H"
    cost_class = 1
    requires_fp64 = False
    scope = "per_regime_layers"

    def compute(
        self, attn_layers: list[torch.Tensor], ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if not isinstance(attn_layers, list) or len(attn_layers) < 2:
            raise ValueError(
                f"H1 requiert au moins 2 couches, reçu {len(attn_layers) if isinstance(attn_layers, list) else type(attn_layers).__name__}"
            )

        L = len(attn_layers)
        per_pair_eps: list[torch.Tensor] = []
        for ell in range(L - 1):
            A1 = attn_layers[ell].to(device=ctx.device, dtype=ctx.dtype)
            A2 = attn_layers[ell + 1].to(device=ctx.device, dtype=ctx.dtype)
            if A1.shape != A2.shape:
                raise ValueError(
                    f"H1 : couches ℓ={ell} et ℓ={ell+1} ont des shapes "
                    f"différentes : {A1.shape} vs {A2.shape}"
                )
            diff = (A1 - A2).flatten(start_dim=-2).norm(dim=-1)  # (B, H)
            denom = A1.flatten(start_dim=-2).norm(dim=-1).clamp_min(1e-30)
            per_pair_eps.append((diff / denom).flatten())

        all_eps = torch.cat(per_pair_eps).float()
        results: dict[str, float | int | str | bool] = {
            "epsilon_consecutive_median": float(all_eps.median().item()),
            "epsilon_consecutive_mean": float(all_eps.mean().item()),
            "epsilon_consecutive_min": float(all_eps.min().item()),
            "epsilon_consecutive_max": float(all_eps.max().item()),
        }

        # Aussi : delta cumulative ℓ=0 → ℓ=L-1
        A_first = attn_layers[0].to(device=ctx.device, dtype=ctx.dtype)
        A_last = attn_layers[-1].to(device=ctx.device, dtype=ctx.dtype)
        full_diff = (A_first - A_last).flatten(start_dim=-2).norm(dim=-1)
        first_norm = A_first.flatten(start_dim=-2).norm(dim=-1).clamp_min(1e-30)
        eps_full = (full_diff / first_norm).float().flatten()
        results["epsilon_first_last_median"] = float(eps_full.median().item())
        results["epsilon_first_last_mean"] = float(eps_full.mean().item())

        # Stats par paire (pour identifier couches "plateau")
        per_pair_medians: list[float] = []
        for ell, eps in enumerate(per_pair_eps):
            per_pair_medians.append(float(eps.float().median().item()))
            results[f"epsilon_layer_{ell}_to_{ell+1}_median"] = per_pair_medians[-1]

        # Monotonie : combien de paires consécutives ont ε décroissant ?
        decreasing_pairs = sum(
            1 for i in range(len(per_pair_medians) - 1)
            if per_pair_medians[i + 1] < per_pair_medians[i]
        )
        results["fraction_pairs_decreasing"] = float(
            decreasing_pairs / max(1, len(per_pair_medians) - 1)
        )

        results["n_layer_pairs"] = L - 1
        results["n_layers"] = L
        return results
