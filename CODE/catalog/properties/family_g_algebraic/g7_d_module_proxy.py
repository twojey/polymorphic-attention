"""
g7_d_module_proxy.py — Property G7 : proxy D-module / holonomicité.

Spec : DOC/CATALOGUE §G7 "dimension caractéristique d'un D-module
associé, indicateur de l'ordre d'irrégularité".

Un D-module M est holonome si dim_char(M) = n (dimension de Krull
minimale possible). Pour un opérateur d'attention vu comme système
d'équations différentielles formelles ∂_λ A ≈ rotation discrète, on
peut estimer un proxy de l'irrégularité via la **vitesse de
décroissance des dérivées discrètes**.

V1 proxy : pour chaque matrice A, calculer la suite des différences
finies d'ordre k :
    Δ^k A[i, j] = sum_{l=0}^k (-1)^l C(k,l) A[i-l, j]
La décroissance de ‖Δ^k A‖_F vs k informe sur l'ordre de l'opérateur :
- décroissance polynomiale (rang k fixé) → holonome bas ordre
- décroissance lente / quasi-constante → ordre élevé, frontière

Sortie : exposant de décroissance + classement (holonome / régulier /
irrégulier).
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class G7DModuleProxy(Property):
    """G7 — proxy holonomicité / irrégularité D-module via différences finies."""

    name = "G7_d_module_proxy"
    family = "G"
    cost_class = 3
    requires_fp64 = True
    scope = "per_regime"

    def __init__(self, k_max: int = 4, eps_floor: float = 1e-30) -> None:
        self.k_max = k_max
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, _ = A.shape
        if N < self.k_max + 2:
            return {"n_matrices": int(B * H), "skip_reason": f"N<{self.k_max+2}"}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Différence finie itérée le long de la dim row
        diffs_norms: list[float] = []
        current = A_work
        diffs_norms.append(float(current.norm(dim=(-2, -1)).mean().item()))
        for _ in range(self.k_max):
            current = current[..., 1:, :] - current[..., :-1, :]
            diffs_norms.append(float(current.norm(dim=(-2, -1)).mean().item()))

        # Régression log(‖Δ^k A‖) ~ -α · k → α = exposant de décroissance
        ks = torch.arange(len(diffs_norms), dtype=torch.float64)
        ys = torch.tensor([math.log(max(n, self.eps_floor)) for n in diffs_norms],
                          dtype=torch.float64)
        # Régression linéaire least-squares manuelle
        k_mean = ks.mean()
        y_mean = ys.mean()
        num = ((ks - k_mean) * (ys - y_mean)).sum()
        den = ((ks - k_mean) ** 2).sum().clamp_min(self.eps_floor)
        slope = (num / den).item()  # négatif si décroissance
        alpha = -float(slope)  # exposant de décroissance positif

        # Classification frontière
        if alpha > 1.0:
            regime_class = "holonomic_low_order"
        elif alpha > 0.3:
            regime_class = "regular"
        else:
            regime_class = "irregular_frontier"

        return {
            "d_module_decay_alpha": alpha,
            "d_module_log_diff_k0": float(ys[0].item()),
            "d_module_log_diff_kmax": float(ys[-1].item()),
            "d_module_class": regime_class,
            "d_module_is_holonomic_proxy": bool(alpha > 1.0),
            "n_matrices": int(B * H),
            "k_max": int(self.k_max),
        }
