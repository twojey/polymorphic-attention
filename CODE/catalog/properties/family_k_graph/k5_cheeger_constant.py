"""
k5_cheeger_constant.py — Property K5 : constante de Cheeger (graph conductance).

Pour un graphe construit à partir de A (poids w_ij = A_ij), la constante
de Cheeger h(G) mesure la "bottleneck" minimale :
    h(G) = min_{S ⊂ V, |S| ≤ |V|/2}  cut(S, S̄) / vol(S)

Approximation rapide (Cheeger upper bound) :
    h(G) ≤ √(2 λ_2)  où λ_2 est la 2nd eigval du Laplacien normalisé.

On approxime h par cette borne. Faible h = graphe avec bottleneck (peut
être split en 2). Élevé h = bien connecté.

Reuse possible du Laplacien spectrum cf. K1.
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class K5CheegerConstant(Property):
    """K5 — borne supérieure Cheeger h ≤ √(2 λ_2) du Laplacien normalisé."""

    name = "K5_cheeger_constant"
    family = "K"
    cost_class = 3
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self, eps_floor: float = 1e-30) -> None:
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N != N2:
            return {"skip_reason": "non-square", "n_matrices": int(B * H)}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Symétriser pour Laplacien
        W = 0.5 * (A_work + A_work.transpose(-1, -2))
        # Degré
        d = W.sum(dim=-1).clamp_min(self.eps_floor)  # (B, H, N)
        d_inv_sqrt = d.rsqrt()
        # L_norm = I − D^{-1/2} W D^{-1/2}
        Dinv = d_inv_sqrt.unsqueeze(-1) * d_inv_sqrt.unsqueeze(-2)  # (B, H, N, N)
        L = torch.eye(N, device=A_work.device, dtype=A_work.dtype) - Dinv * W
        # eigvalsh (L_norm est symétrique)
        try:
            eig = torch.linalg.eigvalsh(L)  # (B, H, N) sorted ascending
        except Exception:
            return {"skip_reason": "eigvalsh failed", "n_matrices": int(B * H)}
        # λ_1 = 0 (graphe connecté), λ_2 = "algebraic connectivity"
        lam2 = eig[..., 1].clamp_min(0.0)
        cheeger_upper = (2.0 * lam2).sqrt()  # h ≤ √(2 λ_2)

        c_flat = cheeger_upper.float().flatten()
        l_flat = lam2.float().flatten()

        return {
            "cheeger_upper_median": float(c_flat.median().item()),
            "cheeger_upper_mean": float(c_flat.mean().item()),
            "cheeger_upper_min": float(c_flat.min().item()),
            "lambda2_laplacian_median": float(l_flat.median().item()),
            "fraction_bottleneck_below_0p10": float(
                (c_flat < 0.10).float().mean().item()
            ),
            "n_matrices": int(B * H),
        }
