"""
e1_mutual_info.py — Property E1 : mutual information moyenne entre tokens.

Spec : DOC/CATALOGUE §E1 "I(t_1; t_2) via attention".

Approche : on traite chaque ligne d'attention A[t, :] comme une distribution
de probabilité conditionnelle p(j | t). Une "MI globale" peut être estimée
en utilisant la distribution marginale p(j) = (1/N) Σ_t A[t, j] et la
distribution jointe approchée p(t, j) = (1/N) A[t, j].

    MI = Σ_{t, j} p(t, j) log( p(t, j) / (p(t) · p(j)) )
       = Σ_t (1/N) Σ_j A[t, j] log( A[t, j] · N / p(j) )

(en supposant p(t) = 1/N uniforme sur les positions query)

Normalisée par log(N) (max théorique).
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class E1MutualInfo(Property):
    """E1 — mutual information moyenne entre position query et key."""

    name = "E1_mutual_info"
    family = "E"
    cost_class = 1
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

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Renormalise les rows si elles ne somment pas exactement à 1
        row_sum = A_work.sum(dim=-1, keepdim=True).clamp_min(self.eps_floor)
        P_cond = A_work / row_sum  # p(j | t)
        # p(t) = 1/N pour tous t (uniform query prior)
        # p(j) = (1/N) Σ_t p(j | t)
        P_marg = P_cond.mean(dim=-2, keepdim=True)  # (B, H, 1, N2)
        P_marg_safe = P_marg.clamp_min(self.eps_floor)

        # MI = (1/N) Σ_t Σ_j p(j | t) · log( p(j | t) / p(j) )
        # = (1/N) Σ_t KL( p(.|t) ‖ p(.) )
        P_cond_safe = P_cond.clamp_min(self.eps_floor)
        log_ratio = P_cond_safe.log() - P_marg_safe.log()
        kl_per_query = (P_cond * log_ratio).sum(dim=-1)  # (B, H, N), nats
        mi = kl_per_query.mean(dim=-1)  # (B, H), moyenne sur les queries

        # Normaliser par log(N) (max théorique : full conditional certainty)
        mi_norm = mi / math.log(N2)
        mi_flat = mi.float().flatten()
        norm_flat = mi_norm.float().flatten()

        return {
            "mutual_info_median": float(mi_flat.median().item()),
            "mutual_info_mean": float(mi_flat.mean().item()),
            "mutual_info_norm_median": float(norm_flat.median().item()),
            "mutual_info_norm_mean": float(norm_flat.mean().item()),
            "fraction_low_info_norm_below_0p10": float(
                (norm_flat < 0.10).float().mean().item()
            ),
            "n_matrices": int(B * H),
        }
