"""
g8_syzygies_proxy.py — Property G8 : proxy syzygies (relations linéaires entre colonnes).

Spec : DOC/CATALOGUE §G8 "module des syzygies S = ker(A · v = 0)".

Pour une matrice A, le module des syzygies est le noyau de l'application
v → A·v. Sa dimension dim ker(A) = N − rang(A). Au-delà du rang trivial,
les **relations linéaires non triviales** entre colonnes contribuent
aux syzygies, et leurs degrés (Hilbert) encodent la complexité.

V1 proxy numérique :
- rang numérique nullité = N − rang(A, tol)
- énergie kernel : Σ σ_i² (i > rang) / Σ σ_i² (toutes)
- "depth" Hilbert simplifiée : log(nullité) / log(N)

Si A est génériquement plein rang → nullité = 0, depth = 0
Si A est rang-1 (structure simple) → nullité = N−1, depth ~ 1
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class G8SyzygiesProxy(Property):
    """G8 — proxy syzygies via nullité numérique + énergie noyau."""

    name = "G8_syzygies_proxy"
    family = "G"
    cost_class = 2
    requires_fp64 = True
    scope = "per_regime"

    def __init__(self, tol_relative: float = 1e-8, eps_floor: float = 1e-30) -> None:
        self.tol_relative = tol_relative
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, _ = A.shape

        sigmas = ctx.svdvals_cached(A)  # (B, H, N) — cache partagé Famille A
        max_per = sigmas.amax(dim=-1, keepdim=True).clamp_min(self.eps_floor)
        ratio = sigmas / max_per
        # Nullité numérique = nb σ_i < tol · σ_max
        nullity = (ratio < self.tol_relative).float().sum(dim=-1)  # (B, H)

        # Énergie kernel : Σ σ_i² au-delà du seuil
        s2 = sigmas.pow(2)
        s2_max = s2.amax(dim=-1, keepdim=True).clamp_min(self.eps_floor)
        kernel_mask = (s2 / s2_max) < self.tol_relative
        kernel_energy = (s2 * kernel_mask).sum(dim=-1)
        total_energy = s2.sum(dim=-1).clamp_min(self.eps_floor)
        kernel_frac = (kernel_energy / total_energy)

        # Hilbert depth proxy : log(nullité+1) / log(N+1)
        log_N = math.log(N + 1)
        depth = (nullity + 1.0).log() / log_N

        return {
            "syzygies_nullity_median": float(nullity.float().median().item()),
            "syzygies_nullity_max": float(nullity.float().max().item()),
            "syzygies_nullity_fraction": float(
                (nullity.float() / N).median().item()
            ),
            "syzygies_kernel_energy_frac_median": float(kernel_frac.float().median().item()),
            "syzygies_depth_proxy_median": float(depth.float().median().item()),
            "fraction_full_rank": float((nullity == 0).float().mean().item()),
            "n_matrices": int(B * H),
            "N": int(N),
        }
