"""
g2_symmetry.py — Property G2 : symétrie / antisymétrie.

Spec : DOC/CATALOGUE §G2.

ε_sym(A) = ‖A − Aᵀ‖_F / ‖A‖_F
ε_anti(A) = ‖A + Aᵀ‖_F / ‖A‖_F

Note : A = A_sym + A_anti où A_sym = (A + Aᵀ)/2, A_anti = (A − Aᵀ)/2, et
‖A‖² = ‖A_sym‖² + ‖A_anti‖² (décomposition orthogonale). On peut donc
définir directement la part symétrique et antisymétrique de la norme :

    sym_fraction = ‖A_sym‖² / ‖A‖²
    anti_fraction = ‖A_anti‖² / ‖A‖²

avec sym_fraction + anti_fraction = 1.

Pour attention softmax causale, on attend anti_fraction proche de 1 (la
causalité tue la symétrie). Pour attention bidirectionnelle, sym_fraction
peut monter selon la tâche.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class G2Symmetry(Property):
    """G2 — fraction symétrique vs antisymétrique de A."""

    name = "G2_symmetry"
    family = "G"
    cost_class = 1
    requires_fp64 = False
    scope = "per_regime"

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N != N2:
            raise ValueError(f"A doit être carrée, reçu N={N} != {N2}")

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        At = A_work.transpose(-1, -2)
        A_sym = 0.5 * (A_work + At)
        A_anti = 0.5 * (A_work - At)

        norm_sq = A_work.flatten(start_dim=-2).pow(2).sum(dim=-1)  # (B, H)
        sym_sq = A_sym.flatten(start_dim=-2).pow(2).sum(dim=-1)
        anti_sq = A_anti.flatten(start_dim=-2).pow(2).sum(dim=-1)

        denom = norm_sq.clamp_min(1e-30)
        sym_frac = (sym_sq / denom).float().flatten()
        anti_frac = (anti_sq / denom).float().flatten()

        # ε_sym = ‖A − Aᵀ‖_F / ‖A‖_F (norme du "défaut de symétrie")
        eps_sym = (A_work - At).flatten(start_dim=-2).norm(dim=-1)
        eps_sym_rel = (eps_sym / norm_sq.sqrt().clamp_min(1e-30)).float().flatten()

        return {
            "sym_fraction_median": float(sym_frac.median().item()),
            "sym_fraction_mean": float(sym_frac.mean().item()),
            "anti_fraction_median": float(anti_frac.median().item()),
            "anti_fraction_mean": float(anti_frac.mean().item()),
            "epsilon_asymmetry_median": float(eps_sym_rel.median().item()),
            "epsilon_asymmetry_mean": float(eps_sym_rel.mean().item()),
            "fraction_quasi_symmetric_0p10": float(
                (eps_sym_rel < 0.10).float().mean().item()
            ),
            "n_matrices": int(B * H),
        }
