"""
r1_mercer_psd.py — Property R1 : test Mercer (semi-définie positive).

Spec : DOC/CATALOGUE §R1.

Une matrice de noyau Mercer K doit être semi-définie positive (PSD) :
toutes ses valeurs propres ≥ 0. Pour une attention A non-nécessairement
symétrique, on teste la PSD-ness sur A_sym = (A + Aᵀ) / 2 (symétrique
partie) en regardant ses eigvalsh.

Métriques :
- fraction_psd : fraction de matrices avec λ_min ≥ -tol
- min_eigenvalue_median : λ_min médiane (négatif si pas PSD)
- nullity : nombre d'eigvals < tol (rang manquant)
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class R1MercerPSD(Property):
    """R1 — test PSD sur A_sym (partie symétrique de A)."""

    name = "R1_mercer_psd"
    family = "R"
    cost_class = 2
    requires_fp64 = True
    scope = "per_regime"

    def __init__(self, eigvalsh_tol: float = -1e-10) -> None:
        self.tol = eigvalsh_tol

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N != N2:
            raise ValueError(f"A doit être carrée")

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        A_sym = 0.5 * (A_work + A_work.transpose(-1, -2))

        eigs = torch.linalg.eigvalsh(A_sym)  # (B, H, N), triés croissants

        lambda_min = eigs[..., 0]  # plus petit eigenvalue
        lambda_max = eigs[..., -1]
        is_psd = lambda_min >= self.tol  # (B, H) bool
        # Nullity = nb eigvals "presque zero"
        nullity = (eigs.abs() < 1e-8).sum(dim=-1).float()  # (B, H)

        lmin_flat = lambda_min.float().flatten()
        lmax_flat = lambda_max.float().flatten()
        nullity_flat = nullity.flatten()

        return {
            "psd_fraction": float(is_psd.float().mean().item()),
            "min_eigenvalue_median": float(lmin_flat.median().item()),
            "min_eigenvalue_mean": float(lmin_flat.mean().item()),
            "max_eigenvalue_median": float(lmax_flat.median().item()),
            "nullity_median": float(nullity_flat.median().item()),
            "nullity_max": float(nullity_flat.max().item()),
            "tol": self.tol,
            "n_matrices": int(B * H),
            "seq_len": int(N),
        }
