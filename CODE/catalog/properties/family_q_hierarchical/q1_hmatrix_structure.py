"""
q1_hmatrix_structure.py — Property Q1 : structure H-matrix (rang faible blocs).

Spec : DOC/CATALOGUE §Q1.

Une H-matrix est définie par la propriété que ses **off-diagonal blocs**,
à chaque niveau d'une décomposition hiérarchique, sont de rang faible.
V1 : on découpe A à 2-3 niveaux (N → N/2 → N/4), et pour chaque off-diagonal
bloc on calcule r_eff. Une vraie H-matrix a r_eff faible (typique : ≤ 5)
sur tous les blocs à tous les niveaux.

Reporte la moyenne et le max de r_eff sur les off-diagonal blocs.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


def _r_eff(A: torch.Tensor, theta: float, eps_floor: float) -> torch.Tensor:
    """r_eff(θ) sur une matrice (..., M, N) batchée."""
    s = torch.linalg.svdvals(A)
    s2 = s.pow(2)
    cumsum = s2.cumsum(dim=-1)
    total = cumsum[..., -1:].clamp_min(eps_floor)
    ratio = cumsum / total
    return (ratio >= theta).float().argmax(dim=-1) + 1


@register_property
class Q1HMatrixStructure(Property):
    """Q1 — rang effectif moyen des off-diagonal blocs à 2-3 niveaux."""

    name = "Q1_hmatrix_structure"
    family = "Q"
    cost_class = 3  # SVD × O(2^levels) blocs
    requires_fp64 = True
    scope = "per_regime"

    def __init__(
        self,
        n_levels: int = 2,
        theta_cumulative: float = 0.99,
        eps_floor: float = 1e-30,
    ) -> None:
        if n_levels < 1:
            raise ValueError("n_levels ≥ 1")
        self.n_levels = n_levels
        self.theta = theta_cumulative
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)

        # À chaque niveau ℓ, on découpe en (2^ℓ)² blocs et on regarde les
        # off-diagonal (i ≠ j parmi 2^ℓ paquets)
        results: dict[str, float | int | str | bool] = {}
        all_r_eff_off: list[torch.Tensor] = []

        for level in range(1, self.n_levels + 1):
            n_splits = 2 ** level
            if N < n_splits or N2 < n_splits:
                continue
            block_h = N // n_splits
            block_w = N2 // n_splits

            r_effs_at_level: list[torch.Tensor] = []
            for i in range(n_splits):
                for j in range(n_splits):
                    if i == j:
                        continue  # diagonal blocs ignorés (H-matrix permet rang plein)
                    sub = A_work[..., i * block_h: (i + 1) * block_h,
                                       j * block_w: (j + 1) * block_w]
                    r = _r_eff(sub, self.theta, self.eps_floor).float().flatten()
                    r_effs_at_level.append(r)
                    all_r_eff_off.append(r)

            if r_effs_at_level:
                r_all = torch.cat(r_effs_at_level)
                results[f"r_eff_offdiag_level_{level}_median"] = float(
                    r_all.median().item()
                )
                results[f"r_eff_offdiag_level_{level}_mean"] = float(r_all.mean().item())
                results[f"r_eff_offdiag_level_{level}_max"] = float(r_all.max().item())

        if all_r_eff_off:
            r_global = torch.cat(all_r_eff_off)
            results["r_eff_offdiag_global_median"] = float(r_global.median().item())
            results["r_eff_offdiag_global_max"] = float(r_global.max().item())
            results["fraction_blocs_rang_le_5"] = float(
                (r_global <= 5).float().mean().item()
            )

        results["n_levels"] = self.n_levels
        results["theta"] = self.theta
        return results
