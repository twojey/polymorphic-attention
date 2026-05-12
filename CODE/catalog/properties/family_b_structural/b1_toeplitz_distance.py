"""
b1_toeplitz_distance.py — Property B1 : distance de Frobenius à la classe Toeplitz.

Spec : DOC/00b B1.

ε_T(A) = ‖A − P_T(A)‖_F / ‖A‖_F

où P_T est la projection orthogonale Frobenius sur les matrices Toeplitz
(moyenne par diagonale). Une matrice parfaitement Toeplitz a ε_T = 0,
une matrice orthogonale à la classe a ε_T = 1.

Utilisé par Battery A pour identifier la classe dominante d'une attention.
Phase 2 sur SMNIST (2026-05-12) a montré ε_T médian ≈ 0.98 sur l'attention
dense → non-Toeplitz au sens strict, mais ε_T min = 0.45 sur les régimes
dégénérés (ω=2, Δ=0).
"""

from __future__ import annotations

import torch

from catalog.projectors import Toeplitz
from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class B1ToeplitzDistance(Property):
    """B1 — distance de Frobenius normalisée à la classe Toeplitz.

    Calcule ε_T(A) par matrice du batch, retourne stats sur la dim (B, H).
    Le Projector Toeplitz est en cache PropertyContext pour réutilisation
    par Battery B (analyse résidu).
    """

    name = "B1_toeplitz_distance"
    family = "B"
    cost_class = 2  # projection O(N²) + Frobenius norm
    requires_fp64 = False
    requires_symmetric = False
    scope = "per_regime"

    def __init__(self) -> None:
        self._projector = Toeplitz()

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")

        # Cache la projection Toeplitz : Battery B (résidu) la réutilise.
        cache_key = ctx.cache_key(
            "projection_toeplitz", tuple(A.shape), str(A.dtype)
        )

        def _project() -> torch.Tensor:
            A_work = A.to(device=ctx.device, dtype=ctx.dtype)
            return self._projector.project(A_work)

        proj = ctx.get_or_compute(cache_key, _project)
        A_work = A.to(device=ctx.device, dtype=ctx.dtype)

        # ε_T par batch elem
        residual = A_work - proj
        eps = (
            residual.flatten(start_dim=-2).norm(dim=-1)
            / A_work.flatten(start_dim=-2).norm(dim=-1).clamp_min(1e-30)
        )  # (B, H)
        eps_flat = eps.float().flatten()

        return {
            "epsilon_median": float(eps_flat.median().item()),
            "epsilon_mean": float(eps_flat.mean().item()),
            "epsilon_min": float(eps_flat.min().item()),
            "epsilon_max": float(eps_flat.max().item()),
            "epsilon_p10": float(eps_flat.quantile(0.10).item()),
            "epsilon_p90": float(eps_flat.quantile(0.90).item()),
            "n_matrices": int(eps_flat.numel()),
            "fraction_below_0p30": float((eps_flat < 0.30).float().mean().item()),
        }
