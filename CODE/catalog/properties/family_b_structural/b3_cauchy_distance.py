"""
b3_cauchy_distance.py — Property B3 : distance à la classe Cauchy.

Spec : DOC/CATALOGUE §B3 "distance aux familles paramétriques".

Pour une matrice A donnée, on cherche la meilleure approximation par une
matrice de Cauchy C(x, y) = [1/(y_i - x_j)]_{ij} sur une grille de
paramétrisations standard :

- equispace : x = linspace(0, 1, N), y = x + δ pour δ ∈ {0.1, 0.5, 1.0, 2.0}
- chebyshev : nodes Chebyshev avec offset anti-collision

Pour chaque (x, y), projection orthogonale (rank-1, via Cauchy Projector)
puis ε_C = ‖A − P_C(A)‖_F / ‖A‖_F. On rapporte le min sur la grille.

Cohérent avec O2 (rang de déplacement Cauchy-like) : O2 et B3 sont les
deux faces du même invariant. O2 mesure si rang(∇A) ≤ 1 ; B3 mesure si
A est ε-proche d'une Cauchy concrète.

Cost class 2 : N projections × |grille| (~5).
"""

from __future__ import annotations

import math

import torch

from catalog.projectors import Cauchy
from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


def _equispace_xy(n: int, offset: float, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    y = x + offset
    return x, y


def _chebyshev_xy(n: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.arange(1, n + 1, device=device, dtype=dtype)
    x = torch.cos((2 * k - 1) * math.pi / (2 * n))
    y = torch.cos((k - 0.5) * math.pi / n) + 1.5
    return x, y


@register_property
class B3CauchyDistance(Property):
    """B3 — distance Frobenius minimale à la classe Cauchy sur grille (x, y)."""

    name = "B3_cauchy_distance"
    family = "B"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(
        self,
        offsets: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0),
        include_chebyshev: bool = True,
    ) -> None:
        self.offsets = offsets
        self.include_chebyshev = include_chebyshev

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N != N2:
            raise ValueError(f"A doit être carrée, reçu N={N} != {N2}")

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        device = A_work.device
        dtype = A_work.dtype

        param_grid: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for off in self.offsets:
            x, y = _equispace_xy(N, off, device, dtype)
            tag = f"equi_{off:.1f}".replace(".", "p")
            param_grid.append((tag, x, y))
        if self.include_chebyshev:
            x, y = _chebyshev_xy(N, device, dtype)
            param_grid.append(("cheby", x, y))

        # Pour chaque paramétrisation, projection + ε
        best_eps_per_mat: torch.Tensor | None = None
        per_param: dict[str, float] = {}
        for tag, x, y in param_grid:
            proj = Cauchy(x, y)
            eps = proj.epsilon(A_work)  # (B, H)
            eps_flat = eps.float().flatten()
            per_param[f"{tag}_epsilon_median"] = float(eps_flat.median().item())
            if best_eps_per_mat is None:
                best_eps_per_mat = eps.float()
            else:
                best_eps_per_mat = torch.minimum(best_eps_per_mat, eps.float())

        best_flat = best_eps_per_mat.flatten()
        results: dict[str, float | int | str | bool] = {
            "epsilon_best_median": float(best_flat.median().item()),
            "epsilon_best_mean": float(best_flat.mean().item()),
            "epsilon_best_min": float(best_flat.min().item()),
            "epsilon_best_max": float(best_flat.max().item()),
            "fraction_close_to_cauchy_below_0p30": float(
                (best_flat < 0.30).float().mean().item()
            ),
            "n_param_grid": len(param_grid),
            "n_matrices": int(B * H),
            "seq_len": int(N),
        }
        results.update(per_param)
        return results
