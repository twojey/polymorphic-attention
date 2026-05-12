"""
o3_vandermonde_displacement_rank.py — Property O3 : rang Vandermonde-like.

Spec : DOC/CATALOGUE §O3.

L'opérateur de déplacement Vandermonde standard est :

    ∇_{D_x, Z}(A) = D_x · A − A · Z

où D_x = diag(x) et Z est le shift-down operator. Pour une matrice de
Vandermonde V[i, j] = x_i^j, on a rang(∇ V) ≤ 1 (Pan).

V1 : on teste plusieurs paramétrisations standard (x = equispace,
chebyshev) et on retourne le min rang sur la grille.
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


def _shift_down(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    Z = torch.zeros(n, n, device=device, dtype=dtype)
    if n > 1:
        idx = torch.arange(n - 1, device=device)
        Z[idx + 1, idx] = 1.0
    return Z


@register_property
class O3VandermondeDisplacementRank(Property):
    """O3 — rang effectif de D_x A − A Z sur grille de x."""

    name = "O3_vandermonde_displacement_rank"
    family = "O"
    cost_class = 3
    requires_fp64 = True
    scope = "per_regime"

    def __init__(
        self,
        theta_cumulative: float = 0.99,
        eps_floor: float = 1e-30,
        rank_atol: float = 1e-10,
    ) -> None:
        self.theta = theta_cumulative
        self.eps_floor = eps_floor
        self.rank_atol = rank_atol

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N != N2:
            raise ValueError("A doit être carrée")

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        Z = _shift_down(N, A_work.device, A_work.dtype)

        # Grille de x
        x_grid: list[tuple[str, torch.Tensor]] = [
            ("equi", torch.linspace(0.5, N + 0.5, N, device=A_work.device, dtype=A_work.dtype)),
            ("equi_unit", torch.linspace(0.1, 1.0, N, device=A_work.device, dtype=A_work.dtype)),
            ("chebyshev", torch.cos((2 * torch.arange(1, N + 1, device=A_work.device, dtype=A_work.dtype) - 1) * math.pi / (2 * N)) + 1.5),
        ]

        best_r_eff: torch.Tensor | None = None
        best_strict: torch.Tensor | None = None
        per_param: dict[str, float] = {}
        for tag, x in x_grid:
            DxA = A_work * x.view(1, 1, N, 1)
            AZ = A_work @ Z
            nabla = DxA - AZ
            sigmas = torch.linalg.svdvals(nabla)
            s2 = sigmas.pow(2)
            cumsum = s2.cumsum(dim=-1)
            total = cumsum[..., -1:].clamp_min(self.eps_floor)
            ratio = cumsum / total
            r_eff = (ratio >= self.theta).float().argmax(dim=-1) + 1
            rank_strict = (sigmas > self.rank_atol).sum(dim=-1)
            per_param[f"{tag}_r_eff_median"] = float(r_eff.float().median().item())
            if best_r_eff is None:
                best_r_eff = r_eff.float()
                best_strict = rank_strict.float()
            else:
                best_r_eff = torch.minimum(best_r_eff, r_eff.float())
                best_strict = torch.minimum(best_strict, rank_strict.float())

        results: dict[str, float | int | str | bool] = {
            "best_r_eff_median": float(best_r_eff.flatten().median().item()),
            "best_rank_strict_median": float(best_strict.flatten().median().item()),
            "fraction_best_rank_le_1": float(
                (best_strict <= 1).float().mean().item()
            ),
            "n_param_grid": len(x_grid),
            "n_matrices": int(B * H),
        }
        results.update(per_param)
        return results
