"""
o2_cauchy_displacement_rank.py — Property O2 : rang de déplacement Cauchy-like.

Spec : DOC/CATALOGUE §O2.

Pour une matrice de Cauchy C avec C[i,j] = 1/(y_i − x_j) où x et y sont
deux ensembles de poles distincts, l'opérateur de déplacement Cauchy est :

    ∇_{x,y}(A) = D_y · A − A · D_x

où D_y = diag(y), D_x = diag(x). **Invariant fondamental** : pour toute
matrice de Cauchy C(x, y), rang(∇_{x,y} C) ≤ 1.

Comme on ne connaît pas a priori (x, y), on teste plusieurs paramétrisations
standard :
- equispace : x = [0, 1, ..., N-1] / N, y = x + δ pour δ ∈ {0.1, 0.5, 1.0}
- chebyshev : x = cos((2k-1)π / 2N), y = cos(kπ/N)

et on prend le min du rang effectif sur la grille. Indique "compatibilité
avec une classe de Cauchy" pour une paramétrisation au moins. Si min rang ≪ N,
la matrice est "Cauchy-friendly".
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


def _equispace_xy(n: int, offset: float, device: str, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    y = x + offset
    return x, y


def _chebyshev_xy(n: int, device: str, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.arange(1, n + 1, device=device, dtype=dtype)
    x = torch.cos((2 * k - 1) * math.pi / (2 * n))
    y = torch.cos((k - 0.5) * math.pi / n) + 1.5  # offset pour éviter collision avec x
    return x, y


@register_property
class O2CauchyDisplacementRank(Property):
    """O2 — rang effectif min de D_y·A − A·D_x sur grille de paires (x, y)."""

    name = "O2_cauchy_displacement_rank"
    family = "O"
    cost_class = 3  # SVD × |grid|
    requires_fp64 = True
    scope = "per_regime"

    def __init__(
        self,
        offsets: tuple[float, ...] = (0.1, 0.5, 1.0),
        include_chebyshev: bool = True,
        theta_cumulative: float = 0.99,
        rank_atol: float = 1e-10,
    ) -> None:
        self.offsets = offsets
        self.include_chebyshev = include_chebyshev
        self.theta = theta_cumulative
        self.rank_atol = rank_atol

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N != N2:
            raise ValueError(f"A doit être carrée, reçu N={N} != {N2}")

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)

        # Construit toutes les paires (x, y) à tester
        param_grid: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for off in self.offsets:
            x, y = _equispace_xy(N, off, str(A_work.device), A_work.dtype)
            tag = f"equi_{off:.1f}".replace(".", "p")
            param_grid.append((tag, x, y))
        if self.include_chebyshev:
            x, y = _chebyshev_xy(N, str(A_work.device), A_work.dtype)
            param_grid.append(("cheby", x, y))

        # Pour chaque (x, y), calcule ∇A = D_y A − A D_x et son rang effectif
        best_r_eff_per_mat: torch.Tensor | None = None
        best_strict_per_mat: torch.Tensor | None = None
        per_param: dict[str, dict[str, float]] = {}

        for tag, x, y in param_grid:
            # D_y A : multiplier chaque ligne i par y[i]
            DyA = A_work * y.view(1, 1, N, 1)
            # A D_x : multiplier chaque colonne j par x[j]
            ADx = A_work * x.view(1, 1, 1, N)
            nabla = DyA - ADx  # (B, H, N, N)

            sigmas = torch.linalg.svdvals(nabla)  # (B, H, N)
            cumsum = (sigmas ** 2).cumsum(dim=-1)
            total = cumsum[..., -1:].clamp_min(1e-30)
            ratio = cumsum / total
            r_eff = (ratio >= self.theta).float().argmax(dim=-1) + 1  # (B, H)
            rank_strict = (sigmas > self.rank_atol).sum(dim=-1)  # (B, H)

            r_eff_flat = r_eff.float().flatten()
            rank_strict_flat = rank_strict.float().flatten()
            per_param[tag] = {
                "r_eff_median": float(r_eff_flat.median().item()),
                "rank_strict_median": float(rank_strict_flat.median().item()),
            }

            if best_r_eff_per_mat is None:
                best_r_eff_per_mat = r_eff.float()
                best_strict_per_mat = rank_strict.float()
            else:
                best_r_eff_per_mat = torch.minimum(best_r_eff_per_mat, r_eff.float())
                best_strict_per_mat = torch.minimum(
                    best_strict_per_mat, rank_strict.float()
                )

        best_r_eff_flat = best_r_eff_per_mat.flatten()
        best_strict_flat = best_strict_per_mat.flatten()

        results: dict[str, float | int | str | bool] = {
            "best_r_eff_median": float(best_r_eff_flat.median().item()),
            "best_r_eff_mean": float(best_r_eff_flat.mean().item()),
            "best_rank_strict_median": float(best_strict_flat.median().item()),
            "fraction_best_rank_le_1_strict": float(
                (best_strict_per_mat <= 1.0).float().mean().item()
            ),
            "fraction_best_rank_le_2_strict": float(
                (best_strict_per_mat <= 2.0).float().mean().item()
            ),
            "n_param_grid": len(param_grid),
            "n_matrices": int(B * H),
            "seq_len": int(N),
        }
        for tag, stats in per_param.items():
            for k, v in stats.items():
                results[f"{tag}_{k}"] = v
        return results
