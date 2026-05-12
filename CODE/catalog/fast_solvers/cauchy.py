"""
cauchy.py — Algorithmes pour matrices Cauchy.

Spec : DOC/CATALOGUE §1.2 (Kailath §3.5), §O2.

Une matrice Cauchy C(x, y) ∈ ℝ^{N×N} est définie par C[i, j] = 1 / (x_i − y_j).
Propriétés :
- rang de déplacement Stein-Cauchy : Δ_C = diag(x) C − C diag(y) a rang 1 (= 1·1ᵀ)
- factorisation GKO (Gohberg-Kailath-Olshevsky 1995) : Cauchy système
  résolu en O(N²) avec coefficients de pivotage
- multiplication Cauchy matvec : O(N log² N) via Trummer-Tyrtyshnikov

V1 : matvec naïf O(N²) + résolution Gauss avec pivot O(N³) (référence
mathématique). V2 si nécessaire : Trummer O(N log² N).

API :
- `cauchy_matrix(x, y)` : construit la matrice C ∈ ℝ^{N×M}
- `cauchy_matvec_naive(x, y, v)` : produit C·v en O(N²)
- `cauchy_solve(x, y, b)` : résout C·u = b via Gauss + pivot (référence)
"""

from __future__ import annotations

import torch


def cauchy_matrix(
    x: torch.Tensor, y: torch.Tensor, *, eps_floor: float = 1e-12
) -> torch.Tensor:
    """Construit la matrice Cauchy C[i, j] = 1 / (x_i − y_j).

    Args
    ----
    x : Tensor (N,) — nœuds "rows"
    y : Tensor (M,) — nœuds "cols"
    eps_floor : seuil pour éviter division par zéro si x_i = y_j

    Returns
    -------
    C : Tensor (N, M)
    """
    if x.dim() != 1 or y.dim() != 1:
        raise ValueError(f"x, y doivent être 1D, reçus x.dim()={x.dim()}, y.dim()={y.dim()}")
    diff = x.unsqueeze(-1) - y.unsqueeze(0)  # (N, M)
    if (diff.abs() < eps_floor).any():
        raise ValueError("Cauchy matrix singulière : x_i == y_j pour au moins une paire")
    return 1.0 / diff


def cauchy_matvec_naive(
    x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, *, eps_floor: float = 1e-12
) -> torch.Tensor:
    """Produit C·v où C est Cauchy(x, y), en O(N²) naïf.

    V2 : Trummer O(N log² N) via FFT et fast multipole.
    """
    C = cauchy_matrix(x, y, eps_floor=eps_floor)
    return C @ v


def cauchy_solve(
    x: torch.Tensor, y: torch.Tensor, b: torch.Tensor, *, eps_floor: float = 1e-12,
) -> torch.Tensor:
    """Résout C·u = b où C est Cauchy(x, y) carrée.

    V1 : Gauss + pivot via `torch.linalg.solve`. C'est O(N³) mais correct.
    V2 (futur) : algorithme GKO O(N²) si performance nécessaire.

    Args
    ----
    x : Tensor (N,) — nœuds rows (distincts)
    y : Tensor (N,) — nœuds cols (distincts, x_i ≠ y_j pour tout i, j)
    b : Tensor (N,) ou (N, K) — RHS
    """
    C = cauchy_matrix(x, y, eps_floor=eps_floor)
    if b.dim() == 1:
        return torch.linalg.solve(C, b.unsqueeze(-1)).squeeze(-1)
    return torch.linalg.solve(C, b)
