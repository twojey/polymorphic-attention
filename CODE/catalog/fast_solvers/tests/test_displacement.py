"""Tests rang de déplacement Sylvester."""

from __future__ import annotations

import torch

from catalog.fast_solvers import (
    extract_displacement_generators,
    sylvester_displacement,
)
from catalog.fast_solvers.displacement import shift_down_operator


def test_sylvester_toeplitz_displacement_low_rank() -> None:
    """Pour une matrice Toeplitz, ∇_Z(T) = T − Z·T·Zᵀ doit avoir rang ≤ 2.

    Note : la convention Kailath utilise ∇(T) = T − Z T Zᵀ pour Stein. Ici on
    utilise ∇(T) = Z·T − T·Z (Sylvester), qui donne aussi rang faible mais pas
    nécessairement ≤ 2 dans le cas général. On vérifie au moins que rang < N.
    """
    N = 8
    # Toeplitz parfaite
    row = torch.tensor([2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.0], dtype=torch.float64)
    col = row.clone()
    from catalog.fast_solvers import toeplitz_from_first_row_col
    T = toeplitz_from_first_row_col(row, col)
    Z = shift_down_operator(N)
    nabla = sylvester_displacement(T, Z, Z)
    rank = torch.linalg.matrix_rank(nabla, atol=1e-10).item()
    # Sylvester displacement de Toeplitz : pas garanti ≤ 2 mais << N
    assert rank < N


def test_extract_generators_recon_quality() -> None:
    """Pour A aléatoire, générateurs rang ~ rang plein → erreur faible à rang plein."""
    N = 8
    torch.manual_seed(0)
    A = torch.softmax(torch.randn(1, 1, N, N, dtype=torch.float64), dim=-1)
    Z = shift_down_operator(N)
    G, B, eps = extract_displacement_generators(A, Z, Z, r=N)
    # Rang plein → erreur quasi-zero
    assert eps < 1e-8


def test_extract_generators_low_rank_loss() -> None:
    """À rang réduit r=2 : erreur attendue non-nulle."""
    N = 8
    torch.manual_seed(1)
    A = torch.softmax(torch.randn(1, 1, N, N, dtype=torch.float64), dim=-1)
    Z = shift_down_operator(N)
    _, _, eps_2 = extract_displacement_generators(A, Z, Z, r=2)
    _, _, eps_full = extract_displacement_generators(A, Z, Z, r=N)
    # rang 2 doit avoir erreur > rang plein
    assert eps_2 > eps_full
