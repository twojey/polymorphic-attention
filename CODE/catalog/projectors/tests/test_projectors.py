"""Tests math pour Projectors (Identity, Toeplitz, Hankel)."""

from __future__ import annotations

import pytest
import torch

from catalog.projectors import Hankel, Identity, Toeplitz


# -----------------------------------------------------------------------------
# Identity
# -----------------------------------------------------------------------------


def test_identity_keeps_diagonal() -> None:
    A = torch.randn(5, 5, dtype=torch.float64)
    proj = Identity().project(A)
    assert torch.allclose(torch.diagonal(proj), torch.diagonal(A))


def test_identity_zeros_off_diagonal() -> None:
    A = torch.randn(5, 5, dtype=torch.float64)
    proj = Identity().project(A)
    off = proj - torch.diag(torch.diagonal(proj))
    assert off.abs().max().item() < 1e-12


def test_identity_epsilon_on_diagonal_matrix_is_zero() -> None:
    A = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    eps = Identity().epsilon(A.unsqueeze(0))  # (1, 3, 3)
    assert eps.item() < 1e-12


def test_identity_rejects_non_square() -> None:
    A = torch.randn(3, 5)
    with pytest.raises(ValueError, match="carrée"):
        Identity().project(A)


# -----------------------------------------------------------------------------
# Toeplitz
# -----------------------------------------------------------------------------


def test_toeplitz_projection_idempotent_on_toeplitz_matrix() -> None:
    """Si A est Toeplitz exacte, P_T(A) ≈ A."""
    n = 6
    c = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
    A = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            A[i, j] = c[abs(i - j)]
    proj = Toeplitz().project(A)
    assert torch.allclose(proj, A, atol=1e-10)


def test_toeplitz_epsilon_on_pure_toeplitz_is_zero() -> None:
    n = 5
    A = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            A[i, j] = float(abs(i - j))
    eps = Toeplitz().epsilon(A.unsqueeze(0))
    assert eps.item() < 1e-10


def test_toeplitz_projection_constant_per_diagonal() -> None:
    """Après projection, chaque diagonale a une valeur constante."""
    torch.manual_seed(0)
    A = torch.randn(4, 4, dtype=torch.float64)
    proj = Toeplitz().project(A)
    for d in range(-(4 - 1), 4):
        diag = torch.diagonal(proj, offset=d)
        if diag.numel() > 1:
            assert diag.std().item() < 1e-12, f"diag offset={d} not constant"


def test_toeplitz_batched() -> None:
    """Vectorisé sur dims batch en préfixe."""
    A = torch.randn(2, 3, 5, 5, dtype=torch.float64)
    proj = Toeplitz().project(A)
    assert proj.shape == A.shape


# -----------------------------------------------------------------------------
# Hankel
# -----------------------------------------------------------------------------


def test_hankel_projection_idempotent_on_hankel_matrix() -> None:
    n = 6
    A = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            A[i, j] = float(i + j)
    proj = Hankel().project(A)
    assert torch.allclose(proj, A, atol=1e-10)


def test_hankel_epsilon_on_pure_hankel_is_zero() -> None:
    n = 5
    A = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            A[i, j] = float(i + j)
    eps = Hankel().epsilon(A.unsqueeze(0))
    assert eps.item() < 1e-10


def test_hankel_projection_constant_per_antidiagonal() -> None:
    torch.manual_seed(0)
    A = torch.randn(4, 4, dtype=torch.float64)
    proj = Hankel().project(A)
    for s in range(4 + 4 - 1):
        positions = [(i, s - i) for i in range(max(0, s - 4 + 1), min(4, s + 1))]
        if len(positions) > 1:
            vals = torch.tensor([proj[i, j].item() for i, j in positions])
            assert vals.std().item() < 1e-12


# -----------------------------------------------------------------------------
# Cross-Projector — comportement sur matrice aléatoire
# -----------------------------------------------------------------------------


def test_random_matrix_epsilon_above_zero_all_projectors() -> None:
    """Sur une matrice aléatoire (non structurée), ε > 0 sur Toeplitz/Hankel/Identity."""
    torch.manual_seed(42)
    A = torch.randn(1, 8, 8, dtype=torch.float64)
    for cls in (Toeplitz, Hankel, Identity):
        eps = cls().epsilon(A)
        assert eps.item() > 0.1, f"{cls.__name__} ε={eps.item()} sur random matrix"


def test_residual_orthogonal_to_projection_frobenius() -> None:
    """Math : <A - P_C(A), P_C(A)>_F = 0 pour projection orthogonale."""
    A = torch.randn(6, 6, dtype=torch.float64)
    for cls in (Toeplitz, Hankel, Identity):
        proj = cls().project(A)
        res = cls().residual(A)
        # <res, proj>_F = somme(res * proj)
        inner = (res * proj).sum().item()
        assert abs(inner) < 1e-10, f"{cls.__name__} non orthogonal : <res, proj>={inner}"
