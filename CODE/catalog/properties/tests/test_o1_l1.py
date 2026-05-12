"""Tests pour O1 (Toeplitz displacement rank) et L1 (FFT 2D energy)."""

from __future__ import annotations

import torch

from catalog.properties.base import PropertyContext
from catalog.properties.family_l_frequency.l1_fft2d_energy import L1FFT2DEnergy
from catalog.properties.family_o_displacement.o1_toeplitz_displacement_rank import (
    O1ToeplitzDisplacementRank,
    _shift_down,
)
from catalog.properties.registry import REGISTRY


# -----------------------------------------------------------------------------
# O1 ToeplitzDisplacementRank
# -----------------------------------------------------------------------------


def test_o1_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(2, 3, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = O1ToeplitzDisplacementRank().compute(A, ctx)
    assert "displacement_rank_eff_median" in out
    assert out["n_matrices"] == 6
    assert out["seq_len"] == 16


def test_o1_perfect_toeplitz_has_rank_le_2() -> None:
    """Toeplitz par construction → rang(∇A) ≤ 2."""
    n = 12
    c = torch.randn(n, dtype=torch.float64)  # première colonne
    r = torch.randn(n, dtype=torch.float64)
    r[0] = c[0]
    # Construire Toeplitz A[i,j] = c[i-j] si i>=j sinon r[j-i]
    A_toep = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            if i >= j:
                A_toep[i, j] = c[i - j]
            else:
                A_toep[i, j] = r[j - i]
    A = A_toep.unsqueeze(0).unsqueeze(0)  # (1, 1, n, n)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = O1ToeplitzDisplacementRank(rank_threshold_atol=1e-8).compute(A, ctx)
    # Spec Kailath : rang ≤ 2 sur Toeplitz dense
    assert out["displacement_rank_strict_max"] <= 2
    assert out["fraction_rank_le_2_strict"] == 1.0


def test_o1_random_matrix_has_high_rank() -> None:
    """Matrice aléatoire générique → rang ≈ N."""
    torch.manual_seed(0)
    A = torch.randn(1, 1, 16, 16, dtype=torch.float64)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = O1ToeplitzDisplacementRank(rank_threshold_atol=1e-8).compute(A, ctx)
    # Quasi plein rang attendu (au plus N − 1 par construction shift)
    assert out["displacement_rank_strict_median"] >= 10


def test_o1_shift_down_correct() -> None:
    Z = _shift_down(4, device="cpu", dtype=torch.float64)
    expected = torch.tensor(
        [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        dtype=torch.float64,
    )
    assert torch.allclose(Z, expected)


def test_o1_registered() -> None:
    assert REGISTRY.get("O1_toeplitz_displacement_rank") is O1ToeplitzDisplacementRank


# -----------------------------------------------------------------------------
# L1 FFT2DEnergy
# -----------------------------------------------------------------------------


def test_l1_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(2, 3, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = L1FFT2DEnergy().compute(A, ctx)
    assert "low_freq_energy_fraction_median" in out
    assert "spectral_entropy_norm_median" in out
    assert out["n_matrices"] == 6


def test_l1_uniform_matrix_all_in_dc() -> None:
    """A constante → toute l'énergie est en DC."""
    A = torch.ones(1, 1, 8, 8, dtype=torch.float64) / 8.0
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = L1FFT2DEnergy().compute(A, ctx)
    assert out["dc_energy_fraction_median"] > 0.999
    # entropie spectrale ≈ 0 (un seul pixel)
    assert out["spectral_entropy_norm_median"] < 0.01


def test_l1_identity_spectrum_is_anti_diagonal() -> None:
    """I_n → FFT 2D = anti-diagonale (N pixels non-nuls sur N²),
    entropie normalisée = log(N)/log(N²) = 1/2."""
    n = 8
    A = torch.eye(n, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = L1FFT2DEnergy().compute(A, ctx)
    # entropie spectrale normalisée doit être ≈ 0.5
    assert abs(out["spectral_entropy_norm_median"] - 0.5) < 0.01


def test_l1_registered() -> None:
    assert REGISTRY.get("L1_fft2d_energy") is L1FFT2DEnergy
