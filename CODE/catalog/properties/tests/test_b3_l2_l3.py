"""Tests pour B3 Cauchy distance + L2 Haar wavelets + L3 quasi-périodicité."""

from __future__ import annotations

import math

import torch

from catalog.projectors import Cauchy
from catalog.properties.base import PropertyContext
from catalog.properties.family_b_structural.b3_cauchy_distance import (
    B3CauchyDistance,
    _equispace_xy,
)
from catalog.properties.family_l_frequency.l2_haar_wavelets import L2HaarWavelets
from catalog.properties.family_l_frequency.l3_quasi_periodicity import (
    L3QuasiPeriodicity,
)
from catalog.properties.registry import REGISTRY


# -----------------------------------------------------------------------------
# Cauchy projector standalone
# -----------------------------------------------------------------------------


def test_cauchy_projector_on_real_cauchy() -> None:
    """C(x, y) construite exactement → epsilon(C) = 0."""
    n = 8
    x = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    y = x + 0.5
    proj = Cauchy(x, y)
    C = proj._build_cauchy(torch.device("cpu"), torch.float64)
    A = C.unsqueeze(0).unsqueeze(0)
    eps = proj.epsilon(A)
    # La projection rank-1 sur span(C) capture parfaitement C
    assert eps.max().item() < 1e-10


def test_cauchy_projector_rejects_size_mismatch() -> None:
    import pytest

    x = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    y = x + 0.5
    proj = Cauchy(x, y)
    A = torch.randn(1, 1, 16, 16, dtype=torch.float64)
    with pytest.raises(ValueError, match="A doit être"):
        proj.project(A)


# -----------------------------------------------------------------------------
# B3 CauchyDistance
# -----------------------------------------------------------------------------


def test_b3_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(2, 3, 8, 8, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = B3CauchyDistance().compute(A, ctx)
    assert "epsilon_best_median" in out
    assert "fraction_close_to_cauchy_below_0p30" in out


def test_b3_cauchy_matrix_low_epsilon() -> None:
    """Matrice Cauchy construite (avec equi δ=0.5) → epsilon_best très petit."""
    n = 8
    x, y = _equispace_xy(n, 0.5, torch.device("cpu"), torch.float64)
    C = Cauchy(x, y)._build_cauchy(torch.device("cpu"), torch.float64)
    A = C.unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    # Forcer la grille à inclure δ=0.5
    out = B3CauchyDistance(offsets=(0.5,), include_chebyshev=False).compute(A, ctx)
    assert out["epsilon_best_median"] < 1e-10


def test_b3_random_matrix_high_epsilon() -> None:
    """Matrice aléatoire → epsilon_best élevé (proche de 1)."""
    torch.manual_seed(0)
    A = torch.randn(1, 1, 16, 16, dtype=torch.float64)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = B3CauchyDistance().compute(A, ctx)
    assert out["epsilon_best_median"] > 0.5


def test_b3_registered() -> None:
    assert REGISTRY.get("B3_cauchy_distance") is B3CauchyDistance


# -----------------------------------------------------------------------------
# L2 HaarWavelets
# -----------------------------------------------------------------------------


def test_l2_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(2, 3, 8, 8, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = L2HaarWavelets().compute(A, ctx)
    assert "energy_LL_fraction_median" in out
    assert "energy_high_freq_fraction_median" in out
    # 4 fractions individuellement ∈ [0, 1]
    for k in ["energy_LL_fraction_median", "energy_LH_fraction_median",
              "energy_HL_fraction_median", "energy_HH_fraction_median"]:
        assert 0.0 <= out[k] <= 1.0


def test_l2_constant_matrix_all_in_LL() -> None:
    """Matrice constante → toute l'énergie dans LL (basse fréq)."""
    A = torch.full((1, 1, 8, 8), 0.5, dtype=torch.float64)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = L2HaarWavelets().compute(A, ctx)
    assert out["energy_LL_fraction_median"] > 0.999
    assert out["energy_high_freq_fraction_median"] < 1e-6


def test_l2_alternating_signal_has_high_freq() -> None:
    """Pattern alterné (1,0,1,0) → toute l'énergie dans HH."""
    n = 8
    A = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            A[i, j] = 1.0 if (i + j) % 2 == 0 else 0.0
    A = A.unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = L2HaarWavelets().compute(A, ctx)
    assert out["energy_high_freq_fraction_median"] > 0.1


def test_l2_registered() -> None:
    assert REGISTRY.get("L2_haar_wavelets") is L2HaarWavelets


# -----------------------------------------------------------------------------
# L3 QuasiPeriodicity
# -----------------------------------------------------------------------------


def test_l3_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(2, 3, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = L3QuasiPeriodicity().compute(A, ctx)
    assert "best_period_median" in out
    assert "peak_amplitude_median" in out
    assert "fraction_quasi_periodic" in out


def test_l3_periodic_signal_detects_period() -> None:
    """Pattern périodique : sin(2π·k·t/period) → autocorr peak à 'period'."""
    n = 64
    period = 8
    t = torch.arange(n, dtype=torch.float64)
    signal = torch.sin(2 * math.pi * t / period)
    # Construit A où chaque ligne est ce signal
    A = signal.unsqueeze(0).expand(n, n).unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = L3QuasiPeriodicity(min_lag=2).compute(A, ctx)
    # Période détectée doit être proche de 8
    assert abs(out["best_period_median"] - period) < 2  # tolérance lag
    assert out["peak_amplitude_median"] > 0.5
    assert out["fraction_quasi_periodic"] > 0.5


def test_l3_random_signal_low_peak() -> None:
    """Bruit blanc : autocorr peak faible (pas de période)."""
    torch.manual_seed(0)
    A = torch.randn(1, 1, 64, 64, dtype=torch.float64)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = L3QuasiPeriodicity(min_lag=2).compute(A, ctx)
    assert out["fraction_quasi_periodic"] < 0.5


def test_l3_rejects_invalid_min_lag() -> None:
    import pytest

    with pytest.raises(ValueError, match="min_lag"):
        L3QuasiPeriodicity(min_lag=0)


def test_l3_registered() -> None:
    assert REGISTRY.get("L3_quasi_periodicity") is L3QuasiPeriodicity
