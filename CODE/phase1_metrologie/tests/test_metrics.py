"""Tests métriques phase 1.4 — rang Hankel, entropie spectrale.

Sur des matrices synthétiques où l'on connaît la vérité analytique, on vérifie
que les métriques se comportent comme attendu.
"""

from __future__ import annotations

import math

import torch

from phase1_metrologie.metrics.hankel import (
    hankel_rank_of_signal,
    hankelize,
    numerical_rank,
)
from phase1_metrologie.metrics.spectral import normalized_spectral_entropy, spectral_entropy


# ----------------------------------------------------------------
# Hankel
# ----------------------------------------------------------------


def test_hankelize_shape() -> None:
    x = torch.arange(10, dtype=torch.float64)
    H = hankelize(x, p=4)
    assert H.shape == (4, 7)
    assert H[0, 0].item() == 0.0
    assert H[0, 6].item() == 6.0
    assert H[3, 0].item() == 3.0
    assert H[3, 6].item() == 9.0


def test_numerical_rank_full() -> None:
    A = torch.eye(8, dtype=torch.float64)
    assert numerical_rank(A, tau=1e-3) == 8


def test_numerical_rank_zero() -> None:
    A = torch.zeros(5, 5, dtype=torch.float64)
    assert numerical_rank(A) == 0


def test_numerical_rank_outer_product() -> None:
    u = torch.randn(8, dtype=torch.float64)
    v = torch.randn(8, dtype=torch.float64)
    A = u.unsqueeze(1) @ v.unsqueeze(0)
    assert numerical_rank(A, tau=1e-6) == 1


def test_hankel_rank_constant_signal_is_one() -> None:
    # Un signal constant ⇒ Hankel = matrice de tous identiques ⇒ rang 1
    x = torch.ones(32, dtype=torch.float64) * 3.14
    assert hankel_rank_of_signal(x, p=16, tau=1e-6) == 1


def test_hankel_rank_pure_exponential_is_one() -> None:
    # Un signal x_k = α · r^k a une matrice de Hankel de rang 1 (système 1er ordre)
    n = 32
    r = 0.7
    x = torch.tensor([r**k for k in range(n)], dtype=torch.float64)
    assert hankel_rank_of_signal(x, p=16, tau=1e-6) == 1


def test_hankel_rank_sum_of_two_exponentials_is_two() -> None:
    # x_k = α · r1^k + β · r2^k → rang Hankel = 2 (système 2nd ordre)
    n = 64
    r1, r2 = 0.6, 0.9
    x = torch.tensor([r1**k + 0.5 * r2**k for k in range(n)], dtype=torch.float64)
    assert hankel_rank_of_signal(x, p=32, tau=1e-9) == 2


def test_hankel_rank_random_signal_high() -> None:
    torch.manual_seed(0)
    x = torch.randn(64, dtype=torch.float64)
    rank = hankel_rank_of_signal(x, p=32, tau=1e-3)
    # Signal aléatoire ≈ pleine échelle de rang
    assert rank > 16


# ----------------------------------------------------------------
# Spectral entropy
# ----------------------------------------------------------------


def test_spectral_entropy_rank_one_is_zero() -> None:
    u = torch.randn(16, dtype=torch.float64)
    v = torch.randn(16, dtype=torch.float64)
    A = u.unsqueeze(1) @ v.unsqueeze(0)
    H = spectral_entropy(A)
    assert abs(H.item()) < 1e-9


def test_spectral_entropy_uniform_is_log_n() -> None:
    # Matrice avec valeurs singulières égales ⇒ H = log(N)
    N = 8
    s = torch.ones(N, dtype=torch.float64)
    Q = torch.linalg.qr(torch.randn(N, N, dtype=torch.float64))[0]
    A = Q @ torch.diag(s) @ Q.T
    H = spectral_entropy(A).item()
    assert abs(H - math.log(N)) < 1e-6


def test_spectral_entropy_normalized_in_unit_interval() -> None:
    torch.manual_seed(1)
    A = torch.softmax(torch.randn(32, 32, dtype=torch.float64), dim=-1)
    H_norm = normalized_spectral_entropy(A).item()
    assert 0.0 <= H_norm <= 1.0 + 1e-6


def test_spectral_entropy_batched() -> None:
    torch.manual_seed(0)
    A = torch.randn(4, 8, 16, 16, dtype=torch.float64)
    A = torch.softmax(A, dim=-1)
    H = spectral_entropy(A)
    assert H.shape == (4, 8)
    assert (H >= 0).all()
    assert (H <= math.log(16) + 1e-6).all()
