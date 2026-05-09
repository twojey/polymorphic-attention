"""Tests des 3 signaux + agrégation. Sur matrices synthétiques."""

from __future__ import annotations

import torch

from phase1b_calibration_signal.signals.aggregation import aggregate_signal_per_token
from phase1b_calibration_signal.signals.s_kl import GlobalKLBaseline, compute_s_kl
from phase1b_calibration_signal.signals.s_spectral import compute_s_spectral


def _fake_attention(L: int = 2, B: int = 2, H: int = 4, N: int = 16) -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.softmax(torch.randn(B, H, N, N, dtype=torch.float64), dim=-1) for _ in range(L)]


def test_baseline_construction_normalizes() -> None:
    dumps = _fake_attention()
    baseline = GlobalKLBaseline.from_attention_dumps(dumps)
    sums = baseline.baseline.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-10)


def test_s_kl_zero_when_attention_matches_baseline() -> None:
    # Si A = baseline, KL = 0
    L, B, H, N = 2, 1, 2, 8
    baseline_dist = torch.full((H, N), 1.0 / N, dtype=torch.float64)
    baseline_per_layer = baseline_dist.unsqueeze(0).expand(L, -1, -1).contiguous()
    baseline = GlobalKLBaseline(baseline=baseline_per_layer)
    A = torch.full((B, H, N, N), 1.0 / N, dtype=torch.float64)
    s_kl = compute_s_kl([A] * L, baseline)
    assert torch.allclose(s_kl, torch.zeros_like(s_kl), atol=1e-9)


def test_s_kl_positive_when_attention_diverges() -> None:
    L, B, H, N = 2, 1, 2, 8
    baseline_dist = torch.full((H, N), 1.0 / N, dtype=torch.float64)
    baseline = GlobalKLBaseline(baseline=baseline_dist.unsqueeze(0).expand(L, -1, -1).contiguous())
    # A concentré sur le premier token : très éloigné du uniform baseline
    A = torch.zeros(B, H, N, N, dtype=torch.float64)
    A[..., 0] = 1.0
    s_kl = compute_s_kl([A] * L, baseline)
    assert (s_kl > 0).all()


def test_s_spectral_uniform_attention_low_rank() -> None:
    # A uniforme ⇒ rang 1
    L, B, H, N = 1, 1, 1, 16
    A = torch.full((B, H, N, N), 1.0 / N, dtype=torch.float64)
    s = compute_s_spectral([A], K=8, tau=1e-3)
    assert s.shape == (L, B, H, N)
    # Hors warmup, rang ≤ 2 (ligne uniforme = rang 1)
    assert (s[..., 8:] <= 2).all()


def test_aggregation_shapes_and_values() -> None:
    L, B, H, N = 4, 3, 8, 16
    signal = torch.zeros(L, B, H, N, dtype=torch.float64)
    # Tête 3 = max sur la couche 1, position 5
    signal[1, :, 3, 5] = 7.0
    aggregated = aggregate_signal_per_token(signal)
    assert aggregated.shape == (B, L, N)
    assert (aggregated[:, 1, 5] == 7.0).all()
