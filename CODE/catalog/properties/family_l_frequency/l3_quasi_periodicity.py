"""
l3_quasi_periodicity.py — Property L3 : quasi-périodicité par autocorrélation.

Spec : DOC/CATALOGUE §L3 "autocorrélation, peaks de Fourier".

Pour chaque ligne A[t,:], calcule l'autocorrélation 1D et cherche le
plus grand peak hors-zero (lag > 0). Si peak strong → ligne quasi-
périodique avec période ≈ lag_peak. Sinon → non-périodique.

Métriques :
- best_period_median : période dominante (lag du peak)
- peak_amplitude : ratio peak/zero-lag (1 = parfaite périodicité)
- fraction_quasi_periodic : peak > 0.5 × zero-lag
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class L3QuasiPeriodicity(Property):
    """L3 — peaks d'autocorrélation par ligne (détection quasi-périodicité)."""

    name = "L3_quasi_periodicity"
    family = "L"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(
        self,
        min_lag: int = 2,
        peak_ratio_threshold: float = 0.5,
        eps_floor: float = 1e-30,
    ) -> None:
        if min_lag < 1:
            raise ValueError(f"min_lag doit être ≥ 1")
        self.min_lag = min_lag
        self.peak_ratio_threshold = peak_ratio_threshold
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N2 < self.min_lag + 1:
            return {
                "n_matrices": int(B * H),
                "best_period_median": 0.0,
                "peak_amplitude_median": 0.0,
                "fraction_quasi_periodic": 0.0,
            }

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Centrer chaque ligne (autocorr classique)
        mean = A_work.mean(dim=-1, keepdim=True)
        centered = A_work - mean  # (B, H, N, N2)

        # Autocorr via FFT 1D : autocorr(x) = IFFT(|FFT(x)|²)
        # On utilise rfft pour signal réel.
        F = torch.fft.rfft(centered, dim=-1, n=2 * N2)  # next pow 2 implicite via doubling
        power = (F.real ** 2 + F.imag ** 2)
        autocorr = torch.fft.irfft(power, dim=-1, n=2 * N2)[..., :N2]  # (B, H, N, N2)

        # Normaliser par autocorr[lag=0] = variance
        zero_lag = autocorr[..., 0:1].clamp_min(self.eps_floor)
        ac_norm = autocorr / zero_lag  # (B, H, N, N2), ∈ [-1, 1]

        # Chercher peak max sur lags ≥ min_lag
        if self.min_lag >= N2:
            return {
                "n_matrices": int(B * H),
                "best_period_median": 0.0,
                "peak_amplitude_median": 0.0,
                "fraction_quasi_periodic": 0.0,
            }
        ac_relevant = ac_norm[..., self.min_lag:]  # (B, H, N, N2-min_lag)
        peak_amp, peak_idx = ac_relevant.max(dim=-1)  # (B, H, N)
        peak_period = peak_idx + self.min_lag  # ramené au lag absolu

        amp_flat = peak_amp.float().flatten()
        per_flat = peak_period.float().flatten()

        return {
            "best_period_median": float(per_flat.median().item()),
            "best_period_mean": float(per_flat.mean().item()),
            "peak_amplitude_median": float(amp_flat.median().item()),
            "peak_amplitude_mean": float(amp_flat.mean().item()),
            "peak_amplitude_p90": float(amp_flat.quantile(0.90).item()),
            "fraction_quasi_periodic": float(
                (amp_flat > self.peak_ratio_threshold).float().mean().item()
            ),
            "min_lag": self.min_lag,
            "n_rows": int(amp_flat.numel()),
            "n_matrices": int(B * H),
        }
