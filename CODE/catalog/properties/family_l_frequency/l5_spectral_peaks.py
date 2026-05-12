"""
l5_spectral_peaks.py — Property L5 : pics spectraux FFT-1D.

Pour chaque ligne A[t, :], on calcule la FFT-1D et on compte le nombre
de pics de magnitude > τ · max_peak. Permet de détecter une structure
quasi-périodique (peu de pics) vs bruit blanc (beaucoup de pics).

n_peaks faible (1-3) = signal mono/bi-fréquentiel.
n_peaks élevé (≫ N/4) = pas de fréquence dominante.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class L5SpectralPeaks(Property):
    """L5 — nombre de pics FFT row-wise au-dessus de τ · max."""

    name = "L5_spectral_peaks"
    family = "L"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self, peak_threshold: float = 0.30) -> None:
        if not 0.0 < peak_threshold < 1.0:
            raise ValueError(f"peak_threshold doit être ∈ (0, 1)")
        self.peak_threshold = peak_threshold

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, _ = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # FFT row-wise, magnitudes positives uniquement
        F = torch.fft.rfft(A_work, dim=-1).abs()  # (B, H, N, K)
        max_per_row = F.max(dim=-1, keepdim=True).values.clamp_min(1e-30)
        peaks_mask = F > (self.peak_threshold * max_per_row)
        n_peaks_per_row = peaks_mask.sum(dim=-1).float()  # (B, H, N)

        # Position du pic dominant (= argmax FFT)
        argmax_pos = F.argmax(dim=-1).float()  # (B, H, N)
        # Dispersion intra-matrice : std de argmax_pos sur N
        argmax_std = argmax_pos.std(dim=-1)  # (B, H)

        n_peaks_flat = n_peaks_per_row.flatten()
        std_flat = argmax_std.float().flatten()

        return {
            "n_peaks_per_row_median": float(n_peaks_flat.median().item()),
            "n_peaks_per_row_mean": float(n_peaks_flat.mean().item()),
            "argmax_freq_std_median": float(std_flat.median().item()),
            "fraction_mono_freq_n_peaks_le_2": float(
                (n_peaks_flat <= 2.0).float().mean().item()
            ),
            "fraction_broadband_n_peaks_ge_quarter_N": float(
                (n_peaks_flat >= N / 4).float().mean().item()
            ),
            "K_freq_bins": int(F.shape[-1]),
            "n_matrices": int(B * H),
        }
