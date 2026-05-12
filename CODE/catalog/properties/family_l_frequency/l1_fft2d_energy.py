"""
l1_fft2d_energy.py — Property L1 : énergie spectrale 2D.

Spec : DOC/CATALOGUE §L1 "spectre Fourier 2D de A".

Calcule la FFT 2D de A et reporte :
- Énergie basse fréquence (|f| ≤ r_low × N/2) / énergie totale
- Énergie haute fréquence (|f| > r_high × N/2) / énergie totale
- Position du peak spectral (fréquence dominante hors DC)
- Entropie spectrale (concentration de l'énergie)
- DC component magnitude relative

Interpretation :
- Identité → spectre plat (toutes fréquences)
- Toeplitz → spectre invariant le long d'une direction (anti-diag)
- Uniforme → tout en DC (basse fréq = 1.0)
- Quasi-périodique → peak hors DC fort

Cost class 2 : FFT 2D batched, O(B·H·N² log N).
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class L1FFT2DEnergy(Property):
    """L1 — spectre Fourier 2D : décomposition basse/haute fréquence et peak."""

    name = "L1_fft2d_energy"
    family = "L"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(
        self,
        low_freq_radius_ratio: float = 0.25,
        high_freq_radius_ratio: float = 0.75,
    ) -> None:
        self.r_low = low_freq_radius_ratio
        self.r_high = high_freq_radius_ratio

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # FFT 2D centrée (DC au centre)
        spec = torch.fft.fft2(A_work, dim=(-2, -1))
        spec = torch.fft.fftshift(spec, dim=(-2, -1))
        power = spec.abs() ** 2  # (B, H, N, N2)

        # Grille de fréquences distance au centre (DC)
        cy, cx = N // 2, N2 // 2
        ii = torch.arange(N, device=A_work.device, dtype=A_work.dtype)
        jj = torch.arange(N2, device=A_work.device, dtype=A_work.dtype)
        rii = (ii - cy).abs()
        rjj = (jj - cx).abs()
        # Distance ∞-norm (rectangle band) — plus simple et N pas trop grand
        r_max = max(cy, cx)
        radius = torch.maximum(rii.view(-1, 1), rjj.view(1, -1))  # (N, N2)

        low_mask = (radius <= self.r_low * r_max).to(A_work.dtype)
        high_mask = (radius > self.r_high * r_max).to(A_work.dtype)
        # DC = pixel central uniquement
        dc_mask = torch.zeros_like(low_mask)
        dc_mask[cy, cx] = 1.0

        total = power.flatten(start_dim=-2).sum(dim=-1).clamp_min(1e-30)
        low_frac = (power * low_mask).flatten(start_dim=-2).sum(dim=-1) / total
        high_frac = (power * high_mask).flatten(start_dim=-2).sum(dim=-1) / total
        dc_frac = (power * dc_mask).flatten(start_dim=-2).sum(dim=-1) / total

        # Peak hors DC : argmax power où radius > 0
        power_no_dc = power.clone()
        power_no_dc[..., cy, cx] = 0.0
        peak_val = power_no_dc.flatten(start_dim=-2).max(dim=-1).values
        peak_frac = peak_val / total  # (B, H)

        # Entropie spectrale normalisée H(p) / log(N²)
        p_norm = power.flatten(start_dim=-2) / total.unsqueeze(-1)
        # éviter log(0)
        log_p = (p_norm.clamp_min(1e-30)).log()
        entropy = -(p_norm * log_p).sum(dim=-1)  # (B, H), nats
        entropy_norm = entropy / float(torch.tensor(N * N2, dtype=A_work.dtype).log().item())

        low_frac_f = low_frac.float().flatten()
        high_frac_f = high_frac.float().flatten()
        dc_frac_f = dc_frac.float().flatten()
        peak_frac_f = peak_frac.float().flatten()
        ent_f = entropy_norm.float().flatten()

        return {
            "low_freq_energy_fraction_median": float(low_frac_f.median().item()),
            "low_freq_energy_fraction_mean": float(low_frac_f.mean().item()),
            "high_freq_energy_fraction_median": float(high_frac_f.median().item()),
            "high_freq_energy_fraction_mean": float(high_frac_f.mean().item()),
            "dc_energy_fraction_median": float(dc_frac_f.median().item()),
            "peak_offdc_energy_fraction_median": float(peak_frac_f.median().item()),
            "spectral_entropy_norm_median": float(ent_f.median().item()),
            "spectral_entropy_norm_mean": float(ent_f.mean().item()),
            "r_low": self.r_low,
            "r_high": self.r_high,
            "n_matrices": int(B * H),
        }
