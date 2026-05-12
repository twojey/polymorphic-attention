"""
r3_bochner_stationarity.py — Property R3 : stationnarité Bochner.

Spec : DOC/CATALOGUE §R3 "test Bochner : un noyau K est stationnaire
K(x, y) = κ(x − y) ssi son spectre Fourier est positif (mesure positive)".

Pour A vue comme noyau discret K(i, j) :
- K stationnaire ⟺ K(i, j) = κ(i − j) ⟺ A est une matrice Toeplitz
- Bochner exige aussi que FFT(κ) ≥ 0 (mesure spectrale positive)

V1 :
1. Extraire κ(d) = moyenne A[i, j] sur i − j = d (= meilleure approx Toeplitz)
2. Calculer FFT(κ) → spectre P(ω)
3. Fraction de P(ω) < 0 = violation Bochner
4. Erreur reconstruction Toeplitz : ‖A − Toeplitz(κ)‖_F / ‖A‖_F

Stationnaire Bochner ⟺ (1) faible erreur recon Toeplitz ET (2) FFT positive partout.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class R3BochnerStationarity(Property):
    """R3 — test Bochner : stationnarité + spectre positif."""

    name = "R3_bochner_stationarity"
    family = "R"
    cost_class = 2
    requires_fp64 = True
    scope = "per_regime"

    def __init__(self, eps_floor: float = 1e-30) -> None:
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, _ = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # κ(d) = mean over (i, j) such that i - j = d
        # On a 2N-1 valeurs possibles (d = -(N-1) à N-1)
        # On utilise un padding circulaire pour simplifier : κ(d) pour d ∈ [0, N)
        # avec convention κ(-d) = κ(d) si on symétrise
        idx_i = torch.arange(N, device=A_work.device).unsqueeze(1).expand(N, N)
        idx_j = torch.arange(N, device=A_work.device).unsqueeze(0).expand(N, N)
        diff = idx_i - idx_j  # (N, N), allant de -(N-1) à N-1
        # On wrap : d_circulaire = diff mod N
        d_circ = diff % N  # (N, N)
        # On somme A[i, j] groupé par d_circ
        kappa = torch.zeros(B, H, N, dtype=A_work.dtype, device=A_work.device)
        counts = torch.zeros(N, dtype=A_work.dtype, device=A_work.device)
        for d in range(N):
            mask = d_circ == d  # (N, N)
            counts[d] = mask.sum().to(A_work.dtype)
            kappa[..., d] = A_work[..., mask].sum(dim=-1) / counts[d].clamp_min(self.eps_floor)

        # Toeplitz reconstruite à partir de kappa
        toep = kappa[..., d_circ]  # (B, H, N, N)
        err = (A_work - toep).flatten(start_dim=-2).norm(dim=-1)
        A_norm = A_work.flatten(start_dim=-2).norm(dim=-1).clamp_min(self.eps_floor)
        rel_err = (err / A_norm).float()

        # FFT du noyau circulaire → spectre
        spectrum = torch.fft.fft(kappa, dim=-1)  # (B, H, N) complex
        spectrum_real = spectrum.real
        spectrum_imag = spectrum.imag
        # Mesure positivity violation : fraction de bins avec partie réelle < 0
        violation_mask = spectrum_real < -1e-10
        violation_frac = violation_mask.float().mean(dim=-1)  # (B, H)
        # Magnitude négative max
        neg_max = (-spectrum_real).clamp_min(0.0).max(dim=-1).values  # (B, H)

        # Test Bochner combiné : stationnaire ET spectre positif
        bochner_pass = (rel_err.flatten() < 0.10) & (violation_frac.flatten() < 0.05)

        return {
            "bochner_recon_err_median": float(rel_err.flatten().median().item()),
            "bochner_negative_fraction_median": float(violation_frac.flatten().median().item()),
            "bochner_negative_max_median": float(neg_max.flatten().float().median().item()),
            "fraction_pass_bochner": float(bochner_pass.float().mean().item()),
            "spectrum_imag_mag_median": float(spectrum_imag.abs().float().median().item()),
            "n_matrices": int(B * H),
            "seq_len": int(N),
        }
