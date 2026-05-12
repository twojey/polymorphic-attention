"""
v1_pseudodifferential.py — Property V1 : opérateurs pseudo-différentiels.

Spec : DOC/CATALOGUE §V1 "opérateur pseudo-différentiel : si A est
représentable comme Op(σ)f(x) = ∫ σ(x, ξ) e^{ix·ξ} f̂(ξ) dξ où σ est
le symbole, on classifie par la régularité de σ".

V1 numérique : on regarde la matrice A vue dans la base (espace × fréquence)
en faisant une FFT par lignes ET par colonnes :
    σ̃(ω_row, ω_col) = FFT_row FFT_col (A)
Si A est pseudo-différentiel d'ordre m, alors |σ̃| décroît en |ξ|^m
en hautes fréquences. On mesure :
- exposant de décroissance (régression log |σ̃| ~ -m · log|ω|)
- "lissité" du symbole = ratio énergie basses fréquences / hautes
- bornitude : sup_ω |σ̃(ω)|

Symbole lisse → opérateur smooth (intuition : décroissance forte = m grand).
Symbole rugueux → opérateur frontière (m faible / négatif).
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class V1Pseudodifferential(Property):
    """V1 — proxy ordre m pseudo-différentiel via décroissance FFT2D du symbole."""

    name = "V1_pseudodifferential"
    family = "V"
    cost_class = 3
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
        if N < 4:
            return {"n_matrices": int(B * H), "skip_reason": "N too small"}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # FFT 2D : symbole σ̃(ω_row, ω_col)
        sigma = torch.fft.fft2(A_work)  # (B, H, N, N) complex
        magn = sigma.abs()  # (B, H, N, N)

        # Construire grille de fréquences radiale
        freqs = torch.fft.fftfreq(N, d=1.0, device=A_work.device).abs()
        f_row = freqs.unsqueeze(-1).expand(N, N)
        f_col = freqs.unsqueeze(0).expand(N, N)
        f_radial = (f_row ** 2 + f_col ** 2).sqrt()
        f_radial = f_radial.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)

        # Régression log|σ̃| ~ -m · log(|ω| + eps)
        mask = f_radial > 1e-6  # exclure DC
        log_f = (f_radial.clamp_min(self.eps_floor)).log()
        log_mag = magn.clamp_min(self.eps_floor).log()

        # Pour chaque (B, H), régression linéaire (least-squares)
        # Aplatir spatial
        flat_log_f = log_f.flatten(start_dim=-2)  # (1, 1, N²)
        flat_log_m = log_mag.flatten(start_dim=-2)  # (B, H, N²)
        flat_mask = mask.flatten(start_dim=-2)  # (1, 1, N²)

        # On suppose même grille pour tous batch elem
        log_f_valid = flat_log_f[flat_mask].squeeze()  # 1D
        log_m_valid = flat_log_m[..., flat_mask.squeeze()]  # (B, H, K)
        if log_m_valid.numel() == 0 or log_f_valid.numel() < 3:
            return {"n_matrices": int(B * H), "skip_reason": "no valid freqs"}

        # Régression : slope = -m
        log_f_mean = log_f_valid.mean()
        log_m_mean = log_m_valid.mean(dim=-1, keepdim=True)
        num = ((log_f_valid - log_f_mean) * (log_m_valid - log_m_mean)).sum(dim=-1)
        den = ((log_f_valid - log_f_mean) ** 2).sum().clamp_min(self.eps_floor)
        slope = num / den  # (B, H), typically < 0
        m_order = -slope  # ordre pseudo-diff positif si décroissance

        # Lissité : ratio énergie basses fréquences (|ω| < 0.25) / hautes (|ω| ≥ 0.25)
        low_mask = (f_radial < 0.25) & (f_radial > 1e-6)
        high_mask = f_radial >= 0.25
        energy_low = (magn ** 2 * low_mask.float()).sum(dim=(-2, -1))
        energy_high = (magn ** 2 * high_mask.float()).sum(dim=(-2, -1)).clamp_min(self.eps_floor)
        smoothness_ratio = energy_low / energy_high

        return {
            "pseudodiff_order_m_median": float(m_order.float().median().item()),
            "pseudodiff_order_m_mean": float(m_order.float().mean().item()),
            "pseudodiff_smoothness_ratio_median": float(smoothness_ratio.float().median().item()),
            "pseudodiff_smoothness_ratio_log10_median": float(
                smoothness_ratio.clamp_min(self.eps_floor).log10().float().median().item()
            ),
            "pseudodiff_symbol_max_magn_median": float(magn.amax(dim=(-2, -1)).float().median().item()),
            "fraction_order_positive": float((m_order > 0).float().mean().item()),
            "n_matrices": int(B * H),
            "seq_len": int(N),
        }
