"""
p3_hsv_decay.py — Property P3 : décroissance HSV (loi exponentielle vs polynomiale).

Spec : DOC/CATALOGUE §P3 "décroissance des HSV : indicateur de la
régularité du système (LTI à pôles bien séparés → exponentielle ;
système quasi-singulier → polynomiale)".

Pour un système LTI minimal de dimension n, ses HSV σ_1 ≥ … ≥ σ_n
typiquement décroissent en σ_k ≈ C · ρ^k avec ρ ∈ (0, 1) (régulier).
Si décroissance polynomiale σ_k ≈ C · k^{-α} (frontière, pôles fusionnés).

V1 : régression log(σ_k) ~ slope · k pour distinguer exp vs poly.
- exp : slope < 0, mais log-linéaire (R² élevé en log vs k)
- poly : log-linéaire en log(k) (régression log-log dominante)
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class P3HSVDecay(Property):
    """P3 — détecte décroissance exponentielle vs polynomiale des HSV."""

    name = "P3_hsv_decay"
    family = "P"
    cost_class = 3
    requires_fp64 = True
    scope = "per_regime"

    def __init__(self, k_top: int = 12, eps_floor: float = 1e-30) -> None:
        self.k_top = k_top
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N < 4:
            return {"n_matrices": int(B * H), "skip_reason": "N too small"}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        k = N // 2
        n_col = N - k
        H_block = torch.zeros(B, H, k, n_col * N2, device=A_work.device, dtype=A_work.dtype)
        for i in range(k):
            for j in range(n_col):
                if i + j < N:
                    H_block[..., i, j * N2: (j + 1) * N2] = A_work[..., i + j, :]

        sigmas = torch.linalg.svdvals(H_block)  # (B, H, K)
        K = min(self.k_top, sigmas.shape[-1])
        if K < 3:
            return {"n_matrices": int(B * H), "skip_reason": "too few HSV"}
        s = sigmas[..., :K]  # (B, H, K)
        s_max = s[..., :1].clamp_min(self.eps_floor)
        s_norm = s / s_max  # σ_1 = 1
        log_s = s_norm.clamp_min(self.eps_floor).log()  # (B, H, K)

        ks = torch.arange(1, K + 1, dtype=log_s.dtype, device=log_s.device)  # 1..K
        log_ks = ks.log()  # log(1..K)

        # Régression linéaire (least-squares) log(σ_k) ~ slope_exp · k + b
        k_mean = ks.mean()
        log_s_mean = log_s.mean(dim=-1, keepdim=True)
        num_e = ((ks - k_mean) * (log_s - log_s_mean)).sum(dim=-1)
        den_e = ((ks - k_mean) ** 2).sum().clamp_min(self.eps_floor)
        slope_exp = num_e / den_e  # (B, H), négatif

        # R² exponentiel
        pred_e = slope_exp.unsqueeze(-1) * ks + (log_s_mean.squeeze(-1) - slope_exp * k_mean).unsqueeze(-1)
        ss_res_e = ((log_s - pred_e) ** 2).sum(dim=-1)
        ss_tot = ((log_s - log_s_mean) ** 2).sum(dim=-1).clamp_min(self.eps_floor)
        r2_exp = 1.0 - ss_res_e / ss_tot

        # Régression log-log : log(σ_k) ~ slope_poly · log(k)
        log_ks_mean = log_ks.mean()
        num_p = ((log_ks - log_ks_mean) * (log_s - log_s_mean)).sum(dim=-1)
        den_p = ((log_ks - log_ks_mean) ** 2).sum().clamp_min(self.eps_floor)
        slope_poly = num_p / den_p
        pred_p = slope_poly.unsqueeze(-1) * log_ks + (log_s_mean.squeeze(-1) - slope_poly * log_ks_mean).unsqueeze(-1)
        ss_res_p = ((log_s - pred_p) ** 2).sum(dim=-1)
        r2_poly = 1.0 - ss_res_p / ss_tot

        # Type majoritaire : exp si r2_exp > r2_poly
        is_exp = (r2_exp > r2_poly).float()

        return {
            "hsv_decay_slope_exp_median": float(slope_exp.float().median().item()),
            "hsv_decay_slope_poly_median": float(slope_poly.float().median().item()),
            "hsv_decay_r2_exp_median": float(r2_exp.float().median().item()),
            "hsv_decay_r2_poly_median": float(r2_poly.float().median().item()),
            "fraction_exp_dominant": float(is_exp.mean().item()),
            "K_used": int(K),
            "n_matrices": int(B * H),
        }
