"""
v2_cz_decay.py — Property V2 : décroissance off-diagonal (proxy CZ).

Spec : DOC/CATALOGUE §V2.

Une matrice d'opérateur Calderón-Zygmund a un noyau K(x, y) qui décroît
comme |x - y|^{-α} avec α > 0. Sur une matrice A discrétisée, ça se
traduit par :

    |A_ij| ≲ C · |i - j|^{-α}

V1 : on regroupe les coefficients de A par lag d = |i - j|, on calcule la
moyenne |A| par lag, et on fait un fit log-log (slope = -α). α grand →
forte décroissance CZ-like.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class V2CZDecay(Property):
    """V2 — exposant de décroissance off-diagonal log-log (proxy CZ regularity)."""

    name = "V2_cz_decay"
    family = "V"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self, min_lag: int = 1, max_lag: int | None = None, eps_floor: float = 1e-30) -> None:
        if min_lag < 1:
            raise ValueError("min_lag ≥ 1")
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype).abs()
        max_lag = self.max_lag or min(N, N2) - 1
        max_lag = min(max_lag, min(N, N2) - 1)

        # Pour chaque lag d, moyenne sur les coefficients |A_ij| où |i - j| = d
        lags = list(range(self.min_lag, max_lag + 1))
        if len(lags) < 3:
            return {"n_matrices": int(B * H), "skip_reason": "Not enough lags"}

        # Construit un tensor (n_lags,) de moyennes par lag
        means_per_lag: list[torch.Tensor] = []
        for d in lags:
            # Diagonale d et -d
            ind_pos = torch.arange(max(0, -d), min(N, N2 - d), device=A_work.device)
            ind_neg = torch.arange(max(0, d), min(N, N2 + d), device=A_work.device)
            diag_pos = A_work[..., torch.arange(N - d, device=A_work.device), torch.arange(d, N, device=A_work.device)] if d < N else None
            diag_neg = A_work[..., torch.arange(d, N, device=A_work.device), torch.arange(N - d, device=A_work.device)] if d < N else None
            if diag_pos is not None and diag_neg is not None:
                stacked = torch.cat([diag_pos, diag_neg], dim=-1)
                means_per_lag.append(stacked.mean(dim=-1))  # (B, H)

        if len(means_per_lag) < 3:
            return {"n_matrices": int(B * H), "skip_reason": "Not enough lags for fit"}

        means = torch.stack(means_per_lag, dim=-1)  # (B, H, n_lags)
        # Fit log-log : log mean ~ -α log lag + β
        means_safe = means.clamp_min(self.eps_floor)
        log_m = means_safe.log()
        log_lag = torch.tensor(lags, device=A_work.device, dtype=A_work.dtype).log()  # (n_lags,)

        # OLS centered
        log_lag_c = log_lag - log_lag.mean()
        log_m_c = log_m - log_m.mean(dim=-1, keepdim=True)
        cov = (log_m_c * log_lag_c).mean(dim=-1)
        var_l = (log_lag_c ** 2).mean()
        alpha = -cov / var_l.clamp_min(self.eps_floor)

        # R²
        beta = log_m.mean(dim=-1) + alpha * log_lag.mean()
        pred = beta.unsqueeze(-1) - alpha.unsqueeze(-1) * log_lag.unsqueeze(0).unsqueeze(0)
        ss_res = ((log_m - pred) ** 2).sum(dim=-1)
        ss_tot = ((log_m - log_m.mean(dim=-1, keepdim=True)) ** 2).sum(dim=-1).clamp_min(self.eps_floor)
        r_squared = 1.0 - ss_res / ss_tot

        alpha_flat = alpha.float().flatten()
        r2_flat = r_squared.float().flatten()

        return {
            "cz_decay_alpha_median": float(alpha_flat.median().item()),
            "cz_decay_alpha_mean": float(alpha_flat.mean().item()),
            "cz_decay_r_squared_median": float(r2_flat.median().item()),
            "fraction_cz_decay_alpha_gt_1": float(
                (alpha_flat > 1.0).float().mean().item()
            ),
            "n_lags_used": len(lags),
            "n_matrices": int(B * H),
        }
