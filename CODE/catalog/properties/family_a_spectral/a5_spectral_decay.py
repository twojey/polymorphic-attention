"""
a5_spectral_decay.py — Property A5 : décroissance spectrale log-log.

Spec : DOC/CATALOGUE §A5.

On suppose un fit log(σ_i) ~ −α · log(i + 1) + β sur les K top-σ.
- α grand → décroissance rapide (rang effectif faible)
- α petit → décroissance lente (spectre plat)

Calcul : OLS sur log(σ_i) vs log(i+1) pour i ∈ [0, K_fit).
Métriques : α (pente), résidu R² du fit.

Réutilise cache `svd_singular_values` posé par A1.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class A5SpectralDecay(Property):
    """A5 — exposant de décroissance log-log (slope OLS sur log σ_i)."""

    name = "A5_spectral_decay"
    family = "A"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(
        self, k_fit_max: int = 32, eps_floor: float = 1e-30
    ) -> None:
        if k_fit_max < 3:
            raise ValueError(f"k_fit_max doit être ≥ 3, reçu {k_fit_max}")
        self.k_fit_max = k_fit_max
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")

        cache_key = ctx.cache_key(
            "svd_singular_values", tuple(A.shape), str(A.dtype)
        )

        def _svd() -> torch.Tensor:
            return torch.linalg.svdvals(A.to(device=ctx.device, dtype=ctx.dtype))

        s = ctx.get_or_compute(cache_key, _svd)  # (B, H, K)
        K = s.shape[-1]
        k_fit = min(K, self.k_fit_max)

        s_fit = s[..., :k_fit].clamp_min(self.eps_floor)
        log_s = s_fit.log()  # (B, H, k_fit)
        idx = torch.arange(1, k_fit + 1, device=s.device, dtype=s.dtype)
        log_i = idx.log()  # (k_fit,)

        # OLS batch : minimize Σ (log_s - (β - α log_i))²
        # → α = -cov(log_s, log_i) / var(log_i)
        log_i_centered = log_i - log_i.mean()  # (k_fit,)
        log_s_centered = log_s - log_s.mean(dim=-1, keepdim=True)
        cov = (log_s_centered * log_i_centered).mean(dim=-1)  # (B, H)
        var_li = (log_i_centered ** 2).mean()
        alpha = -cov / var_li.clamp_min(self.eps_floor)  # pente négative → α > 0

        # R² du fit
        beta = log_s.mean(dim=-1) + alpha * log_i.mean()  # intercept
        pred = beta.unsqueeze(-1) - alpha.unsqueeze(-1) * log_i.unsqueeze(0).unsqueeze(0)
        ss_res = ((log_s - pred) ** 2).sum(dim=-1)
        ss_tot = ((log_s - log_s.mean(dim=-1, keepdim=True)) ** 2).sum(dim=-1).clamp_min(self.eps_floor)
        r_squared = 1.0 - ss_res / ss_tot

        alpha_flat = alpha.float().flatten()
        r2_flat = r_squared.float().flatten()

        return {
            "decay_alpha_median": float(alpha_flat.median().item()),
            "decay_alpha_mean": float(alpha_flat.mean().item()),
            "decay_alpha_p10": float(alpha_flat.quantile(0.10).item()),
            "decay_alpha_p90": float(alpha_flat.quantile(0.90).item()),
            "decay_r_squared_median": float(r2_flat.median().item()),
            "decay_r_squared_mean": float(r2_flat.mean().item()),
            "k_fit": k_fit,
            "n_matrices": int(alpha_flat.numel()),
        }
