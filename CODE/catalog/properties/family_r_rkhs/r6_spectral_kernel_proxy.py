"""
r6_spectral_kernel_proxy.py — Property R6 : proxy de spectre kernel via Nyström.

Approximation Nyström : on échantillonne m positions aléatoires, on calcule
le sous-noyau A_sub (m × m), et on compare son spectre à celui de A complet.

Si A est un kernel Mercer compact, le spectre Nyström approxime bien le
spectre vrai. Métrique : corrélation Pearson entre top-k σ_Nyström et
top-k σ_full.

Pour m = N (échantillon complet), corrélation = 1 trivialement. On utilise
m = min(N, 32) → diagnostic d'approximabilité.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class R6SpectralKernelProxy(Property):
    """R6 — proxy Nyström : spectre sous-échantillonné vs spectre complet."""

    name = "R6_spectral_kernel_proxy"
    family = "R"
    cost_class = 3
    requires_fp64 = False
    scope = "per_regime"

    def __init__(
        self,
        m_sample: int = 32,
        top_k_corr: int = 8,
        seed_offset: int = 0,
        eps_floor: float = 1e-30,
    ) -> None:
        self.m_sample = m_sample
        self.top_k_corr = top_k_corr
        self.seed_offset = seed_offset
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N != N2 or N < 4:
            return {"skip_reason": "non-square or too small", "n_matrices": int(B * H)}

        m = min(self.m_sample, N)
        if m == N:
            return {
                "skip_reason": "sample size = N, trivial",
                "n_matrices": int(B * H),
            }

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        s_full = ctx.svdvals_cached(A_work)  # (B, H, K)

        # Sample m indices (déterministe seed)
        seed = int(ctx.metadata.get("seed", 0)) + self.seed_offset
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(N, generator=g)[:m].to(device=A_work.device)
        A_sub = A_work.index_select(-2, idx).index_select(-1, idx)  # (B, H, m, m)
        s_sub = torch.linalg.svdvals(A_sub)  # (B, H, m)

        # Scale Nyström : σ_Nys ≈ (N/m) · σ_sub
        s_sub_scaled = s_sub * (N / m)

        # Compare top_k σ (in log space pour robustesse)
        k = min(self.top_k_corr, s_full.shape[-1], s_sub_scaled.shape[-1])
        log_full = torch.log(s_full[..., :k].clamp_min(self.eps_floor))
        log_sub = torch.log(s_sub_scaled[..., :k].clamp_min(self.eps_floor))
        # Pearson per (b, h)
        lf = log_full - log_full.mean(dim=-1, keepdim=True)
        ls = log_sub - log_sub.mean(dim=-1, keepdim=True)
        num = (lf * ls).sum(dim=-1)
        den = (lf.pow(2).sum(dim=-1) * ls.pow(2).sum(dim=-1)).sqrt().clamp_min(self.eps_floor)
        corr = (num / den).float().flatten()

        return {
            "nystrom_spectrum_corr_median": float(corr.median().item()),
            "nystrom_spectrum_corr_mean": float(corr.mean().item()),
            "fraction_high_corr_above_0p90": float(
                (corr > 0.90).float().mean().item()
            ),
            "m_sample": int(m),
            "N": int(N),
            "top_k_corr": int(k),
            "n_matrices": int(B * H),
        }
