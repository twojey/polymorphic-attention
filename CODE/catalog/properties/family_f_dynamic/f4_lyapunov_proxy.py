"""
f4_lyapunov_proxy.py — Property F4 : proxy d'exposant de Lyapunov.

Pour A interprétée comme opérateur de transition, λ_max ≈ log ρ(A) où ρ(A)
est le rayon spectral.

λ < 0 : contraction (système stable)
λ > 0 : expansion (système chaotique)
λ ≈ 0 : marginal

Pour Oracle stochastique par ligne : ρ(A) = 1 toujours (Perron-Frobenius).
Diagnostic : on calcule λ via top eigenvalue de A·Aᵀ (Gram), donne le taux
de croissance moyen sous itération.
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class F4LyapunovProxy(Property):
    """F4 — proxy Lyapunov : log σ_max(A) (Gram top eigenvalue)."""

    name = "F4_lyapunov_proxy"
    family = "F"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self, eps_floor: float = 1e-30) -> None:
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, _ = A.shape

        s = ctx.svdvals_cached(A)  # (B, H, K)
        sigma_max = s[..., 0].clamp_min(self.eps_floor)
        lam = sigma_max.log()  # (B, H), nats
        # FTLE proxy : 1/N log σ_max (average over horizon N)
        ftle = lam / max(N, 1)

        lam_flat = lam.float().flatten()
        ftle_flat = ftle.float().flatten()

        return {
            "lyapunov_log_smax_median": float(lam_flat.median().item()),
            "lyapunov_log_smax_mean": float(lam_flat.mean().item()),
            "ftle_proxy_median": float(ftle_flat.median().item()),
            "fraction_contractive_lam_neg": float(
                (lam_flat < 0.0).float().mean().item()
            ),
            "fraction_expansive_lam_pos": float(
                (lam_flat > 0.0).float().mean().item()
            ),
            "n_matrices": int(B * H),
        }
