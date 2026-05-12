"""
a4_spectral_entropy.py — Property A4 : entropie spectrale.

Spec : DOC/CATALOGUE §A4.

H_spec(A) = − Σ_i p_i log p_i, où p_i = σ_i² / Σ_j σ_j².

Normalisée par log(rang_spec) ∈ [0, 1] : 0 = un seul σ domine (rang 1),
1 = toutes σ égales (spectre uniforme).

Réutilise cache `svd_singular_values` posé par A1.
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class A4SpectralEntropy(Property):
    """A4 — entropie de la distribution {σ_i² / Σ σ_j²}."""

    name = "A4_spectral_entropy"
    family = "A"
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

        cache_key = ctx.cache_key(
            "svd_singular_values", tuple(A.shape), str(A.dtype)
        )

        def _svd() -> torch.Tensor:
            return torch.linalg.svdvals(A.to(device=ctx.device, dtype=ctx.dtype))

        s = ctx.get_or_compute(cache_key, _svd)
        s2 = s.pow(2)
        total = s2.sum(dim=-1, keepdim=True).clamp_min(self.eps_floor)
        p = s2 / total  # (B, H, K), distribution sur K valeurs singulières
        p_safe = p.clamp_min(self.eps_floor)
        H = -(p_safe * p_safe.log()).sum(dim=-1)  # (B, H), nats

        K = s.shape[-1]
        H_norm = H / math.log(K)

        h_flat = H.float().flatten()
        n_flat = H_norm.float().flatten()

        return {
            "spectral_entropy_median": float(h_flat.median().item()),
            "spectral_entropy_mean": float(h_flat.mean().item()),
            "spectral_entropy_norm_median": float(n_flat.median().item()),
            "spectral_entropy_norm_mean": float(n_flat.mean().item()),
            "fraction_concentrated_norm_below_0p20": float(
                (n_flat < 0.20).float().mean().item()
            ),
            "fraction_diffuse_norm_above_0p80": float(
                (n_flat > 0.80).float().mean().item()
            ),
            "log_K_normalizer": math.log(K),
            "n_matrices": int(h_flat.numel()),
        }
