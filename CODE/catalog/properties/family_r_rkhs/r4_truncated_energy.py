"""
r4_truncated_energy.py — Property R4 : énergie tronquée du noyau (decay spectral).

Spec : DOC/CATALOGUE §R4.

Pour un opérateur compact, le spectre {σ_n} décroît vers 0 quand n → ∞.
La fraction d'énergie capturée par les top-k σ est une mesure de compacité :

    E_truncated(k) = (Σ_{i<k} σ_i²) / (Σ_i σ_i²)

V1 : on rapporte E_truncated pour plusieurs k (5%, 10%, 25% du rang max).
Réutilise cache svd_singular_values.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class R4TruncatedEnergy(Property):
    """R4 — fraction d'énergie capturée par top-k σ pour multiple k."""

    name = "R4_truncated_energy"
    family = "R"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(
        self,
        k_fractions: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50),
        eps_floor: float = 1e-30,
    ) -> None:
        for f in k_fractions:
            if not 0.0 < f <= 1.0:
                raise ValueError("k_fraction ∈ (0, 1]")
        self.k_fractions = k_fractions
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape

        cache_key = ctx.cache_key(
            "svd_singular_values", tuple(A.shape), str(A.dtype)
        )

        def _svd() -> torch.Tensor:
            return torch.linalg.svdvals(A.to(device=ctx.device, dtype=ctx.dtype))

        s = ctx.get_or_compute(cache_key, _svd)  # (B, H, K)
        K = s.shape[-1]
        s2 = s.pow(2)
        total = s2.sum(dim=-1).clamp_min(self.eps_floor)

        results: dict[str, float | int | str | bool] = {}
        for f in self.k_fractions:
            k = max(1, int(f * K))
            kept = s2[..., :k].sum(dim=-1)
            frac = (kept / total).float().flatten()
            tag = f"{f:.2f}".replace(".", "p")
            results[f"truncated_energy_top_{tag}_median"] = float(frac.median().item())
            results[f"truncated_energy_top_{tag}_mean"] = float(frac.mean().item())

        results["K"] = K
        results["n_matrices"] = int(B * H)
        return results
