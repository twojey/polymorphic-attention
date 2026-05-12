"""
a1_r_eff.py — Property A1 : rang effectif r_eff(θ).

Spec : DOC/CATALOGUE A1 + DOC/02 §3 + DOC/glossaire §Rang effectif.

r_eff(θ) = min { k : (Σ_{i<k} σ_i²) / Σ_j σ_j² ≥ θ }

Valeurs canoniques : θ = 0.95 (par défaut), θ = 0.99 (strict).

Mesure centrale de la SCH (Sparse-Component Hypothesis). Phase 2 sur
SMNIST (2026-05-12) a montré r_eff médian = 2 sur N×N avec N jusqu'à
1287, 78.9 % des matrices à r_eff ≤ 3.

Output : médiane, mean, std, p10, p90, min, max sur la dim (B, H) du
batch d'un régime. La Battery agrège ensuite cross-régime en RegimeStats.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class A1ReffTheta99(Property):
    """A1 — rang effectif r_eff(θ=0.99) par matrice du batch.

    Calcule la SVD batchée FP64 (ou FP32 si requires_fp64 désactivé par
    PropertyContext) et le compteur r_eff via cumsum normalisé.

    Cache : `svd_singular_values` partagé dans le PropertyContext pour
    réutilisation par A2 (entropie spectrale), A3 (conditionnement), etc.
    """

    name = "A1_r_eff_theta099"
    family = "A"
    cost_class = 2  # SVD batchée ~1-10s par régime selon N
    requires_fp64 = False  # FP32 GPU suffit pour le compteur r_eff
    requires_symmetric = False
    scope = "per_regime"

    def __init__(self, theta: float = 0.99) -> None:
        self.theta = theta
        if not (0.0 < theta < 1.0):
            raise ValueError(f"theta doit être dans (0, 1), reçu {theta}")

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        """A : (B, H, N, N). Retourne stats r_eff sur la dim (B, H) flatten."""
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu shape {A.shape}")

        # Cache key inclut shape + dtype pour invalidation correcte
        cache_key = ctx.cache_key(
            "svd_singular_values",
            tuple(A.shape),
            str(A.dtype),
        )

        def _compute_svdvals() -> torch.Tensor:
            A_work = A.to(device=ctx.device, dtype=ctx.dtype)
            return torch.linalg.svdvals(A_work)  # (B, H, min(M, N))

        s = ctx.get_or_compute(cache_key, _compute_svdvals)
        r_eff = _r_eff_from_singular_values(s, self.theta)  # (B, H) int

        r_flat = r_eff.float().flatten()
        return {
            "r_eff_median": float(r_flat.median().item()),
            "r_eff_mean": float(r_flat.mean().item()),
            "r_eff_std": float(r_flat.std().item()),
            "r_eff_min": int(r_flat.min().item()),
            "r_eff_max": int(r_flat.max().item()),
            "r_eff_p10": float(r_flat.quantile(0.10).item()),
            "r_eff_p90": float(r_flat.quantile(0.90).item()),
            "n_matrices": int(r_flat.numel()),
            "theta": float(self.theta),
        }


def _r_eff_from_singular_values(s: torch.Tensor, theta: float) -> torch.Tensor:
    """r_eff(θ) à partir de valeurs singulières.

    Fonction pure (sans state, sans I/O) — réutilisée par tests + Properties.
    """
    s2 = s.pow(2)
    cumsum = s2.cumsum(dim=-1)
    total = s2.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    ratio = cumsum / total
    above = ratio >= theta
    r_eff = above.float().argmax(dim=-1) + 1
    all_zero = s2.sum(dim=-1) == 0
    return torch.where(all_zero, torch.zeros_like(r_eff), r_eff)
