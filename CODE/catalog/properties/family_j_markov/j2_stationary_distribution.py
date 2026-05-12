"""
j2_stationary_distribution.py — Property J2 : distribution stationnaire π.

Spec : DOC/CATALOGUE §J2 "eigenvector dominant π de A^T".

Calcule π via méthode directe (eigenvalue problem sur A^T) plutôt que
power iteration (cf. K3). Métriques :
- |λ_1| : eigenvalue dominante (= 1 pour A row-stochastique parfait)
- |λ_2| : 2ème eigenvalue → contrôle le mixing rate
- Spectral gap (= 1 - |λ_2|), proxy mixing time

Diffère de K3 (qui faisait power iteration damped pour PageRank) par
la fiabilité numérique sur des matrices presque-singulières.
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class J2StationaryDistribution(Property):
    """J2 — distribution stationnaire via eigvals(A^T) + |λ_2|."""

    name = "J2_stationary_distribution"
    family = "J"
    cost_class = 3
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self, eps_floor: float = 1e-30) -> None:
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N != N2:
            raise ValueError(f"A doit être carrée")

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        row_sum = A_work.sum(dim=-1, keepdim=True).clamp_min(self.eps_floor)
        P = A_work / row_sum

        # Eigenvalues complexes (A non symétrique en général)
        eigs = torch.linalg.eigvals(P)  # (B, H, N) complex
        eigs_abs = eigs.abs()  # (B, H, N) float

        # Trier décroissant
        eigs_sorted, _ = eigs_abs.sort(dim=-1, descending=True)
        lambda1 = eigs_sorted[..., 0]
        lambda2 = eigs_sorted[..., 1] if N >= 2 else torch.zeros_like(lambda1)
        spectral_gap = 1.0 - lambda2  # (B, H), proxy mixing
        # Mixing time estimé : log(eps) / log(|λ_2|) (pour eps modéré, log < 0)
        # On retourne -1/log(|λ_2|) qui est une scale "temps de mélange"
        l2_log = lambda2.clamp_min(self.eps_floor).log()  # log < 0 pour |λ_2| < 1
        mixing_scale = -1.0 / l2_log.clamp_max(-self.eps_floor)
        # Si |λ_2| ≈ 1, mixing_scale → +inf ; clip à N pour stat
        mixing_scale = mixing_scale.clamp_max(float(N))

        l1_flat = lambda1.float().flatten()
        l2_flat = lambda2.float().flatten()
        gap_flat = spectral_gap.float().flatten()
        mix_flat = mixing_scale.float().flatten()

        return {
            "lambda1_abs_median": float(l1_flat.median().item()),
            "lambda1_abs_mean": float(l1_flat.mean().item()),
            "lambda2_abs_median": float(l2_flat.median().item()),
            "lambda2_abs_mean": float(l2_flat.mean().item()),
            "lambda2_abs_p90": float(l2_flat.quantile(0.90).item()),
            "spectral_gap_median": float(gap_flat.median().item()),
            "spectral_gap_mean": float(gap_flat.mean().item()),
            "mixing_scale_median": float(mix_flat.median().item()),
            "n_matrices": int(B * H),
            "seq_len": int(N),
        }
