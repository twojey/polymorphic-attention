"""
q5_hmatrix_distribution.py — Property Q5 : distribution rangs H-matrix.

Spec : DOC/CATALOGUE §Q5 "histogramme des rangs des blocs admissibles
d'une partition récursive ; signature compression Hackbusch".

Pour une partition récursive binaire de [0, N) :
- Niveau 0 : 1 bloc N×N (le tout)
- Niveau ℓ : 2^ℓ × 2^ℓ blocs de taille (N/2^ℓ)
- Un bloc (i, j) est **admissible** au niveau ℓ si dist(i, j) ≥ admissibility_eta
  (typ. η = 2)

Pour chaque bloc admissible à chaque niveau, calcule rang(bloc).
Sortie : distribution rangs (mean, max, p90, fraction rang ≤ k).
Signature compression : si max_rang reste borné quand N croît, alors
A est H-matrix compressible.
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


def _r_eff(A: torch.Tensor, theta: float, eps_floor: float) -> torch.Tensor:
    s = torch.linalg.svdvals(A)
    s2 = s.pow(2)
    cumsum = s2.cumsum(dim=-1)
    total = cumsum[..., -1:].clamp_min(eps_floor)
    ratio = cumsum / total
    return (ratio >= theta).float().argmax(dim=-1) + 1


@register_property
class Q5HMatrixDistribution(Property):
    """Q5 — distribution rangs des blocs admissibles (H-matrix Hackbusch)."""

    name = "Q5_hmatrix_distribution"
    family = "Q"
    cost_class = 4
    requires_fp64 = True
    scope = "per_regime"

    def __init__(
        self,
        max_depth: int = 3,
        admissibility_eta: int = 2,
        theta_cumulative: float = 0.99,
        eps_floor: float = 1e-30,
    ) -> None:
        self.max_depth = max_depth
        self.eta = admissibility_eta
        self.theta = theta_cumulative
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, _ = A.shape
        log2N = int(math.log2(N)) if N > 0 else 0
        depth = min(self.max_depth, log2N)
        if depth < 1:
            return {"n_matrices": int(B * H), "skip_reason": "N too small"}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        all_ranks: list[float] = []
        ranks_per_level: dict[int, list[float]] = {}
        for level in range(1, depth + 1):
            n_splits = 2 ** level
            if N % n_splits != 0:
                continue
            block_size = N // n_splits
            level_ranks: list[float] = []
            for i in range(n_splits):
                for j in range(n_splits):
                    if abs(i - j) < self.eta:
                        continue
                    sub = A_work[..., i * block_size: (i + 1) * block_size,
                                       j * block_size: (j + 1) * block_size]
                    r = _r_eff(sub, self.theta, self.eps_floor).float().flatten()
                    level_ranks.extend(r.tolist())
            ranks_per_level[level] = level_ranks
            all_ranks.extend(level_ranks)

        if not all_ranks:
            return {"n_matrices": int(B * H), "skip_reason": "no admissible blocks"}

        ranks_t = torch.tensor(all_ranks, dtype=torch.float64)
        results: dict[str, float | int | str | bool] = {
            "hmatrix_rank_max": float(ranks_t.max().item()),
            "hmatrix_rank_p90": float(ranks_t.quantile(0.90).item()),
            "hmatrix_rank_median": float(ranks_t.median().item()),
            "hmatrix_rank_mean": float(ranks_t.mean().item()),
            "fraction_rank_le_2": float((ranks_t <= 2).float().mean().item()),
            "fraction_rank_le_4": float((ranks_t <= 4).float().mean().item()),
            "fraction_rank_le_8": float((ranks_t <= 8).float().mean().item()),
            "n_admissible_blocks_total": int(ranks_t.numel()),
            "depth_used": int(depth),
            "n_matrices": int(B * H),
        }
        # Évolution par niveau
        for level, lst in ranks_per_level.items():
            if lst:
                t = torch.tensor(lst, dtype=torch.float64)
                results[f"avg_rank_level{level}"] = float(t.mean().item())
                results[f"max_rank_level{level}"] = float(t.max().item())
        return results
