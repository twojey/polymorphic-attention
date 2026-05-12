"""
b7_tropical_degeneracy.py — Property B7 : tropical degeneracy.

Spec : DOC/CATALOGUE §B7 "hiérarchie magnitude log-échelle : ordonnancement des
termes dominants".

L'idée tropicale : sur le semiring (max, +), une matrice est "dégénérée"
si chaque ligne est dominée par une (ou peu d') entrées. Pour l'attention
softmax post-exp, ça équivaut à demander si softmax(s) ≈ one-hot ou
quasi-one-hot.

Métriques :
- `log_gap_top1_top2_*` : log A_top1 − log A_top2 par ligne (∞ si pleinement
  one-hot, 0 si top1=top2)
- `tropical_rank_proxy` : nb d'entrées par ligne avec log A_ij ≥ log A_top1
  − δ (δ=2 par défaut → cellules à un facteur e² du max)
- `argmax_stability` : fraction des lignes où argmax est unique (pas de tie)
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class B7TropicalDegeneracy(Property):
    """B7 — degré de degeneracy tropicale par ligne.

    Pour chaque ligne A[t, :], regarde le profil log A_ij. Si une seule
    entrée domine massivement (log_gap_top1_top2 ≫ 1), la ligne est
    tropicalement dégénérée. Sur des softmax très piqués (ω élevé), on
    attend log_gap_top1_top2 → ∞ et `tropical_rank_proxy` → 1.
    """

    name = "B7_tropical_degeneracy"
    family = "B"
    cost_class = 1
    requires_fp64 = False
    scope = "per_regime"

    def __init__(
        self,
        log_delta_thresholds: tuple[float, ...] = (1.0, 2.0, 4.0),
        eps_floor: float = 1e-30,
    ) -> None:
        self.log_delta_thresholds = log_delta_thresholds
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")

        A_work = A.to(device=ctx.device, dtype=ctx.dtype).abs().clamp_min(self.eps_floor)
        log_A = A_work.log()  # (B, H, N, N)

        # Pour chaque ligne (t-ième ligne de chaque (B, H) matrice) :
        # top1 et top2 selon log A
        topk_vals, _ = log_A.topk(k=2, dim=-1)  # (B, H, N, 2)
        top1 = topk_vals[..., 0]
        top2 = topk_vals[..., 1]
        log_gap = (top1 - top2)  # (B, H, N)

        results: dict[str, float | int | str | bool] = {}
        log_gap_flat = log_gap.float().flatten()
        results["log_gap_top1_top2_median"] = float(log_gap_flat.median().item())
        results["log_gap_top1_top2_mean"] = float(log_gap_flat.mean().item())
        results["log_gap_top1_top2_p10"] = float(log_gap_flat.quantile(0.10).item())
        results["log_gap_top1_top2_p90"] = float(log_gap_flat.quantile(0.90).item())

        # Tropical rank proxy : pour chaque ligne, cb d'entrées à δ du max ?
        top1_kept = topk_vals[..., 0:1]  # (B, H, N, 1)
        for delta in self.log_delta_thresholds:
            near_max = (log_A >= (top1_kept - delta)).float().sum(dim=-1)  # (B, H, N)
            near_max_flat = near_max.float().flatten()
            tag = f"{delta:.1f}".replace(".", "p")
            results[f"tropical_rank_proxy_delta_{tag}_median"] = float(
                near_max_flat.median().item()
            )
            results[f"tropical_rank_proxy_delta_{tag}_mean"] = float(
                near_max_flat.mean().item()
            )

        # Argmax stability : fraction de lignes où argmax est strictement unique
        # (top1 > top2 → log_gap > 0 strictement, mais en pratique on prend
        # un epsilon mou pour éviter le bruit numérique).
        strict_unique = (log_gap > 1e-6).float().mean()
        results["argmax_unique_fraction"] = float(strict_unique.item())

        # Couverture massique : Σ exp(log_top1) / Σ exp(log_A_ij) = top1_softmax_mass
        # — déjà mesuré par B4 top1_concentration mais en log-domain.
        log_max = log_A.max(dim=-1, keepdim=True).values
        log_sum_exp = (log_A - log_max).exp().sum(dim=-1, keepdim=True).log() + log_max
        log_top1_mass = (top1 - log_sum_exp.squeeze(-1))  # log(p_top1)
        top1_mass = log_top1_mass.exp().float().flatten()
        results["top1_softmax_mass_median"] = float(top1_mass.median().item())
        results["top1_softmax_mass_mean"] = float(top1_mass.mean().item())

        results["n_rows"] = int(log_gap.numel())
        return results
