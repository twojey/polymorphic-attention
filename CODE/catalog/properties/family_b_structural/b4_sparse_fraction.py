"""
b4_sparse_fraction.py — Property B4 : sparsité effective.

Spec : DOC/00b §B4 "ratio coefficients > seuil".

Mesure plusieurs proxies de sparsité par matrice :
- `sparse_fraction_eps_*` : ratio |A_ij| < ε × max|A| (entrées négligeables)
- `effective_density` : 1 − sparse_fraction (entrées significatives)
- `top_k_concentration` : Σ top-k / Σ tous (Pareto concentration)
- `gini_coefficient` : inégalité distribution coefficients

Pour de l'attention softmax sur séquence longue, on attend `sparse_fraction`
élevé (la plupart des coefficients ≈ 0). Sur attention dense uniforme,
fraction faible.

Cost class 1 : pas de SVD ni projection — juste des comparaisons et sums.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class B4SparseFraction(Property):
    """B4 — sparsité effective : combien de coefficients sont négligeables ?

    Calcule plusieurs seuils relatifs (ε ∈ {0.01, 0.05, 0.10} de max|A|)
    + top-k concentration (k=1, 5, 10 fraction de mass) + Gini.

    Sur softmax dense, on attend Gini élevé (≥ 0.7) avec top-1 fraction
    modérée. Sur attention orthogonale / near-uniform, Gini bas.
    """

    name = "B4_sparse_fraction"
    family = "B"
    cost_class = 1
    requires_fp64 = False
    scope = "per_regime"

    def __init__(
        self,
        epsilons: tuple[float, ...] = (0.01, 0.05, 0.10),
        top_ks: tuple[int, ...] = (1, 5, 10),
    ) -> None:
        self.epsilons = epsilons
        self.top_ks = top_ks

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")

        A_work = A.to(device=ctx.device, dtype=ctx.dtype).abs()
        # (B, H, N, N) → flatten (B, H, N*N)
        flat = A_work.flatten(start_dim=-2)
        N2 = flat.shape[-1]

        results: dict[str, float | int | str | bool] = {}

        # 1. sparse_fraction par seuil relatif au max par matrice
        max_per_mat = flat.max(dim=-1, keepdim=True).values.clamp_min(1e-30)
        for eps in self.epsilons:
            threshold = eps * max_per_mat
            below = (flat < threshold).float().mean(dim=-1)  # (B, H)
            below_flat = below.flatten()
            tag = f"{eps:.2f}".replace(".", "p")
            results[f"sparse_frac_eps_{tag}_median"] = float(below_flat.median().item())
            results[f"sparse_frac_eps_{tag}_mean"] = float(below_flat.mean().item())

        # 2. Top-k concentration : Σ top-k / Σ tous
        sorted_desc, _ = flat.sort(dim=-1, descending=True)
        total_sum = sorted_desc.sum(dim=-1).clamp_min(1e-30)  # (B, H)
        for k in self.top_ks:
            if k > N2:
                continue
            topk_sum = sorted_desc[..., :k].sum(dim=-1)
            conc = (topk_sum / total_sum).flatten()
            results[f"top{k}_concentration_median"] = float(conc.median().item())
            results[f"top{k}_concentration_mean"] = float(conc.mean().item())

        # 3. Coefficient de Gini sur la distribution des |A_ij|
        # G = (Σ |x_i − x_j|) / (2 n² mean) — formule directe via sorted
        # Utilise la formule G = (2 Σ i·x_(i) − (n+1) Σ x_(i)) / (n Σ x_(i))
        # avec x triés croissants.
        sorted_asc = sorted_desc.flip(dims=(-1,))
        n = sorted_asc.shape[-1]
        idx = torch.arange(1, n + 1, device=sorted_asc.device, dtype=sorted_asc.dtype)
        weighted = (sorted_asc * idx).sum(dim=-1)
        unw = sorted_asc.sum(dim=-1).clamp_min(1e-30)
        gini = (2 * weighted - (n + 1) * unw) / (n * unw)
        gini_flat = gini.float().flatten()
        results["gini_median"] = float(gini_flat.median().item())
        results["gini_mean"] = float(gini_flat.mean().item())
        results["gini_min"] = float(gini_flat.min().item())
        results["gini_max"] = float(gini_flat.max().item())

        results["n_matrices"] = int(flat.shape[0] * flat.shape[1])
        return results
