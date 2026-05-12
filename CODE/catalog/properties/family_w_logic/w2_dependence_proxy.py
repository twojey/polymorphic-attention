"""
w2_dependence_proxy.py — Property W2 : proxy NTP_2 / dependence measure.

Spec : DOC/CATALOGUE §W2-W3.

NTP_2 (not the tree property) caractérise les théories model-theoretic
sans "tree structure" complexe. Pour une matrice, un proxy est de mesurer
la **dépendance non-tree-like** entre lignes :

- Indice de "branching" : pour chaque ligne, calculer la matrice de
  similarité avec les autres lignes, et compter combien forment un arbre
  de dépendance (single-link strict) vs un graphe plus complexe.

V1 simplifié : pour chaque ligne, on regarde ses 2 voisins les plus
proches (Euclidean) et on calcule le triangle "ligne, voisin1, voisin2".
Si triangle inégalité stricte (d(v1, v2) > d(L, v1) + d(L, v2)), c'est
"tree-like". Sinon, dépendance non-tree.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class W2DependenceProxy(Property):
    """W2 — proxy NTP_2 via violation triangle inégalité sur graphes de lignes."""

    name = "W2_dependence_proxy"
    family = "W"
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
        B, H, N, N2 = A.shape
        if N < 3:
            return {"n_matrices": int(B * H), "skip_reason": "N < 3"}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        rows = A_work  # (B, H, N, N) où dim=-2 indexe les lignes
        D = torch.cdist(rows, rows)  # (B, H, N, N), O(N²) mémoire

        # Pour chaque ligne, prendre top-2 voisins (excluant soi-même)
        # On masque la diagonale
        D_masked = D.clone()
        eye = torch.eye(N, device=A_work.device, dtype=torch.bool)
        D_masked = D_masked.masked_fill(eye, float("inf"))
        # 2 plus petits par ligne
        top2_vals, top2_idx = D_masked.topk(k=2, dim=-1, largest=False)
        # top2_idx (B, H, N, 2)

        # Pour chaque ligne i, ses 2 voisins v1, v2 → vérifier triangle inégalité
        # d(v1, v2) ≤ d(i, v1) + d(i, v2) — toujours vrai pour distance Euclidienne
        # On regarde plutôt **gap** : (d(v1, v2) / (d(i,v1) + d(i,v2)) → "tree-like" si ≈ 1
        v1_idx = top2_idx[..., 0]  # (B, H, N)
        v2_idx = top2_idx[..., 1]
        d_i_v1 = top2_vals[..., 0]  # (B, H, N)
        d_i_v2 = top2_vals[..., 1]
        # Gather d(v1, v2)
        # Pour ça, on indexe D avec v1_idx puis v2_idx
        Bb, Hh, Nn, _ = D.shape
        # D[b, h, v1_idx[b, h, i], v2_idx[b, h, i]]
        flat_idx = v1_idx * Nn + v2_idx  # (B, H, N)
        D_flat = D.reshape(Bb, Hh, Nn * Nn)
        d_v1_v2 = torch.gather(D_flat, dim=-1, index=flat_idx)

        # Tree-like indicator : d_v1_v2 close to d_i_v1 + d_i_v2
        sum_d = (d_i_v1 + d_i_v2).clamp_min(self.eps_floor)
        ratio = d_v1_v2 / sum_d  # ∈ [0, 1]
        ratio_flat = ratio.float().flatten()

        return {
            "tree_likeness_ratio_median": float(ratio_flat.median().item()),
            "tree_likeness_ratio_mean": float(ratio_flat.mean().item()),
            "fraction_quasi_tree_above_0p9": float(
                (ratio_flat > 0.9).float().mean().item()
            ),
            "fraction_non_tree_below_0p3": float(
                (ratio_flat < 0.3).float().mean().item()
            ),
            "n_triangles": int(ratio_flat.numel()),
            "n_matrices": int(B * H),
        }
