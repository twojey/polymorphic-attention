"""
i3_head_clustering.py — Property I3 : clustering hiérarchique des têtes.

Spec : DOC/CATALOGUE §I3 "hierarchical clustering sur pairs (h_1, h_2)".

V1 : matrice de distances pairwise entre signatures heads (= mean_batch
A[b, h, :, :]). Puis on calcule combien de "clusters distincts" existent
au seuil τ (= τ × diamètre).

Métriques :
- n_clusters_at_threshold pour plusieurs τ
- mean_intra_cluster_distance
- silhouette score approximé (intra vs inter)
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


def _single_link_clusters(D: torch.Tensor, threshold: float) -> int:
    """Single-linkage clustering : compte nb de clusters connectés à threshold.

    D : (H, H) matrice symétrique de distances. Retourne nb composantes
    connexes du graphe {(i, j) : D[i, j] ≤ threshold}.
    """
    H = D.shape[0]
    parent = list(range(H))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(H):
        for j in range(i + 1, H):
            if D[i, j].item() <= threshold:
                union(i, j)
    roots = {find(i) for i in range(H)}
    return len(roots)


@register_property
class I3HeadClustering(Property):
    """I3 — clustering hiérarchique single-link des têtes (n_clusters @ seuils)."""

    name = "I3_head_clustering"
    family = "I"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(
        self,
        threshold_ratios: tuple[float, ...] = (0.25, 0.50, 0.75),
        eps_floor: float = 1e-30,
    ) -> None:
        self.threshold_ratios = threshold_ratios
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if H < 2:
            return {"n_heads": H, "n_matrices": int(B * H)}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Signature head h = mean_b A[b, h, :, :] (H, N, N2)
        sig = A_work.mean(dim=0)
        sig_flat = sig.flatten(start_dim=-2)  # (H, N*N2)
        # Pairwise distance euclidienne
        diff = sig_flat.unsqueeze(0) - sig_flat.unsqueeze(1)  # (H, H, N*N2)
        D = diff.norm(dim=-1)  # (H, H)
        diam = D.max().item()
        if diam < self.eps_floor:
            # Toutes têtes identiques → 1 cluster pour tous les seuils
            results: dict[str, float | int | str | bool] = {
                "diameter": diam, "n_heads": H, "n_matrices": int(B * H),
            }
            for r in self.threshold_ratios:
                tag = f"{r:.2f}".replace(".", "p")
                results[f"n_clusters_at_threshold_{tag}"] = 1
            return results

        results = {
            "diameter": float(diam),
            "n_heads": H,
            "n_matrices": int(B * H),
        }
        for r in self.threshold_ratios:
            threshold = r * diam
            n_c = _single_link_clusters(D, threshold)
            tag = f"{r:.2f}".replace(".", "p")
            results[f"n_clusters_at_threshold_{tag}"] = n_c

        # Distance moyenne pairwise (inter-paires non identité)
        triu = torch.triu_indices(H, H, offset=1)
        pair_d = D[triu[0], triu[1]]
        results["mean_pair_distance"] = float(pair_d.mean().item())
        results["max_pair_distance"] = float(pair_d.max().item())
        return results
