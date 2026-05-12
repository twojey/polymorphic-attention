"""
g6_bernstein_sato_proxy.py — Property G6 : proxy Bernstein-Sato.

Spec : DOC/CATALOGUE §G6 "polynôme de Bernstein-Sato b(s) de
det(A − λI), indicateur frontière géométrie algébrique".

La théorie des polynômes de Bernstein-Sato fournit, pour un polynôme f
(ici f = det(A − λI) vu comme fonction holomorphe de λ), un polynôme
b(s) minimal tel que :
    b(s) · f^s = P(s, λ, ∂_λ) · f^{s+1}
Les racines de b(s) sont rationnelles et < 0 ; leur log_2 minimum
encode la singularité de f.

V1 proxy numérique (sans Risa-Asir, Macaulay2) : on mesure la
**non-générique** structure du discriminant via :
- nombre de valeurs propres "dégénérées" (clusters proches dans le plan complexe)
- ratio log(N_cluster) / log(N) — proxy de la complexité de singularité
- minimum séparation eigenvalues / N (proxy log-rationality)

Si A est générique → eigvalues sont bien séparées, ratio ~ 1, proxy bas.
Si A a haute symétrie / structure → clusters denses → proxy haut.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class G6BernsteinSatoProxy(Property):
    """G6 — proxy de la complexité Bernstein-Sato via clustering eigvalues."""

    name = "G6_bernstein_sato_proxy"
    family = "G"
    cost_class = 4
    requires_fp64 = True
    scope = "per_regime"

    def __init__(self, cluster_radius: float = 1e-3, eps_floor: float = 1e-30) -> None:
        self.cluster_radius = cluster_radius
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, _ = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        flat = A_work.reshape(B * H, N, N)

        n_clusters_list: list[int] = []
        min_sep_list: list[float] = []
        for i in range(flat.shape[0]):
            try:
                eigs = torch.linalg.eigvals(flat[i])  # complex
            except Exception:
                continue
            # Sort eigvalues by real part for stable cluster detection
            real = eigs.real
            imag = eigs.imag
            # Stack pour distance dans le plan complexe
            pts = torch.stack([real, imag], dim=-1).float()  # (N, 2)
            # Distance matrice pairwise
            d = torch.cdist(pts.unsqueeze(0), pts.unsqueeze(0)).squeeze(0)  # (N, N)
            # Mask diagonale
            d.fill_diagonal_(float("inf"))
            min_sep = float(d.min().item())
            min_sep_list.append(min_sep)
            # Comptage simple : nb composantes connexes du graphe (d < radius)
            adj = d < self.cluster_radius  # (N, N)
            visited = torch.zeros(N, dtype=torch.bool)
            n_comp = 0
            for v in range(N):
                if visited[v]:
                    continue
                # BFS
                stack = [v]
                while stack:
                    u = stack.pop()
                    if visited[u]:
                        continue
                    visited[u] = True
                    neighbours = torch.where(adj[u])[0].tolist()
                    for w in neighbours:
                        if not visited[w]:
                            stack.append(w)
                n_comp += 1
            n_clusters_list.append(n_comp)

        if not n_clusters_list:
            return {"n_matrices": int(B * H), "skip_reason": "all eigvals failed"}

        n_clusters_t = torch.tensor(n_clusters_list, dtype=torch.float64)
        min_sep_t = torch.tensor(min_sep_list, dtype=torch.float64)
        log_N = torch.log(torch.tensor(float(N))).clamp_min(self.eps_floor)
        log_clusters = (n_clusters_t.clamp_min(1.0)).log()
        complexity_ratio = log_clusters / log_N  # ~1 si générique, <1 si très clusterisé

        return {
            "bs_complexity_ratio_median": float(complexity_ratio.median().item()),
            "bs_complexity_ratio_mean": float(complexity_ratio.mean().item()),
            "bs_n_clusters_median": float(n_clusters_t.median().item()),
            "bs_n_clusters_min": float(n_clusters_t.min().item()),
            "bs_min_separation_median": float(min_sep_t.median().item()),
            "bs_fraction_highly_degenerate": float(
                (complexity_ratio < 0.5).float().mean().item()
            ),
            "n_matrices_ok": int(n_clusters_t.numel()),
            "N": int(N),
        }
