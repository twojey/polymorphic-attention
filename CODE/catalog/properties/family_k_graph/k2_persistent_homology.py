"""
k2_persistent_homology.py — Property K2 : persistent homology (TDA).

Spec : DOC/CATALOGUE §K2 "diagrammes de persistance H_0, H_1 sur la
filtration sous-niveaux du graphe (1 − A) ; détecte cycles, composantes
connexes émergent à différentes échelles".

V1 sans gudhi/ripser (qui sont des deps lourdes) : on implémente le calcul
de H_0 (composantes connexes) via union-find sur la filtration des
distances (1 − A). On approxime H_1 (cycles) via la formule d'Euler :
    χ = V − E + F → β_1 = 1 + E − V (pour graphe sans triangle)
Plus précisément : β_1 = E − V + (n composantes connexes).

Si gudhi est disponible (optional import), on peut switcher vers le
vrai calcul. V1 : version pure-torch sans dépendance externe.

Sortie :
- bottleneck lifetime de H_0 (durée de vie max d'une composante)
- nombre de cycles H_1 (Betti 1)
- distribution des "lifetimes" des composantes
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


class _UnionFind:
    """Union-Find léger pour calcul Betti_0."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n_components = n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.n_components -= 1
        return True


@register_property
class K2PersistentHomology(Property):
    """K2 — persistent homology proxy (β_0 lifetimes, β_1 via Euler)."""

    name = "K2_persistent_homology"
    family = "K"
    cost_class = 4
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self, max_edges: int = 4096, eps_floor: float = 1e-30) -> None:
        self.max_edges = max_edges
        self.eps_floor = eps_floor

    def _compute_for_one(self, A_sym: torch.Tensor) -> dict[str, float]:
        N = A_sym.shape[0]
        # Distance proxy = 1 - A_sym (lower = more connected)
        d = 1.0 - A_sym
        d.fill_diagonal_(float("inf"))
        # All edges (i < j) sorted by distance
        iu = torch.triu_indices(N, N, offset=1)
        weights = d[iu[0], iu[1]]  # (E,)
        n_edges = weights.numel()
        if n_edges > self.max_edges:
            top = torch.topk(-weights, k=self.max_edges).indices  # smallest weights
            iu = iu[:, top]
            weights = weights[top]
        order = torch.argsort(weights)
        uf = _UnionFind(N)
        lifetimes: list[float] = []
        comp_birth = [0.0] * N  # all born at t=0 (vertices)
        # Cycle count : E - V + components
        e_used = 0
        n_triangles = 0  # not tracked exactly without expensive scan
        last_thresh = 0.0
        for idx in order.tolist():
            t = float(weights[idx].item())
            i, j = int(iu[0, idx].item()), int(iu[1, idx].item())
            ra, rb = uf.find(i), uf.find(j)
            if ra != rb:
                # Merge → composante meurt
                lifetime = t - 0.0  # tous nés à 0
                lifetimes.append(lifetime)
                uf.union(i, j)
                e_used += 1
            else:
                # Edge ferme cycle
                e_used += 1
            last_thresh = t

        # Betti_0 final = n_components à t = max_threshold
        beta_0 = uf.n_components
        # Betti_1 ≈ E - V + beta_0 (Euler pour graphe sans 2-cell)
        beta_1 = max(e_used - N + beta_0, 0)

        if lifetimes:
            lifetimes_t = torch.tensor(lifetimes, dtype=torch.float32)
            max_life = float(lifetimes_t.max().item())
            median_life = float(lifetimes_t.median().item())
        else:
            max_life = 0.0
            median_life = 0.0

        return {
            "beta_0_final": float(beta_0),
            "beta_1": float(beta_1),
            "max_lifetime": max_life,
            "median_lifetime": median_life,
            "n_edges_used": float(e_used),
        }

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, _ = A.shape
        if N < 4:
            return {"n_matrices": int(B * H), "skip_reason": "N too small"}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Symétriser : A_sym = (A + A.T) / 2
        A_sym = (A_work + A_work.transpose(-2, -1)) / 2
        # Limiter coût : sub-échantillonner si trop de matrices (B*H > 16)
        n_total = B * H
        n_proc = min(n_total, 16)
        flat = A_sym.reshape(n_total, N, N)[:n_proc]

        all_stats: list[dict[str, float]] = []
        for k in range(n_proc):
            stats = self._compute_for_one(flat[k])
            all_stats.append(stats)

        if not all_stats:
            return {"n_matrices": int(n_total), "skip_reason": "no valid mats"}

        def _aggr(key: str) -> float:
            vals = [s[key] for s in all_stats]
            t = torch.tensor(vals, dtype=torch.float32)
            return float(t.median().item())

        return {
            "tda_beta_0_median": _aggr("beta_0_final"),
            "tda_beta_1_median": _aggr("beta_1"),
            "tda_max_lifetime_median": _aggr("max_lifetime"),
            "tda_median_lifetime_median": _aggr("median_lifetime"),
            "tda_n_processed": int(n_proc),
            "tda_n_total": int(n_total),
            "n_matrices": int(n_total),
            "seq_len": int(N),
        }
