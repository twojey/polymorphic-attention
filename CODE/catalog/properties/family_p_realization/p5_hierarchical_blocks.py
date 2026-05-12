"""
p5_hierarchical_blocks.py — Property P5 : blocs hiérarchiques Ho-Kalman.

Spec : DOC/CATALOGUE §P5 "réalisation par blocs hiérarchiques :
matrice Hankel découpée en sous-Hankel par fenêtres temporelles, et
chaque sous-Hankel a son propre rang minimal".

V1 : partitionne A en p × p sous-blocs égaux (p ∈ {2, 4}), construit
le Hankel block pour CHAQUE sous-bloc puis calcule rang minimal local.
La distribution des rangs locaux (max - min, variance) informe si
A est "uniforme" (même rang partout) ou "hétérogène" (rangs locaux
très différents, dépendance contextuelle forte).
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class P5HierarchicalBlocks(Property):
    """P5 — rangs minimaux Hankel par sous-blocs (hiérarchie temporelle)."""

    name = "P5_hierarchical_blocks"
    family = "P"
    cost_class = 4
    requires_fp64 = True
    scope = "per_regime"

    def __init__(
        self,
        partition_levels: tuple[int, ...] = (2, 4),
        theta_cumulative: float = 0.99,
        eps_floor: float = 1e-30,
    ) -> None:
        self.partition_levels = partition_levels
        self.theta = theta_cumulative
        self.eps_floor = eps_floor

    def _hankel_rank_eff(self, A_block: torch.Tensor) -> torch.Tensor:
        """Construit Hankel d'un sous-bloc et retourne r_eff."""
        N = A_block.shape[-1]
        k = max(N // 2, 1)
        n_col = N - k
        if k < 1 or n_col < 1:
            return torch.tensor(0.0)
        H_block = torch.zeros(
            *A_block.shape[:-2], k, n_col * N,
            dtype=A_block.dtype, device=A_block.device,
        )
        for i in range(k):
            for j in range(n_col):
                if i + j < N:
                    H_block[..., i, j * N: (j + 1) * N] = A_block[..., i + j, :]
        sigmas = torch.linalg.svdvals(H_block)
        s2 = sigmas.pow(2)
        cumsum = s2.cumsum(dim=-1)
        total = cumsum[..., -1:].clamp_min(self.eps_floor)
        r_eff = ((cumsum / total) >= self.theta).float().argmax(dim=-1) + 1
        return r_eff

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, _ = A.shape
        if N < 8:
            return {"n_matrices": int(B * H), "skip_reason": "N<8"}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        results: dict[str, float | int | str | bool] = {
            "n_matrices": int(B * H),
            "seq_len": int(N),
        }
        for p in self.partition_levels:
            if N % p != 0 or N // p < 2:
                continue
            block_size = N // p
            local_ranks: list[torch.Tensor] = []
            for ii in range(p):
                for jj in range(p):
                    sub = A_work[..., ii * block_size: (ii + 1) * block_size,
                                 jj * block_size: (jj + 1) * block_size]
                    r = self._hankel_rank_eff(sub)
                    local_ranks.append(r.float().flatten())
            ranks = torch.stack(local_ranks, dim=0)  # (p², B*H)
            tag = f"p{p}"
            results[f"local_rank_{tag}_median"] = float(ranks.median().item())
            results[f"local_rank_{tag}_max"] = float(ranks.max().item())
            results[f"local_rank_{tag}_min"] = float(ranks.min().item())
            results[f"local_rank_{tag}_range"] = float((ranks.max() - ranks.min()).item())
            # Hétérogénéité = std des rangs locaux normalisée
            mean_r = ranks.mean()
            results[f"local_rank_{tag}_heterogeneity"] = float(
                (ranks.std() / mean_r.clamp_min(self.eps_floor)).item()
            )
        return results
