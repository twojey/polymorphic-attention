"""
c9_wasserstein.py — Property C9 : distances 1-Wasserstein entre paires de lignes.

Spec : DOC/CATALOGUE §C9.

Pour deux distributions p, q sur {0, ..., N-1}, le 1-Wasserstein (Earth
Mover's Distance) sur la métrique d(i, j) = |i - j| est :

    W_1(p, q) = Σ_i |CDF_p(i) − CDF_q(i)|

C'est calculable en O(N) par paire. On échantillonne k paires de lignes
aléatoirement parmi les N²/2 possibles (pour rester O(B·H·k·N) global).
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class C9Wasserstein(Property):
    """C9 — distance 1-Wasserstein entre paires de lignes (échantillonnée)."""

    name = "C9_wasserstein"
    family = "C"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(
        self, n_pairs: int = 32, seed: int = 0, eps_floor: float = 1e-30
    ) -> None:
        if n_pairs < 1:
            raise ValueError("n_pairs ≥ 1")
        self.n_pairs = n_pairs
        self.seed = seed
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N < 2:
            return {"n_matrices": int(B * H), "skip_reason": "N too small for pairs"}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        row_sum = A_work.sum(dim=-1, keepdim=True).clamp_min(self.eps_floor)
        P = A_work / row_sum

        # Tirer n_pairs paires de lignes (t1, t2) sans tirage par batch
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)
        n_pairs = min(self.n_pairs, N * (N - 1) // 2)
        t1 = torch.randint(0, N, (n_pairs,), generator=gen)
        t2 = torch.randint(0, N, (n_pairs,), generator=gen)
        mask_diff = t1 != t2
        t1 = t1[mask_diff]
        t2 = t2[mask_diff]
        if t1.numel() == 0:
            return {"n_matrices": int(B * H), "skip_reason": "All t1=t2"}

        t1 = t1.to(A_work.device)
        t2 = t2.to(A_work.device)

        # CDF des deux lignes
        P1 = P[..., t1, :]  # (B, H, n_pairs, N2)
        P2 = P[..., t2, :]
        cdf1 = P1.cumsum(dim=-1)
        cdf2 = P2.cumsum(dim=-1)
        # W_1 = Σ |CDF1 - CDF2|
        W1 = (cdf1 - cdf2).abs().sum(dim=-1)  # (B, H, n_pairs)

        w_flat = W1.float().flatten()

        return {
            "wasserstein_median": float(w_flat.median().item()),
            "wasserstein_mean": float(w_flat.mean().item()),
            "wasserstein_p90": float(w_flat.quantile(0.90).item()),
            "wasserstein_max": float(w_flat.max().item()),
            "n_pairs_evaluated": int(t1.numel()),
            "n_matrices": int(B * H),
        }
