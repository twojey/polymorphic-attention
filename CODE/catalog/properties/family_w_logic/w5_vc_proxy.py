"""
w5_vc_proxy.py — Property W5 : proxy de la VC-dimension.

VC-dim = plus grand n tel que la famille shatters un ensemble de taille n
(génère toutes les 2^n configurations). Diagnostic combinatoire de la
complexité logique de l'attention binarisée.

V1 : on cherche le **plus grand n** dans une grille pour lequel au moins
un sous-ensemble de n colonnes est shattered (toutes 2^n patterns observés).
"""

from __future__ import annotations

import random

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class W5VcProxy(Property):
    """W5 — VC-dim proxy : plus grand n shatterable par la famille."""

    name = "W5_vc_proxy"
    family = "W"
    cost_class = 3
    requires_fp64 = False
    scope = "per_regime"

    def __init__(
        self,
        binarize_threshold: float = 0.05,
        n_samples: int = 32,
        n_max: int = 8,
        seed: int = 2,
    ) -> None:
        self.tau = binarize_threshold
        self.n_samples = n_samples
        self.n_max = n_max
        self.seed = seed

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, _ = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        row_max = A_work.amax(dim=-1, keepdim=True).clamp_min(1e-30)
        A_bin = (A_work > (self.tau * row_max)).to(torch.long)
        flat = A_bin.reshape(B * H * N, N)

        rng = random.Random(self.seed)
        # Pour chaque n croissant, tester si shatter
        last_shattered = 0
        attempts_per_n: dict[int, int] = {}
        first_unshatter = -1
        for n in range(2, min(self.n_max, N) + 1):
            full = 2 ** n
            shattered = False
            tries = min(self.n_samples, max(1, full * 2))
            attempts_per_n[n] = tries
            for _ in range(tries):
                cols = rng.sample(range(N), n)
                cols_t = torch.tensor(cols, dtype=torch.long, device=flat.device)
                patterns = flat[:, cols_t]
                w = 2 ** torch.arange(n, device=flat.device, dtype=patterns.dtype)
                keys = (patterns * w).sum(dim=-1)
                if int(torch.unique(keys).numel()) >= full:
                    shattered = True
                    break
            if shattered:
                last_shattered = n
            elif first_unshatter < 0:
                first_unshatter = n
                # On peut s'arrêter ici si on veut une borne plus aggressive,
                # mais on continue pour rapporter détail.

        results: dict[str, float | int | str | bool] = {
            "vc_proxy_max_shattered_n": int(last_shattered),
            "vc_proxy_first_unshattered_n": int(first_unshatter),
            "n_max_tested": int(min(self.n_max, N)),
            "tau": self.tau,
            "n_matrices": int(B * H),
            "n_families": int(flat.shape[0]),
        }
        return results
