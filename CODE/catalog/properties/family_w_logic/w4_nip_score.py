"""
w4_nip_score.py — Property W4 : score NIP composite + détection IP.

Complément à W3 : on calcule un score NIP composite combinant :
- shatter_ratio à plusieurs tailles n (3, 4, 5, 6)
- pente log(shatter) vs log(n) (NIP ⇔ pente bornée, IP ⇔ pente ~ n)
- détection IP rigoureuse : famille shattered n-points triés vraiment

Output principal : nip_score ∈ [0, 1] (1 = NIP très probable, 0 = IP).
"""

from __future__ import annotations

import math
import random

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class W4NipScore(Property):
    """W4 — score NIP composite + détection IP rigoureuse."""

    name = "W4_nip_score"
    family = "W"
    cost_class = 3
    requires_fp64 = False
    scope = "per_regime"

    def __init__(
        self,
        binarize_threshold: float = 0.05,
        n_samples: int = 24,
        ns_to_test: tuple[int, ...] = (3, 4, 5, 6),
        seed: int = 1,
    ) -> None:
        self.tau = binarize_threshold
        self.n_samples = n_samples
        self.ns_to_test = ns_to_test
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
        valid_ns = [n for n in self.ns_to_test if N >= n]
        if not valid_ns:
            return {
                "skip_reason": f"N={N} < min ns_to_test",
                "n_matrices": int(B * H),
            }

        shatters = []
        max_ips_at = []
        for n in valid_ns:
            full = 2 ** n
            distincts: list[int] = []
            ip_count = 0
            for _ in range(self.n_samples):
                cols = rng.sample(range(N), n)
                cols_t = torch.tensor(cols, dtype=torch.long, device=flat.device)
                patterns = flat[:, cols_t]
                w = 2 ** torch.arange(n, device=flat.device, dtype=patterns.dtype)
                keys = (patterns * w).sum(dim=-1)
                nd = int(torch.unique(keys).numel())
                distincts.append(nd)
                if nd >= full:
                    ip_count += 1
            mean_ratio = sum(distincts) / (len(distincts) * full)
            shatters.append(mean_ratio)
            max_ips_at.append(ip_count)

        # Pente log(shatter * 2^n) vs log(n) = pente log(distinct) vs log(n)
        # NIP : distinct = O(n^d) ⇒ pente ≈ d
        # IP : distinct = 2^n ⇒ pente log distinct vs log n ≈ n / log n → ∞
        log_ns = torch.tensor([math.log(n) for n in valid_ns])
        log_distincts = torch.tensor(
            [math.log(s * (2 ** n) + 1e-12) for s, n in zip(shatters, valid_ns)]
        )
        if len(valid_ns) >= 2:
            slope = float(
                ((log_ns * log_distincts).sum() - len(valid_ns) * log_ns.mean() * log_distincts.mean())
                / ((log_ns.pow(2).sum() - len(valid_ns) * log_ns.mean() ** 2).clamp_min(1e-12))
            )
        else:
            slope = float("nan")

        # NIP score = 1 si pente <= 3 (polynomial), 0 si > log(2^n_max) (full IP)
        max_n = max(valid_ns)
        max_slope_ip = math.log(2 ** max_n) / math.log(max_n)  # = n_max / log n_max
        nip_score = max(0.0, 1.0 - slope / max_slope_ip) if math.isfinite(slope) else float("nan")

        results: dict[str, float | int | str | bool] = {
            "n_matrices": int(B * H),
            "tau": self.tau,
            "slope_log_distinct_vs_log_n": float(slope),
            "nip_score": float(nip_score),
        }
        for n, s, ip in zip(valid_ns, shatters, max_ips_at):
            results[f"shatter_ratio_n{n}"] = float(s)
            results[f"ip_sample_count_n{n}"] = int(ip)
        results["full_ip_detected"] = bool(max_ips_at[-1] > 0)
        return results
