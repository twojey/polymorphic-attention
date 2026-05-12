"""
w3_nip_dependence.py — Property W3 : NIP / dependence measure (model theory).

Spec : DOC/CATALOGUE §W3 "Independence Property (IP) vs No Independence
Property (NIP) — un opérateur a NIP si toute formule définissable a une
VC-dimension bornée. Mesure combinatoire de la complexité logique".

V1 : on traite A (binarisé via seuillage) comme une famille d'ensembles
(les lignes), et calcule la **shattering function** π_F(n) :
    π_F(n) = max sur ensembles S de taille n : |{F ∩ S : F ∈ family}|
NIP ⟺ π_F(n) polynomiale en n (VC dim bornée).
IP ⟺ π_F(n) = 2^n (full shattering, dim infinie).

V1 proxy via :
- VC-shatter ratio à n=3 et n=4 : nombre de configurations distinctes
  observées / 2^n
- ratio "near-full-shatter" = fraction de pairs/triples avec toutes les
  configurations

Approximation : on échantillonne des sous-ensembles de colonnes de taille n.
"""

from __future__ import annotations

import math
import random

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class W3NIPDependence(Property):
    """W3 — proxy NIP via VC-shattering ratio sur lignes binarisées."""

    name = "W3_nip_dependence"
    family = "W"
    cost_class = 3
    requires_fp64 = False
    scope = "per_regime"

    def __init__(
        self,
        binarize_threshold: float = 0.05,
        n_samples: int = 32,
        ns_to_test: tuple[int, ...] = (3, 4),
        seed: int = 0,
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
        # Binarisation par seuil sur max ligne
        row_max = A_work.amax(dim=-1, keepdim=True).clamp_min(1e-30)
        A_bin = (A_work > (self.tau * row_max)).to(torch.long)  # (B, H, N, N)
        # Aplatir (B, H) → considérer les lignes comme family of sets
        flat = A_bin.reshape(B * H * N, N)  # (n_families, N)

        rng = random.Random(self.seed)
        results: dict[str, float | int | str | bool] = {
            "n_matrices": int(B * H),
            "seq_len": int(N),
            "n_families_total": int(flat.shape[0]),
            "tau": self.tau,
        }

        for n in self.ns_to_test:
            if N < n:
                continue
            full_configs = 2 ** n
            shatter_ratios: list[float] = []
            for _ in range(self.n_samples):
                # Échantillonne n colonnes parmi N
                cols = rng.sample(range(N), n)
                cols_t = torch.tensor(cols, dtype=torch.long, device=flat.device)
                # Pour chaque famille (ligne), extrait pattern sur ces n colonnes
                patterns = flat[:, cols_t]  # (n_families, n)
                # Compte distinct patterns
                # Conversion en clé entière base 2
                weights = (2 ** torch.arange(n, device=flat.device, dtype=patterns.dtype))
                keys = (patterns * weights).sum(dim=-1)  # (n_families,)
                n_distinct = int(torch.unique(keys).numel())
                shatter_ratios.append(n_distinct / full_configs)

            if shatter_ratios:
                t = torch.tensor(shatter_ratios)
                results[f"vc_shatter_ratio_n{n}_mean"] = float(t.mean().item())
                results[f"vc_shatter_ratio_n{n}_max"] = float(t.max().item())
                results[f"fraction_full_shatter_n{n}"] = float(
                    (t > 0.95).float().mean().item()
                )

        # NIP probabilité = inverse du shatter à n_max
        max_n = max(self.ns_to_test)
        key = f"vc_shatter_ratio_n{max_n}_max"
        if key in results:
            results["nip_probability_proxy"] = 1.0 - float(results[key])
        return results
