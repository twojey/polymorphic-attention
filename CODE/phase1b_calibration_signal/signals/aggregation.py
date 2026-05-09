"""
aggregation.py — agrégation cross-layer / cross-head des signaux.

Spec : DOC/01b §3.

1. Per-layer max-pool sur les têtes : s_layer(ℓ, t) = max_h S(ℓ, h, t)
2. Concatenate sur la deep-stack    : s(t) = [s_layer(1, t); ...; s_layer(L, t)]

Sortie : (B, L, N) si on garde la trajectoire couche par couche.
"""

from __future__ import annotations

import torch


def aggregate_signal_per_token(signal: torch.Tensor) -> torch.Tensor:
    """Réduit (L, B, H, N) → (B, L, N) : max sur têtes, concat sur couches.

    Le résultat est la "trajectoire couche par couche" pour chaque token,
    consommable par le Spectromètre comme vecteur de dimension L par signal.
    """
    assert signal.ndim == 4, f"attendu (L, B, H, N), reçu {tuple(signal.shape)}"
    return signal.max(dim=2).values.permute(1, 0, 2).contiguous()  # (B, L, N)
