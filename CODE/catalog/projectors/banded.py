"""
banded.py — Projecteur Banded (matrices à bande).

Spec : DOC/00b B6, DOC/glossaire §Matrice à bande.

Une matrice à bande de largeur w a A_{ij} = 0 si |i − j| > w. Capture
l'attention locale focalisée (chaque token ne regarde que ses ±w voisins).

Naturellement présent dans les Transformers à attention locale (SWA,
Longformer local, etc.). Projection : garder la bande, zéro ailleurs.
"""

from __future__ import annotations

import torch

from catalog.projectors.base import Projector


class Banded(Projector):
    """Projection sur les matrices à bande de largeur fixée."""

    name = "banded"
    family = "B"

    def __init__(self, bandwidth: int) -> None:
        if bandwidth < 0:
            raise ValueError(f"bandwidth doit être ≥ 0, reçu {bandwidth}")
        self.bandwidth = bandwidth

    def project(self, A: torch.Tensor) -> torch.Tensor:
        if A.size(-1) != A.size(-2):
            raise ValueError(
                f"Banded attend carrée (..., N, N), reçu {A.shape}"
            )
        N = A.size(-1)
        i = torch.arange(N, device=A.device).view(-1, 1)
        j = torch.arange(N, device=A.device).view(1, -1)
        mask = (i - j).abs() <= self.bandwidth  # (N, N)
        # Broadcast mask sur les dims batch
        out = A.clone()
        out = torch.where(mask, A, torch.zeros_like(A))
        return out
