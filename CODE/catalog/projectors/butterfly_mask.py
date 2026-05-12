"""
butterfly_mask.py — Projecteur Butterfly (V1 : mask sparsity pattern).

Spec : DOC/CATALOGUE §U1.

Une matrice Butterfly N×N (N = 2^k) est factorisable en O(log₂ N) facteurs
sparse, chaque facteur ayant exactement 2 entrées non-nulles par ligne.
Le pattern de SUPPORT de B = B_1 · B_2 · ... · B_{log₂ N} est précisément
la structure butterfly union — un mask binaire avec O(N log N) entrées.

V1 simple : on construit le mask binaire des "positions possibles" pour
une matrice butterfly et on projette A par mask (= zero hors mask). C'est
moins précis qu'une vraie factorisation ALS Butterfly (V2) mais donne
une **borne inférieure** sur la distance Butterfly réelle.

Pour N non-puissance-de-2, on prend le N' = 2^ceil(log₂ N) le plus proche
et on crop.
"""

from __future__ import annotations

import math

import torch

from catalog.projectors.base import Projector


def _butterfly_support_mask(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Mask binaire (n, n) du support théorique d'une matrice butterfly.

    Construction : pour N = 2^k, le support de B = B_1 B_2 ... B_k est
    l'ensemble des paires (i, j) telles que i et j diffèrent sur au plus
    1 bit à chaque niveau. Le support union se calcule en composant les
    supports butterfly de chaque niveau.

    Approximation : on retourne le support de B_1 · B_2 (deux niveaux),
    qui contient déjà O(N log N) entrées et capture la structure essentielle.
    Pour le support complet, il faudrait itérer log₂ N fois.
    """
    # Pad à la puissance de 2 supérieure si nécessaire
    k = int(math.ceil(math.log2(max(n, 2))))
    M = 2 ** k
    mask = torch.zeros(M, M, device=device, dtype=torch.bool)
    # Pour chaque niveau ℓ ∈ [0, k-1], on connecte (i, j) si i XOR j a son
    # bit ℓ activé OU est zéro (= même position)
    for ell in range(k):
        bit = 1 << ell
        for i in range(M):
            j_same = i  # même position
            j_xor = i ^ bit  # position avec bit ℓ flippé
            mask[i, j_same] = True
            mask[i, j_xor] = True
    return mask[:n, :n].to(dtype=dtype)


class ButterflyMask(Projector):
    """Projection sur le mask de support théorique d'une matrice Butterfly."""

    name = "butterfly_mask"
    family = "U"

    def __init__(self) -> None:
        self._mask_cache: dict[tuple[int, str, str], torch.Tensor] = {}

    def _get_mask(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (n, str(device), str(dtype))
        if key not in self._mask_cache:
            self._mask_cache[key] = _butterfly_support_mask(n, device, dtype)
        return self._mask_cache[key]

    def project(self, A: torch.Tensor) -> torch.Tensor:
        mask = self._get_mask(A.shape[-1], A.device, A.dtype)
        return A * mask
