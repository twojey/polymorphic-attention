"""
monarch_mask.py — Projecteur Monarch (V1 : permutation + bloc-diagonal).

Spec : DOC/CATALOGUE §U2.

Une matrice Monarch N×N (N = m·b) est M = P_1 · D_1 · P_2 · D_2, où D_i sont
bloc-diagonales (m blocs b×b) et P_i sont des permutations. Le support
résultant est sparse : O(N · b) entrées non-nulles (au lieu de N²).

V1 simple : on construit le mask pour le pattern Monarch standard avec
(m, b) = (√N, √N) et on projette A par mask. Comme Butterfly, c'est une
**borne inférieure** sur la distance Monarch (la vraie factorisation
nécessite ALS).
"""

from __future__ import annotations

import math

import torch

from catalog.projectors.base import Projector


def _monarch_support_mask(n: int, m: int, b: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Mask binaire (n, n) du support Monarch avec (m, b) blocs.

    Construction : Monarch = (P_1 D_1) · (P_2 D_2), avec D_i bloc-diag (m
    blocs b×b), P_i permutation "shuffle". Support de D_1 = m × bloc b×b
    diagonaux. Support de P_2 D_2 = colonnes shuffled de D_2.
    """
    assert m * b == n, f"m*b={m*b} doit être == n={n}"
    mask = torch.zeros(n, n, device=device, dtype=torch.bool)
    # D_2 : bloc-diagonal (m blocs b×b)
    D2 = torch.zeros(n, n, device=device, dtype=torch.bool)
    for i in range(m):
        D2[i * b: (i + 1) * b, i * b: (i + 1) * b] = True
    # P_2 : shuffle perfect = permutation interleave (i*b + j → j*m + i)
    perm = torch.empty(n, dtype=torch.long, device=device)
    for i in range(m):
        for j in range(b):
            perm[i * b + j] = j * m + i
    P2_D2 = D2[:, perm]  # P_2 D_2 colonnes permutées
    # D_1 : bloc-diagonal idem
    D1 = D2.clone()
    # Support de (D_1) · (P_2 D_2) = D_1 D_2_perm (intersection des supports)
    # Pour le support : (D_1 · X)_ij ≠ 0 ⟺ ∃ k : (D_1)_ik et X_kj
    M = (D1.float() @ P2_D2.float()) > 0
    # P_1 permute encore : pour la simplicité V1, on prend P_1 = identity
    return M.to(dtype=dtype)


class MonarchMask(Projector):
    """Projection sur mask Monarch (m, b) = (√N, √N) par défaut."""

    name = "monarch_mask"
    family = "U"

    def __init__(self, m: int | None = None, b: int | None = None) -> None:
        self.m = m
        self.b = b
        self._mask_cache: dict[tuple[int, int, int, str, str], torch.Tensor] = {}

    def _get_mask(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Auto-détection (m, b) = (√N, √N) si non spécifié
        if self.m is None or self.b is None:
            sqrt_n = int(math.sqrt(n))
            # Trouver facteur exact
            for cand in range(sqrt_n, 1, -1):
                if n % cand == 0:
                    m_ = cand
                    b_ = n // cand
                    break
            else:
                m_, b_ = 1, n
        else:
            m_, b_ = self.m, self.b
            if m_ * b_ != n:
                raise ValueError(f"Monarch: m·b = {m_*b_} != n = {n}")
        key = (n, m_, b_, str(device), str(dtype))
        if key not in self._mask_cache:
            self._mask_cache[key] = _monarch_support_mask(n, m_, b_, device, dtype)
        return self._mask_cache[key]

    def project(self, A: torch.Tensor) -> torch.Tensor:
        mask = self._get_mask(A.shape[-1], A.device, A.dtype)
        return A * mask
