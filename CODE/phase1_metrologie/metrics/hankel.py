"""
hankel.py — rang de Hankel numérique d'une matrice d'attention.

Définition (DOC/01 §3.1) :
1. Construire la matrice de Hankel H(A) à partir de A ∈ ℝ^{N×N}.
2. Calculer son rang numérique = nombre de valeurs singulières
   au-dessus d'un seuil τ relatif à σ_max.

La matrice de Hankel d'un signal x ∈ ℝ^L est :
    H_{i,j} = x[i + j],  i ∈ [0, p), j ∈ [0, L - p + 1)

Pour A ∈ ℝ^{N×N}, on hankellise sa première super-diagonale (vue comme
un signal 1D), ou — convention plus utilisée en attention — on lit la
ligne moyenne / la diagonale comme signal et on hankellise.

Convention V1 : pour chaque ligne i de A, on construit le vecteur
[A[i, i], A[i, i+1], ..., A[i, N-1]] (la "tail" causale partant du token i),
on en fait la matrice de Hankel, et on prend le rang numérique. Le rang
de Hankel global de A est la moyenne (ou max) sur les lignes.

Cette définition est ad hoc — elle peut être révisée si la phase 2 révèle
une convention plus utile. À pré-enregistrer dans `OPS/configs/phase1/thresholds.yaml`.
"""

from __future__ import annotations

import torch


def hankelize(signal: torch.Tensor, p: int) -> torch.Tensor:
    """Construit la matrice de Hankel `p × (L - p + 1)` du signal.

    H[i, j] = signal[i + j].
    """
    assert signal.ndim == 1, "signal doit être 1D"
    L = signal.numel()
    assert 0 < p <= L
    q = L - p + 1
    H = torch.empty(p, q, dtype=signal.dtype, device=signal.device)
    for i in range(p):
        H[i, :] = signal[i : i + q]
    return H


def numerical_rank(matrix: torch.Tensor, tau: float = 1e-3) -> int:
    """Nombre de valeurs singulières > tau · σ_max."""
    if matrix.numel() == 0:
        return 0
    s = torch.linalg.svdvals(matrix)
    if s.numel() == 0:
        return 0
    s_max = s[0].item()
    if s_max == 0:
        return 0
    threshold = tau * s_max
    return int((s > threshold).sum().item())


def hankel_rank_of_signal(signal: torch.Tensor, *, p: int | None = None, tau: float = 1e-3) -> int:
    """Rang numérique de la matrice de Hankel d'un signal 1D."""
    L = signal.numel()
    if p is None:
        p = L // 2
    p = max(1, min(p, L - 1))
    H = hankelize(signal, p)
    return numerical_rank(H, tau=tau)


def hankel_rank_of_attention(
    A: torch.Tensor, *, p: int | None = None, tau: float = 1e-3, reduction: str = "mean"
) -> torch.Tensor:
    """Rang de Hankel d'une matrice d'attention.

    A : (..., N, N), float (typiquement FP64).
    Retourne un scalaire (reduction="mean" ou "max") ou un tensor (..., N) si
    reduction="none".
    """
    assert A.ndim >= 2 and A.size(-1) == A.size(-2)
    N = A.size(-1)
    flat = A.reshape(-1, N, N)
    B = flat.size(0)
    ranks = torch.zeros(B, N, dtype=torch.float32, device=A.device)
    for b in range(B):
        for i in range(N):
            tail = flat[b, i, i:]  # signal causal partant de la diagonale
            if tail.numel() < 2:
                ranks[b, i] = 0.0
            else:
                ranks[b, i] = float(hankel_rank_of_signal(tail, p=p, tau=tau))
    if reduction == "none":
        return ranks.reshape(*A.shape[:-2], N)
    if reduction == "mean":
        return ranks.mean()
    if reduction == "max":
        return ranks.max()
    raise ValueError(f"reduction inconnue : {reduction}")


def hankel_rank_numerical(
    A: torch.Tensor, *, tau: float = 1e-3, reduction: str = "mean"
) -> torch.Tensor:
    """Alias public — voir `hankel_rank_of_attention`."""
    return hankel_rank_of_attention(A, tau=tau, reduction=reduction)
