"""
s_spectral.py — signal S_Spectral : rang effectif r_eff sur fenêtre glissante K.

Spec : DOC/01b §1, §8 (pré-enregistrer K).

Coût : O(K² log K) par token via SVD partielle randomisée.

Convention :
- Pour chaque token t, fenêtre = A[..., t-K+1:t+1, t-K+1:t+1] (K×K).
- r_eff = nombre de valeurs singulières au-dessus de τ · σ_max.
- Pour t < K-1 : padding par premier token disponible (warmup), ou r_eff=0.
"""

from __future__ import annotations

import torch


def _r_eff(matrix: torch.Tensor, tau: float = 1e-3) -> int:
    s = torch.linalg.svdvals(matrix)
    if s.numel() == 0:
        return 0
    s_max = s[0].item()
    if s_max == 0:
        return 0
    return int((s > tau * s_max).sum().item())


def compute_s_spectral(
    attn_per_layer: list[torch.Tensor],
    *,
    K: int = 64,
    tau: float = 1e-3,
    use_randomized: bool = True,
    q_oversample: int = 8,
) -> torch.Tensor:
    """S_Spectral pour chaque (couche, batch, tête, token).

    `attn_per_layer` : liste de L tensors (B, H, N, N) FP64.
    Retourne (L, B, H, N) FP64.

    Pour t < K-1, on prend une fenêtre élargie (du début à t+1).
    """
    L = len(attn_per_layer)
    B, H, N, _ = attn_per_layer[0].shape
    device = attn_per_layer[0].device

    out = torch.zeros(L, B, H, N, dtype=torch.float64, device=device)

    for ell, A_layer in enumerate(attn_per_layer):
        for b in range(B):
            for h in range(H):
                A = A_layer[b, h]  # (N, N)
                for t in range(N):
                    start = max(0, t - K + 1)
                    window = A[start : t + 1, start : t + 1]
                    if use_randomized and window.size(0) > q_oversample + 1:
                        # SVD lowrank randomisée pour fenêtres assez grandes
                        try:
                            _, s, _ = torch.svd_lowrank(window, q=min(q_oversample, window.size(0) - 1))
                        except Exception:
                            s = torch.linalg.svdvals(window)
                    else:
                        s = torch.linalg.svdvals(window)
                    if s.numel() > 0:
                        s_max = s[0].item()
                        out[ell, b, h, t] = float((s > tau * s_max).sum().item()) if s_max > 0 else 0.0
    return out
