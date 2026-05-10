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
    use_randomized: bool = True,    # ignoré dans la version vectorisée
    q_oversample: int = 8,           # ignoré dans la version vectorisée
    stride: int = 1,                # NOUVEAU : SVD sur tous les `stride` tokens
) -> torch.Tensor:
    """S_Spectral pour chaque (couche, batch, tête, token), version vectorisée.

    `attn_per_layer` : liste de L tensors (B, H, N, N) FP64.
    Retourne (L, B, H, N) FP64.

    Pour chaque token t, fenêtre causale K×K se terminant à t. Pour t < K-1,
    padding par zéros au coin haut-gauche (warmup → r_eff sous-estimé pour
    les premiers tokens, accepté).

    Stratégie vectorisée (vs V1 quadruple boucle Python) :
    1. Padding A à (B, H, N+K-1, N+K-1) avec zéros en haut-gauche.
    2. Stack des N fenêtres (extracted via slicing) en (B, H, N, K, K).
    3. **Cast en FP32** avant SVD (RTX 5090 consumer = FP64 ~1/64 du FP32 ;
       pour un compteur r_eff au-dessus de tau·σ_max, FP32 est largement
       suffisant et 50-100× plus rapide).
    4. SVD batchée sur (B*H*N, K, K) → (B*H*N, K) singular values.
    5. r_eff = (s > tau * s_max) per token.

    Coût : O(B*H*N * K² log K) en SVD batchée GPU vs L*B*H*N en Python pur V1.
    """
    import torch.nn.functional as F

    L = len(attn_per_layer)
    B, H, N, _ = attn_per_layer[0].shape
    device = attn_per_layer[0].device

    out = torch.zeros(L, B, H, N, dtype=torch.float64, device=device)

    if N < K:
        # Pas assez de tokens pour une fenêtre complète, retourne zéros (warmup).
        return out

    # Positions où on calcule la SVD : t ∈ [K-1, N-1] avec pas `stride`.
    # Hors `stride` ou warmup (t < K-1), out reste à 0.
    ts = list(range(K - 1, N, stride))
    n_full = len(ts)
    if n_full == 0:
        return out

    for ell, A_layer in enumerate(attn_per_layer):
        # A_layer : (B, H, N, N). Stack des fenêtres causales aux positions ts.
        windows = torch.stack(
            [A_layer[:, :, t - K + 1 : t + 1, t - K + 1 : t + 1] for t in ts],
            dim=2,
        )  # (B, H, n_full, K, K)
        windows_flat = windows.reshape(-1, K, K).contiguous()  # (B*H*n_full, K, K)

        # Stratégie : eigvalsh(W W.T + εI) au lieu de svdvals(W).
        # Raison 1 : sur attention rank-deficient (concentration token),
        #   cuSolver SVD fail à converger → fallback iterative très lent.
        # Raison 2 : eigvalsh sans ridge fail aussi sur eigenvalues répétées
        #   (matrices low-rank → plusieurs zéros). Ajout d'un ridge ε·I
        #   (jitter) avant eigvalsh garantit la convergence ; soustrait
        #   après pour précision sur σ².
        # σ_i² = eigenvalue_i de (W W.T), donc σ_i = sqrt(λ_i).
        W = windows_flat.float()
        M = W @ W.transpose(-2, -1)             # (B*H*n_full, K, K) PSD
        ridge = 1e-6
        eye = torch.eye(K, device=M.device, dtype=M.dtype).expand_as(M[:1])
        M_reg = M + ridge * eye                 # ridge regularisation
        ev = torch.linalg.eigvalsh(M_reg)       # ascending, (B*H*n_full, K)
        s2_desc = (ev.flip(-1) - ridge).clamp_min(0)   # σ² descending
        s_max2 = s2_desc[:, 0:1].clamp_min(1e-20)
        # r_eff = # σ_i > tau·σ_max ⇔ # σ_i² > tau²·σ_max²
        r_eff = (s2_desc > (tau * tau) * s_max2).sum(dim=-1).to(torch.float64)

        # Place les valeurs aux positions exactes ts (pas au slice contigu)
        out[ell, :, :, torch.tensor(ts, device=device)] = r_eff.reshape(B, H, n_full)

    return out
