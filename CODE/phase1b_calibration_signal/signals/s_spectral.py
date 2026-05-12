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

import os
from multiprocessing import get_context

import numpy as np
import torch


# === Workers multiprocessing pour eigvalsh parallélisé sur 4 cores ===
# Pattern emprunté à bench/spearman.py : fork + variables globales en COW.
# Chaque worker tourne avec OPENBLAS_NUM_THREADS=1 (hérité du parent + defense
# in depth). 4 cores × 1 thread BLAS = 4× speedup sans risque de deadlock.

_WORKER_M_REG: np.ndarray | None = None
_WORKER_RIDGE: float = 1e-6
_WORKER_TAU: float = 1e-3


def _spectral_worker_init(m_reg: np.ndarray, ridge: float, tau: float) -> None:
    global _WORKER_M_REG, _WORKER_RIDGE, _WORKER_TAU
    _WORKER_M_REG = m_reg
    _WORKER_RIDGE = ridge
    _WORKER_TAU = tau
    # Defense in depth (déjà hérité via fork normalement)
    for v in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[v] = '1'


def _spectral_worker_chunk(idx_range: tuple[int, int]) -> np.ndarray:
    start, end = idx_range
    assert _WORKER_M_REG is not None
    M_chunk = torch.from_numpy(_WORKER_M_REG[start:end])
    ev = torch.linalg.eigvalsh(M_chunk)
    s2_desc = (ev.flip(-1) - _WORKER_RIDGE).clamp_min(0)
    s_max2 = s2_desc[:, 0:1].clamp_min(1e-20)
    return (s2_desc > (_WORKER_TAU * _WORKER_TAU) * s_max2).sum(dim=-1).numpy().astype(np.float64)


def _spectral_worker_chunk_serial(m_reg: np.ndarray, ridge: float, tau: float) -> np.ndarray:
    """Chemin séquentiel (utilisé quand le batch est trop petit pour amortir le pool)."""
    M = torch.from_numpy(m_reg)
    ev = torch.linalg.eigvalsh(M)
    s2_desc = (ev.flip(-1) - ridge).clamp_min(0)
    s_max2 = s2_desc[:, 0:1].clamp_min(1e-20)
    return (s2_desc > (tau * tau) * s_max2).sum(dim=-1).numpy().astype(np.float64)


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

    **IMPORTANT** : Pour éviter deadlock BLAS multi-thread, requiert OPENBLAS_NUM_THREADS=1
    ou MKL_NUM_THREADS=1. Cf. DOC/carnet 2026-05-11 bug deadlock eigvalsh.
    """
    # Garde-fou : refuser de tourner si BLAS multi-thread (deadlock 38h reproductible
    # sur eigvalsh batché grandes matrices rank-deficient — cf. carnet 2026-05-11).
    _required_blas_vars = ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    _bad = {v: os.environ.get(v) for v in _required_blas_vars if os.environ.get(v) != '1'}
    if _bad:
        raise RuntimeError(
            f"BLAS multi-thread détecté ({_bad}). compute_s_spectral() refuse de tourner "
            f"pour éviter le deadlock eigvalsh. Lance via OPS/setup/launch_phase1b.sh ou exporte "
            f"manuellement {'='.join(_required_blas_vars)}=1 avant l'import de torch. "
            f"Voir DOC/carnet 2026-05-11."
        )

    L = len(attn_per_layer)
    B, H, N, _ = attn_per_layer[0].shape
    device = attn_per_layer[0].device

    out = torch.zeros(L, B, H, N, dtype=torch.float64, device=device)

    if N < K:
        return out

    ts = list(range(K - 1, N, stride))
    n_full = len(ts)
    if n_full == 0:
        return out

    # sched_getaffinity (vs cpu_count) respecte les cgroups : sur RunPod le
    # container peut voir cpu_count=48 mais n'avoir que 32 cores assignés.
    n_workers = max(1, len(os.sched_getaffinity(0)) - 1)
    ridge = 1e-6
    ts_tensor = torch.tensor(ts, device=device)
    eye = torch.eye(K, dtype=torch.float32).unsqueeze(0)

    # Phase 1 : compute M_reg pour tous les layers et accumuler (FP32).
    # Coût mémoire : L * B * H * n_full * K * K * 4 bytes.
    # Ex : 6 * 2 * 8 * 937 * 64² * 4 ≈ 1.5 GB pour un batch typique phase 1.5.
    M_reg_per_layer: list[np.ndarray] = []
    for A_layer in attn_per_layer:
        windows = torch.stack(
            [A_layer[:, :, t - K + 1 : t + 1, t - K + 1 : t + 1] for t in ts],
            dim=2,
        )  # (B, H, n_full, K, K)
        W = windows.reshape(-1, K, K).contiguous().float()
        M = W @ W.transpose(-2, -1)
        M_reg = (M + ridge * eye).contiguous()
        M_reg_per_layer.append(M_reg.numpy())

    # Phase 2 : un seul pool.map sur le batch concaténé (évite L forks).
    all_M_reg = np.concatenate(M_reg_per_layer, axis=0)
    M_count = all_M_reg.shape[0]
    chunk_size = max(1, (M_count + n_workers - 1) // n_workers)
    chunks = [(i, min(i + chunk_size, M_count)) for i in range(0, M_count, chunk_size)]

    if len(chunks) <= 1:
        r_eff_all = _spectral_worker_chunk_serial(all_M_reg, ridge, tau)
    else:
        ctx = get_context("fork")
        with ctx.Pool(n_workers, initializer=_spectral_worker_init,
                      initargs=(all_M_reg, ridge, tau)) as pool:
            results = pool.map(_spectral_worker_chunk, chunks)
        r_eff_all = np.concatenate(results)

    # Phase 3 : reshape par layer et place dans out aux positions ts.
    r_eff_all = r_eff_all.reshape(L, B, H, n_full)
    for ell in range(L):
        out[ell, :, :, ts_tensor] = torch.from_numpy(r_eff_all[ell]).to(
            device=device, dtype=torch.float64
        )

    return out
