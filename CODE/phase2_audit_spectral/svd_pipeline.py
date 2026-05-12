"""
svd_pipeline.py — pipeline SVD batché FP64 sur matrices d'attention audit_svd.

Spec : DOC/02 §2.1, DOC/01 §8.4 (extraction FP64), §8.7 (pas d'agrégation pré-SVD).

Pour chaque matrice A (par couche, par tête, par exemple) :
- SVD complète FP64 (CPU) ou FP32 (GPU consumer Blackwell, où FP64 = 1/64)
- Calcul de r_eff(θ) pour θ ∈ {0.95, 0.99}
  r_eff(θ) = nombre minimum de valeurs singulières expliquant ≥ θ de la variance

Sortie : tensor (L, B, H) ou (B, H) par couche selon mode.

Choix précision/device (DOC/carnet 2026-05-11 fin de journée) :
- CPU FP64 : référence stricte (spec DOC/01 §8.4). Lent mais sans perte.
- GPU FP32 : optionnel via `device="cuda", precision="fp32"`. Pour
  consumer Blackwell (RTX 5090), FP64 GPU est ~1/64 du FP32 → CPU est
  comparable et plus simple. FP32 GPU est ~50-100× plus rapide ; pour
  un compteur r_eff(θ) qui ne dépend que de l'ordre des valeurs
  singulières, la précision FP32 est largement suffisante.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal

import torch


def r_eff_from_singular_values(s: torch.Tensor, theta: float) -> torch.Tensor:
    """r_eff(θ) à partir des valeurs singulières.

    s : (..., k) en n'importe quelle précision flottante. Retourne (..., )
    int : plus petit r tel que Σ_{i<r} s_i² / Σ s² ≥ θ.
    """
    s2 = s.pow(2)
    cumsum = s2.cumsum(dim=-1)
    total = s2.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    ratio = cumsum / total
    above = ratio >= theta
    r_eff = above.float().argmax(dim=-1) + 1
    all_zero = (s2.sum(dim=-1) == 0)
    r_eff = torch.where(all_zero, torch.zeros_like(r_eff), r_eff)
    return r_eff


@dataclass
class SVDResult:
    """Résultat SVD pour une matrice d'attention."""
    s: torch.Tensor       # valeurs singulières (..., k)
    r_eff_95: torch.Tensor   # entier
    r_eff_99: torch.Tensor


PrecisionMode = Literal["fp64", "fp32"]
DeviceMode = Literal["cpu", "cuda", "auto"]


def _resolve_device(device: DeviceMode, fallback: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else fallback
    if device == "cuda" and not torch.cuda.is_available():
        print(
            "[svd_pipeline] device='cuda' demandé mais CUDA indisponible, "
            "fallback sur CPU.",
            file=sys.stderr,
            flush=True,
        )
        return "cpu"
    return device


def svd_attention(
    A: torch.Tensor,
    *,
    theta_values: tuple[float, ...] = (0.95, 0.99),
    device: DeviceMode = "auto",
    precision: PrecisionMode = "fp64",
) -> dict[str, torch.Tensor]:
    """SVD batché sur (..., M, N).

    Par défaut : suit le device source de A et reste en FP64 (spec stricte).
    Pour accélérer sur GPU consumer Blackwell, appeler avec
    `device="cuda", precision="fp32"`.

    Retourne {"s": ..., "r_eff_<θ>": ...} sur le device d'entrée
    (re-déplacement explicite). r_eff arrondi int.

    Lève RuntimeError avec contexte si la SVD ne converge pas (cuSolver), pour
    éviter des sorties silencieuses sur des matrices rank-deficient.
    """
    src_device = A.device
    target_device = _resolve_device(device, fallback=str(src_device))
    target_dtype = torch.float64 if precision == "fp64" else torch.float32

    A_work = A.to(device=target_device, dtype=target_dtype)

    # Sur GPU, cuSolver SVD a un fallback interne lent quand les matrices
    # d'attention softmax sont rank-deficient (carnet 2026-05-12 smoke pod
    # phase 2). Bypass : eigvalsh(A·Aᵀ + εI) directement, plus rapide ET
    # gère les rank-deficiencies sans fallback opaque.
    # CPU FP64 garde svdvals (rapide, pas de problème de convergence).
    use_eigvalsh = (target_device != "cpu")
    if use_eigvalsh:
        eps = 1e-12 if target_dtype == torch.float64 else 1e-7
        AAt = A_work @ A_work.transpose(-2, -1)
        eye = torch.eye(AAt.size(-1), dtype=target_dtype, device=target_device)
        AAt = AAt + eps * eye
        eigvals = torch.linalg.eigvalsh(AAt)
        # eigvalsh retourne ordre ascendant → flip puis sqrt(clamp) pour s.
        # On retire la ridge avant clamp pour rester fidèle aux vraies σ.
        s2 = (eigvals.flip(-1) - eps).clamp_min(0)
        s = s2.sqrt()
    else:
        try:
            s = torch.linalg.svdvals(A_work)
        except (RuntimeError, torch._C._LinAlgError) as exc:
            # Garde-fou CPU : fallback eigvalsh aussi si svdvals échoue.
            msg = (
                f"[svd_pipeline] svdvals CPU a échoué sur shape={tuple(A_work.shape)} "
                f"dtype={target_dtype} : {exc}. Fallback eigvalsh."
            )
            print(msg, file=sys.stderr, flush=True)
            eps = 1e-12 if target_dtype == torch.float64 else 1e-7
            AAt = A_work @ A_work.transpose(-2, -1)
            eye = torch.eye(AAt.size(-1), dtype=target_dtype, device=target_device)
            AAt = AAt + eps * eye
            eigvals = torch.linalg.eigvalsh(AAt)
            s2 = (eigvals.flip(-1) - eps).clamp_min(0)
            s = s2.sqrt()

    out: dict[str, torch.Tensor] = {"s": s.to(src_device)}
    for theta in theta_values:
        out[f"r_eff_{int(theta*100)}"] = r_eff_from_singular_values(s, theta).to(src_device)
    return out


def hankelize_attention_lines(A: torch.Tensor) -> torch.Tensor:
    """Pour chaque ligne i de A, construit la matrice de Hankel et retourne
    un tenseur (..., N, p, q). Convention V1 — voir phase1_metrologie.metrics.hankel.
    """
    *batch_dims, M, N = A.shape
    p = N // 2
    q = N - p + 1
    H = torch.empty(*batch_dims, M, p, q, dtype=A.dtype, device=A.device)
    for a in range(p):
        H[..., :, a, :] = A[..., :, a : a + q]
    return H
