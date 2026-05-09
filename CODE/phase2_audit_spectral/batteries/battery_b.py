"""
battery_b.py — Batterie B : analyse du résidu après best-fit.

Spec : DOC/02 §5c batterie B.

- B.1 : norme du résidu par régime (déjà calculée par battery_a)
- B.2 : SVD du résidu — low-rank caché ?
- B.3 : FFT du résidu — fréquences structurées ?
- B.4 : PCA cross-régimes du résidu — classe émergente ?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class BatteryBResult:
    norm_residual: float
    svd_top_k_ratio: float       # σ_top / σ_total ; haute valeur → low-rank caché
    fft_top_freq_ratio: float    # |F_top| / |F_mean| ; haute valeur → fréquence structurée
    spectrum_top: np.ndarray     # top valeurs singulières du résidu


def analyze_residual_svd(residual: torch.Tensor, *, top_k: int = 8) -> tuple[float, np.ndarray]:
    """SVD du résidu. Retourne (ratio top-k, top-k singular values)."""
    s = torch.linalg.svdvals(residual.to(torch.float64))
    s_np = s.cpu().numpy()
    if s_np.size == 0:
        return 0.0, np.zeros(0)
    total = float(s_np.sum())
    if total == 0:
        return 0.0, np.zeros(top_k)
    top = s_np[: min(top_k, s_np.size)]
    ratio = float(top.sum() / total)
    return ratio, top


def analyze_residual_fft(residual: torch.Tensor) -> float:
    """FFT 2D du résidu, retourne ratio |F_top| / |F_mean|.

    Détecte les fréquences spatiales structurées (motifs périodiques).
    """
    *batch, M, N = residual.shape
    flat = residual.reshape(-1, M, N).to(torch.float64)
    F = torch.fft.fft2(flat).abs()
    mean = F.mean()
    top = F.amax(dim=(-2, -1)).mean()
    return float((top / mean.clamp_min(1e-30)).item())


def residual_analysis(residual: torch.Tensor) -> BatteryBResult:
    norm = float(residual.norm().item())
    svd_ratio, top_s = analyze_residual_svd(residual)
    fft_ratio = analyze_residual_fft(residual)
    return BatteryBResult(
        norm_residual=norm,
        svd_top_k_ratio=svd_ratio,
        fft_top_freq_ratio=fft_ratio,
        spectrum_top=top_s,
    )


def pca_cross_regime_residuals(
    residuals: dict[tuple, torch.Tensor],
    *,
    n_components: int = 4,
) -> dict[str, np.ndarray]:
    """PCA sur les résidus aplatis cross-régimes — détecte une classe émergente.

    Retourne {"explained_var": (n_components,), "components": (n_components, M*N)}.
    """
    flat_residuals = []
    keys = []
    for k, r in residuals.items():
        flat_residuals.append(r.reshape(-1).cpu().numpy())
        keys.append(k)
    if not flat_residuals:
        return {"explained_var": np.zeros(0), "components": np.zeros((0, 0))}
    X = np.stack(flat_residuals, axis=0)
    X_centered = X - X.mean(axis=0, keepdims=True)
    # SVD en numpy
    U, s, Vh = np.linalg.svd(X_centered, full_matrices=False)
    n_eff = min(n_components, s.size)
    var_total = float((s**2).sum())
    explained = (s[:n_eff] ** 2) / var_total if var_total > 0 else np.zeros(n_eff)
    return {
        "explained_var": np.asarray(explained, dtype=np.float64),
        "components": Vh[:n_eff],
        "regime_keys": np.array(keys, dtype=object),
    }
