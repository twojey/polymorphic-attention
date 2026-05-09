"""
plotting.py — helpers matplotlib pour figures du protocole.

Toutes les fonctions retournent une matplotlib.Figure pour permettre :
- log MLflow direct via mlflow.log_figure()
- sauvegarde fichier via fig.savefig()
- rendu inline en notebook si besoin

Pas de style ni couleur fixée ici — laisse la décision au call site.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend non-interactif (pas de Tk requis sur pod)
import matplotlib.pyplot as plt
import numpy as np


def monovariate_curve(
    *,
    x: np.ndarray,
    y_mean: np.ndarray,
    y_p10: np.ndarray | None = None,
    y_p90: np.ndarray | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> plt.Figure:
    """Courbe monovariée avec bande p10/p90 optionnelle."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    if y_p10 is not None and y_p90 is not None:
        ax.fill_between(x, y_p10, y_p90, alpha=0.2, label="p10/p90")
    ax.plot(x, y_mean, marker="o", label="mean")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def heatmap_2d(
    *,
    matrix: np.ndarray,
    xticks: list[float],
    yticks: list[float],
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    cmap: str = "viridis",
    cbar_label: str = "",
) -> plt.Figure:
    """Heatmap 2D pour balayages croisés (ω × Δ, etc.)."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap)
    ax.set_xticks(range(len(xticks)))
    ax.set_xticklabels([f"{x:g}" for x in xticks])
    ax.set_yticks(range(len(yticks)))
    ax.set_yticklabels([f"{y:g}" for y in yticks])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    return fig


def correlation_matrix_heatmap(
    *,
    rho_matrix: np.ndarray,
    signal_names: list[str],
    axis_names: list[str],
    title: str = "",
) -> plt.Figure:
    """Matrice de corrélation Spearman (signal × axe) pour phase 1.5."""
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    im = ax.imshow(rho_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(axis_names)))
    ax.set_xticklabels(axis_names)
    ax.set_yticks(range(len(signal_names)))
    ax.set_yticklabels(signal_names)
    for i in range(rho_matrix.shape[0]):
        for j in range(rho_matrix.shape[1]):
            ax.text(j, i, f"{rho_matrix[i, j]:.2f}", ha="center", va="center", color="black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Spearman ρ")
    fig.tight_layout()
    return fig


def stress_rank_map(
    *,
    omega_values: list[float],
    delta_values: list[float],
    r_eff_median: np.ndarray,  # (len(omega), len(delta))
    r_eff_iqr: np.ndarray | None = None,
    title: str = "Stress-Rank Map",
) -> plt.Figure:
    """Stress-Rank Map : médiane de r_eff par (ω, Δ), IQR optionnellement annoté."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    im = ax.imshow(r_eff_median, aspect="auto", origin="lower", cmap="plasma")
    ax.set_xticks(range(len(delta_values)))
    ax.set_xticklabels([f"{d:g}" for d in delta_values])
    ax.set_yticks(range(len(omega_values)))
    ax.set_yticklabels([f"{w:g}" for w in omega_values])
    ax.set_xlabel("Δ (distance)")
    ax.set_ylabel("ω (récursion)")
    ax.set_title(title)
    if r_eff_iqr is not None:
        for i in range(r_eff_median.shape[0]):
            for j in range(r_eff_median.shape[1]):
                ax.text(j, i, f"±{r_eff_iqr[i, j]:.1f}", ha="center", va="center",
                        color="white", fontsize=7)
    fig.colorbar(im, ax=ax, label="r_eff médian")
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p
