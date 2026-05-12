"""
paper_figures.py — Génération figures matplotlib pour paper Partie 1 + Partie 2.

Convention :
- Format : PDF + PNG, DPI=300 pour publication
- Tailles : single column (3.4"), double column (7"), spread (full page)
- Style : seaborn-paper (style scientifique), serif fonts

API :
- `generate_figure_signatures(table)` : heatmap Property × Oracle
- `generate_figure_predictions_vs_measured(detailed)` : confrontation paris
- `generate_figure_pareto_curves(...)` : Pareto qualité vs budget
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")  # backend non-interactif
    import matplotlib.pyplot as plt
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    return plt


def generate_figure_signatures(
    table: dict[str, dict[str, float]],
    output_path: str | Path,
    *,
    max_properties: int = 30,
    title: str = "Signatures cross-Oracle",
) -> Path:
    """Heatmap Property × Oracle des signatures."""
    plt = _setup_matplotlib()
    import numpy as np

    oracle_ids = sorted({o for oracles in table.values() for o in oracles.keys()})
    # Top max_properties par variance
    import statistics
    variances = []
    for key, oracles in table.items():
        vals = list(oracles.values())
        if len(vals) >= 2:
            variances.append((key, statistics.variance(vals)))
    variances.sort(key=lambda x: x[1], reverse=True)
    top_keys = [k for k, _ in variances[:max_properties]]
    if not top_keys:
        top_keys = list(table.keys())[:max_properties]

    matrix = np.zeros((len(top_keys), len(oracle_ids)))
    for i, key in enumerate(top_keys):
        for j, oid in enumerate(oracle_ids):
            matrix[i, j] = table[key].get(oid, np.nan)

    # Normalisation par ligne pour comparaison visuelle
    row_max = np.nanmax(np.abs(matrix), axis=1, keepdims=True)
    row_max = np.where(row_max < 1e-30, 1.0, row_max)
    matrix_norm = matrix / row_max

    fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(oracle_ids)),
                                     max(4, 0.25 * len(top_keys))))
    im = ax.imshow(matrix_norm, aspect="auto", cmap="RdBu_r",
                   vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(oracle_ids)))
    ax.set_xticklabels(oracle_ids, rotation=45, ha="right")
    ax.set_yticks(range(len(top_keys)))
    ax.set_yticklabels(top_keys, fontsize=7)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="value (row-normalized)")
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def generate_figure_predictions_vs_measured(
    detailed: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Bar chart confronté : prédiction (couleur) vs validation."""
    plt = _setup_matplotlib()
    import numpy as np

    n = len(detailed)
    if n == 0:
        raise ValueError("detailed vide")
    valid = [d for d in detailed if d.get("validated") is True]
    refut = [d for d in detailed if d.get("validated") is False]
    unkn = [d for d in detailed if d.get("validated") is None]
    counts = [len(valid), len(refut), len(unkn)]
    labels = ["Validés", "Réfutés", "Non évaluables"]
    colors = ["#2ca02c", "#d62728", "#7f7f7f"]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(labels, counts, color=colors)
    for i, c in enumerate(counts):
        ax.text(i, c + 0.1, str(c), ha="center", fontsize=10)
    ax.set_ylabel(f"Nombre de paris (total = {n})")
    ax.set_title("Confrontation paris a priori vs mesures")
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def generate_figure_pareto_curves(
    points_per_method: dict[str, list[tuple[float, float]]],
    output_path: str | Path,
    *,
    xlabel: str = "Budget compute (FLOPs ou wall-time, log scale)",
    ylabel: str = "Qualité (val accuracy)",
    title: str = "Pareto qualité vs budget",
) -> Path:
    """Tracé Pareto (méthode comme couleur, points connectés par budget)."""
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(5, 4))
    for method, pts in points_per_method.items():
        if not pts:
            continue
        xs, ys = zip(*sorted(pts))
        ax.plot(xs, ys, marker="o", label=method)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(prog="livrables.paper_figures")
    p.add_argument("--mode", required=True,
                  choices=["signatures", "predictions", "pareto"])
    p.add_argument("--input", required=True, help="JSON input (table | detailed | pareto)")
    p.add_argument("--output", required=True, help="path figure (PDF / PNG)")
    args = p.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    if args.mode == "signatures":
        generate_figure_signatures(data, args.output)
    elif args.mode == "predictions":
        generate_figure_predictions_vs_measured(data, args.output)
    else:
        # Format attendu : {method: [[x, y], [x, y], ...]}
        generate_figure_pareto_curves(
            {k: [tuple(p) for p in v] for k, v in data.items()},
            args.output,
        )
    print(f"=== Figure écrite : {args.output} ===", flush=True)


if __name__ == "__main__":
    main()
