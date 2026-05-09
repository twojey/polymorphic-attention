"""
report.py — génération automatique du rapport phase 1.

Consomme :
- les statistiques agrégées par régime (rang Hankel, entropie spectrale)
- la précision de l'Oracle par régime (val_acc)

Produit :
- figures (courbes monovariées, heatmaps 2D)
- recommandation R_max préliminaire
- rapport Markdown (DOC/reports/phase1_report.md)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shared.aggregation import RegimeStats


@dataclass
class Phase1Verdict:
    """Verdict go/no-go phase 1 selon DOC/01 §6 + thresholds_phase1.yaml."""
    portion_compressible: float    # fraction des régimes où compressible
    threshold_min_portion: float
    acc_floor_respected: bool
    decision: str                   # "go" / "no-go"
    reason: str


def recommend_r_max(
    rank_stats_per_regime: dict[tuple[str, float], RegimeStats],
    *,
    seq_len: int,
    margin: float = 1.5,
) -> int:
    """Recommandation préliminaire de R_max pour phase 3.

    Heuristique : R_max = ceil(margin × p90 du rang Hankel sur tous les
    régimes). La marge × 1.5 fournit du headroom pour les régimes plus durs
    qui pourraient apparaître phase 2.
    """
    p90s = [s.p90 for s in rank_stats_per_regime.values() if not math.isnan(s.p90)]
    if not p90s:
        return seq_len  # pas d'info → safe default
    r_max = int(math.ceil(margin * max(p90s)))
    return max(1, min(r_max, seq_len))


def evaluate_go_no_go(
    *,
    rank_stats_per_regime: dict[tuple[str, float], RegimeStats],
    entropy_stats_per_regime: dict[tuple[str, float], RegimeStats],
    accuracy_per_regime: dict[tuple[str, float], float],
    seq_len: int,
    max_rank_ratio: float,
    max_entropy_ratio: float,
    acc_floor: float,
    min_portion: float,
) -> Phase1Verdict:
    """DOC/01 §6 : GO si portion ≥ min_portion compressible avec acc ≥ acc_floor."""
    log_n = math.log(max(seq_len, 2))
    rank_threshold = seq_len * max_rank_ratio
    entropy_threshold = log_n * max_entropy_ratio

    n_total = 0
    n_compressible = 0
    acc_violations = 0
    for key in rank_stats_per_regime:
        n_total += 1
        rank_med = rank_stats_per_regime[key].median
        ent_med = entropy_stats_per_regime.get(key, rank_stats_per_regime[key]).median
        acc = accuracy_per_regime.get(key, 0.0)
        compressible = (rank_med < rank_threshold) or (ent_med < entropy_threshold)
        if compressible and acc >= acc_floor:
            n_compressible += 1
        if acc < acc_floor:
            acc_violations += 1

    portion = n_compressible / max(n_total, 1)
    decision = "go" if portion >= min_portion else "no-go"
    reason = (
        f"{n_compressible}/{n_total} régimes compressibles "
        f"(rang < {rank_threshold:.1f} OU H < {entropy_threshold:.2f}) "
        f"AVEC val_acc ≥ {acc_floor}. "
        f"Portion = {portion:.2%}, seuil min_portion = {min_portion:.2%}."
    )
    if acc_violations > 0:
        reason += f" {acc_violations} régime(s) avec acc < {acc_floor} ignorés."
    return Phase1Verdict(
        portion_compressible=portion,
        threshold_min_portion=min_portion,
        acc_floor_respected=acc_violations == 0,
        decision=decision,
        reason=reason,
    )


def render_markdown_report(
    *,
    run_id: str,
    git_hash: str,
    domain: str,
    rank_stats: dict[tuple[str, float], RegimeStats],
    entropy_stats: dict[tuple[str, float], RegimeStats],
    accuracy: dict[tuple[str, float], float],
    verdict: Phase1Verdict,
    r_max: int,
    figures: list[Path],
) -> str:
    """Génère le contenu du rapport phase 1 en Markdown."""
    lines: list[str] = []
    lines.append(f"# Rapport phase 1 — {domain} — run {run_id}")
    lines.append("")
    lines.append(f"- **Run ID** : `{run_id}`")
    lines.append(f"- **Git hash** : `{git_hash}`")
    lines.append(f"- **Domaine** : `{domain}`")
    lines.append("")

    lines.append("## Verdict go/no-go")
    lines.append("")
    lines.append(f"**Décision** : `{verdict.decision.upper()}`")
    lines.append("")
    lines.append(verdict.reason)
    lines.append("")

    lines.append("## Recommandation R_max préliminaire")
    lines.append("")
    lines.append(f"`R_max ≈ {r_max}` (heuristique : 1.5 × max(p90 rang Hankel)).")
    lines.append("À raffiner par phase 2 sur la base du r_eff mesuré.")
    lines.append("")

    lines.append("## Statistiques par régime")
    lines.append("")
    lines.append("### Rang de Hankel")
    lines.append("")
    lines.append("| (axe, valeur) | n | médiane | IQR | p10 | p90 |")
    lines.append("|---|---|---|---|---|---|")
    for (axis, val), s in sorted(rank_stats.items()):
        lines.append(f"| ({axis}, {val:g}) | {s.n} | {s.median:.2f} | {s.iqr:.2f} | {s.p10:.2f} | {s.p90:.2f} |")
    lines.append("")

    lines.append("### Entropie spectrale")
    lines.append("")
    lines.append("| (axe, valeur) | n | médiane | IQR | p10 | p90 |")
    lines.append("|---|---|---|---|---|---|")
    for (axis, val), s in sorted(entropy_stats.items()):
        lines.append(f"| ({axis}, {val:g}) | {s.n} | {s.median:.3f} | {s.iqr:.3f} | {s.p10:.3f} | {s.p90:.3f} |")
    lines.append("")

    lines.append("### Précision Oracle")
    lines.append("")
    lines.append("| (axe, valeur) | val_acc |")
    lines.append("|---|---|")
    for (axis, val), acc in sorted(accuracy.items()):
        lines.append(f"| ({axis}, {val:g}) | {acc:.3f} |")
    lines.append("")

    if figures:
        lines.append("## Figures")
        lines.append("")
        for f in figures:
            lines.append(f"- `{f}`")
        lines.append("")

    return "\n".join(lines)


def save_report(content: str, repo_root: Path, run_id: str) -> Path:
    out = repo_root / "DOC" / "reports" / f"phase1_{run_id}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content)
    return out
