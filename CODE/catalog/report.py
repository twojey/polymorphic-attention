"""
report.py — Génération de rapport Markdown à partir d'un results.json catalog.

Spec : DOC/CATALOGUE §5 "Format du rapport batterie".

Génère :
- Tableau global Property × Régime
- Signatures par Oracle (récap stats clés)
- Décou vertes inattendues (Properties avec variance cross-régime > seuil)

Usage :
    python -m catalog.report --results results.json --output report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def render_markdown_report(results_data: dict[str, Any]) -> str:
    """Construit le rapport Markdown à partir des résultats Battery."""
    meta = results_data.get("metadata", {})
    per_regime = results_data.get("per_regime", {})
    cross_regime = results_data.get("cross_regime", {})

    lines: list[str] = []
    lines.append(f"# Rapport batterie catalog")
    lines.append("")
    lines.append(f"- **Oracle** : {meta.get('oracle_id', 'unknown')}")
    lines.append(f"- **Battery** : {meta.get('battery_name', 'unknown')}")
    lines.append(f"- **N régimes** : {meta.get('n_regimes', 0)}")
    lines.append(f"- **N examples/régime** : {meta.get('n_examples_per_regime', 0)}")
    lines.append(f"- **Device** : {meta.get('device', 'unknown')}")
    lines.append(f"- **Properties activées** : {len(meta.get('properties', []))}")
    lines.append("")

    lines.append("## Tableau Property × Régime (stats résumées)")
    lines.append("")
    if per_regime:
        regimes = sorted(per_regime.keys(), key=str)
        properties = set()
        for regime_key in regimes:
            properties.update(per_regime[regime_key].keys())
        properties = sorted(properties)

        # Header
        header = "| Property | " + " | ".join(str(r) for r in regimes) + " |"
        sep = "|" + "---|" * (len(regimes) + 1)
        lines.append(header)
        lines.append(sep)

        for prop in properties:
            row = [prop]
            for regime in regimes:
                prop_out = per_regime[regime].get(prop, {})
                # Cherche une clé contenant "median" ou première num
                summary = "—"
                for k, v in prop_out.items():
                    if "median" in k.lower() and isinstance(v, (int, float)):
                        summary = f"{v:.3g}"
                        break
                if summary == "—":
                    for k, v in prop_out.items():
                        if isinstance(v, (int, float)):
                            summary = f"{v:.3g}"
                            break
                row.append(summary)
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Cross-régime Properties")
    lines.append("")
    if cross_regime:
        for prop_name, prop_out in cross_regime.items():
            lines.append(f"### {prop_name}")
            for k, v in prop_out.items():
                if isinstance(v, (int, float)):
                    lines.append(f"- `{k}` : {v:.4g}")
                else:
                    lines.append(f"- `{k}` : {v}")
            lines.append("")
    else:
        lines.append("_(aucune Property cross_regime activée)_")
        lines.append("")

    lines.append("## Découvertes (variances cross-régime notables)")
    lines.append("")
    if per_regime:
        # Pour chaque (Property, métrique), variance sur les régimes
        regimes = list(per_regime.keys())
        variances: list[tuple[str, str, float]] = []
        all_keys: set[tuple[str, str]] = set()
        for r in regimes:
            for prop, prop_out in per_regime[r].items():
                for metric in prop_out.keys():
                    all_keys.add((prop, metric))
        for prop, metric in all_keys:
            vals: list[float] = []
            for r in regimes:
                v = per_regime[r].get(prop, {}).get(metric)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            if len(vals) >= 2:
                mean = sum(vals) / len(vals)
                var = sum((x - mean) ** 2 for x in vals) / len(vals)
                std = var ** 0.5
                if abs(mean) > 1e-30 and std / max(abs(mean), 1e-30) > 0.30:
                    variances.append((prop, metric, std / max(abs(mean), 1e-30)))
        variances.sort(key=lambda x: x[2], reverse=True)
        if variances:
            lines.append("Top 10 métriques avec CV > 30 % cross-régime :")
            lines.append("")
            lines.append("| Property | Metric | CV |")
            lines.append("|---|---|---|")
            for prop, metric, cv in variances[:10]:
                lines.append(f"| {prop} | `{metric}` | {cv:.2f} |")
        else:
            lines.append("_(rien au-dessus de CV=0.30)_")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="catalog.report",
        description="Génère un rapport Markdown à partir d'un results.json catalog.",
    )
    parser.add_argument("--results", type=str, required=True,
                        help="Path results.json produit par catalog.run")
    parser.add_argument("--output", type=str, required=True,
                        help="Path du rapport Markdown à écrire")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.is_file():
        print(f"[catalog.report] results.json introuvable : {results_path}", file=sys.stderr)
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)

    markdown = render_markdown_report(data)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(markdown)
    print(f"=== Rapport écrit : {out_path} ===", flush=True)


if __name__ == "__main__":
    main()
