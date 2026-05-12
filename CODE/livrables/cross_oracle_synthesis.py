"""
cross_oracle_synthesis.py — Synthèse cross-Oracle des signatures.

Spec : DOC/CATALOGUE §5.1 "Tableau global propriété × Oracle".

Pour chaque Property X, agrège la valeur cross-régime (médiane par Oracle)
et produit une table Markdown + JSON.

API :
- `build_signatures_table(results_per_oracle)` : dict[oracle_id → results_dict]
  → DataFrame-like + Markdown table
- `compute_signature_variance(results_per_oracle)` : variance d'une Property
  cross-Oracle (= discriminance)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _extract_property_values(
    results_dict: dict[str, Any], property_name: str
) -> dict[str, list[float]]:
    """Extrait toutes les valeurs scalaires d'une Property depuis un results.json.

    Returns :
        dict[metric_name → list_of_values_across_regimes]
    """
    out: dict[str, list[float]] = {}
    for regime_key, regime_out in results_dict.get("per_regime", {}).items():
        prop_out = regime_out.get(property_name, {})
        if not isinstance(prop_out, dict):
            continue
        for k, v in prop_out.items():
            if isinstance(v, (int, float)):
                out.setdefault(k, []).append(float(v))
    return out


def build_signatures_table(
    results_per_oracle: dict[str, dict[str, Any]],
    *,
    aggregation: str = "median",
) -> tuple[dict[str, dict[str, float]], str]:
    """Construit la table cross-Oracle.

    Args
    ----
    results_per_oracle : {oracle_id → results.json dict}
    aggregation : "median" | "mean" | "max" | "min"

    Returns
    -------
    table : {property_metric → {oracle_id → value}}
    markdown : str — table Markdown formatée pour rapport
    """
    if aggregation not in ("median", "mean", "max", "min"):
        raise ValueError(f"aggregation invalide : {aggregation}")
    import statistics

    table: dict[str, dict[str, float]] = {}
    all_properties: set[str] = set()
    for results_dict in results_per_oracle.values():
        for regime_out in results_dict.get("per_regime", {}).values():
            all_properties.update(regime_out.keys())

    for prop_name in sorted(all_properties):
        for oracle_id, results_dict in results_per_oracle.items():
            values_by_metric = _extract_property_values(results_dict, prop_name)
            for metric, values in values_by_metric.items():
                key = f"{prop_name}.{metric}"
                if not values:
                    continue
                if aggregation == "median":
                    agg = statistics.median(values)
                elif aggregation == "mean":
                    agg = statistics.mean(values)
                elif aggregation == "max":
                    agg = max(values)
                else:
                    agg = min(values)
                table.setdefault(key, {})[oracle_id] = agg

    # Markdown
    oracle_ids = sorted(results_per_oracle.keys())
    lines = [
        "## Table cross-Oracle des signatures",
        "",
        f"Agrégation `{aggregation}` cross-régime, une colonne par Oracle.",
        "",
        "| Property.metric | " + " | ".join(oracle_ids) + " | Range |",
        "|" + "---|" * (len(oracle_ids) + 2),
    ]
    for key in sorted(table.keys()):
        row_vals = []
        all_vals: list[float] = []
        for oid in oracle_ids:
            v = table[key].get(oid)
            if v is None:
                row_vals.append("—")
            else:
                row_vals.append(f"{v:.3g}")
                all_vals.append(v)
        if len(all_vals) >= 2:
            rng = max(all_vals) - min(all_vals)
            range_str = f"{rng:.3g}"
        else:
            range_str = "—"
        lines.append(f"| {key} | " + " | ".join(row_vals) + f" | {range_str} |")
    return table, "\n".join(lines)


def compute_signature_variance(
    results_per_oracle: dict[str, dict[str, Any]],
) -> list[tuple[str, float]]:
    """Pour chaque Property × metric, calcule la variance cross-Oracle.

    Returns
    -------
    Liste triée par variance décroissante : [(prop.metric, variance), ...]
    """
    import statistics
    table, _ = build_signatures_table(results_per_oracle, aggregation="median")
    out: list[tuple[str, float]] = []
    for key, oracles in table.items():
        vals = list(oracles.values())
        if len(vals) >= 2:
            v = statistics.variance(vals)
            out.append((key, v))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def main() -> None:
    """CLI : python -m livrables.cross_oracle_synthesis [...]."""
    import argparse
    p = argparse.ArgumentParser(prog="livrables.cross_oracle_synthesis")
    p.add_argument("--results", nargs="+", required=True,
                  help="path1.json:oracle_id1 path2.json:oracle_id2 ...")
    p.add_argument("--output", required=True, help="output_dir")
    p.add_argument("--aggregation", default="median",
                  choices=["median", "mean", "max", "min"])
    args = p.parse_args()

    results_per_oracle: dict[str, dict] = {}
    for spec in args.results:
        if ":" not in spec:
            raise SystemExit(f"Spec doit être 'path.json:oracle_id', reçu : {spec}")
        path, oid = spec.rsplit(":", 1)
        with open(path) as f:
            results_per_oracle[oid] = json.load(f)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    table, md = build_signatures_table(results_per_oracle, aggregation=args.aggregation)
    (out_dir / "signatures_table.json").write_text(json.dumps(table, indent=2))
    (out_dir / "signatures_table.md").write_text(md)

    variances = compute_signature_variance(results_per_oracle)
    var_md = "## Top 30 Properties × metric par variance cross-Oracle\n\n"
    var_md += "| Property.metric | Variance |\n|---|---|\n"
    for key, v in variances[:30]:
        var_md += f"| {key} | {v:.3g} |\n"
    (out_dir / "signatures_variance.md").write_text(var_md)
    print(f"=== Synthesis écrite : {out_dir}/ ===", flush=True)


if __name__ == "__main__":
    main()
