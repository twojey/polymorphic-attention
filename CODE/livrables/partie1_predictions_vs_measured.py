"""
partie1_predictions_vs_measured.py — confrontation paris a priori vs mesures.

Spec : DOC/CATALOGUE §4 "Prédictions a priori par Oracle" + §4.3
"Confrontation paris vs mesures".

Pour chaque pari pré-enregistré (DOC/CATALOGUE §4.2), on extrait la
valeur mesurée correspondante depuis results_per_oracle et on évalue
si le pari est validé.

Pari format YAML / dict :
    {prop_metric: "A1_r_eff_theta099.r_eff_median",
     oracle: "DV",
     prediction: "low" | "medium" | "high" | "<value>",
     threshold_low: 5.0,
     threshold_high: 20.0}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    """Charge un fichier YAML/JSON de prédictions."""
    p = Path(path)
    if p.suffix in (".yaml", ".yml"):
        import yaml
        with open(p) as f:
            return yaml.safe_load(f) or []
    return json.loads(p.read_text())


def evaluate_prediction(
    pred: dict[str, Any],
    results_per_oracle: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Évalue un pari unique vs mesure observée.

    Returns
    -------
    {"prop_metric", "oracle", "prediction", "measured", "validated", "note"}
    """
    prop_metric = pred["prop_metric"]
    oracle_id = pred["oracle"]
    if oracle_id not in results_per_oracle:
        return {**pred, "measured": None, "validated": None,
                "note": f"oracle {oracle_id} absent des results"}

    prop, metric = prop_metric.rsplit(".", 1)
    rd = results_per_oracle[oracle_id]
    values: list[float] = []
    for regime_out in rd.get("per_regime", {}).values():
        prop_out = regime_out.get(prop, {})
        v = prop_out.get(metric) if isinstance(prop_out, dict) else None
        if isinstance(v, (int, float)):
            values.append(float(v))
    if not values:
        return {**pred, "measured": None, "validated": None,
                "note": f"{prop_metric} introuvable pour {oracle_id}"}

    import statistics
    measured = statistics.median(values)
    expected = pred["prediction"]
    if isinstance(expected, (int, float)):
        tol = pred.get("tolerance", 0.10) * abs(float(expected))
        validated = abs(measured - float(expected)) <= tol
        note = f"|{measured:.3g} - {expected}| <= {tol:.3g} ? {validated}"
    elif isinstance(expected, str) and expected in ("low", "medium", "high"):
        lo = pred.get("threshold_low", 0.0)
        hi = pred.get("threshold_high", float("inf"))
        if expected == "low":
            validated = measured < lo
            note = f"{measured:.3g} < {lo} ? {validated}"
        elif expected == "high":
            validated = measured > hi
            note = f"{measured:.3g} > {hi} ? {validated}"
        else:
            validated = lo <= measured <= hi
            note = f"{lo} <= {measured:.3g} <= {hi} ? {validated}"
    else:
        validated = None
        note = f"prediction format invalide : {expected}"

    return {**pred, "measured": measured, "validated": validated, "note": note}


def evaluate_all(
    predictions: list[dict[str, Any]],
    results_per_oracle: dict[str, dict[str, Any]],
) -> tuple[list[dict], dict[str, int]]:
    """Évalue tous les paris. Retourne (résultats détaillés, stats résumé)."""
    detailed = [evaluate_prediction(p, results_per_oracle) for p in predictions]
    summary = {
        "n_total": len(detailed),
        "n_validated": sum(1 for d in detailed if d.get("validated") is True),
        "n_refuted": sum(1 for d in detailed if d.get("validated") is False),
        "n_unevaluable": sum(1 for d in detailed if d.get("validated") is None),
    }
    return detailed, summary


def render_markdown(detailed: list[dict], summary: dict) -> str:
    lines = [
        "## Confrontation paris a priori vs mesures",
        "",
        f"**Total paris** : {summary['n_total']}",
        f"- ✅ Validés : {summary['n_validated']}",
        f"- ❌ Réfutés : {summary['n_refuted']}",
        f"- ❓ Non évaluables : {summary['n_unevaluable']}",
        "",
        "| Property.metric | Oracle | Prédiction | Mesuré | Statut | Note |",
        "|---|---|---|---|---|---|",
    ]
    for d in detailed:
        v = d.get("validated")
        status = "✅" if v is True else ("❌" if v is False else "❓")
        meas = d.get("measured")
        meas_str = f"{meas:.3g}" if isinstance(meas, (int, float)) else "—"
        lines.append(
            f"| {d['prop_metric']} | {d['oracle']} | {d['prediction']} | "
            f"{meas_str} | {status} | {d.get('note', '')} |"
        )
    return "\n".join(lines)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(prog="livrables.partie1_predictions_vs_measured")
    p.add_argument("--predictions", required=True, help="YAML/JSON prédictions")
    p.add_argument("--results", nargs="+", required=True,
                  help="path1.json:oracle_id1 ...")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    predictions = load_predictions(args.predictions)
    results_per_oracle: dict[str, dict] = {}
    for spec in args.results:
        path, oid = spec.rsplit(":", 1)
        with open(path) as f:
            results_per_oracle[oid] = json.load(f)

    detailed, summary = evaluate_all(predictions, results_per_oracle)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions_evaluation.json").write_text(
        json.dumps({"detailed": detailed, "summary": summary}, indent=2)
    )
    (out_dir / "predictions_evaluation.md").write_text(render_markdown(detailed, summary))
    print(f"=== Évaluation écrite : {out_dir}/ ===", flush=True)


if __name__ == "__main__":
    main()
