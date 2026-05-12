"""
run_all.py — Orchestrateur livrables Partie 1 en une commande.

Spec : DOC/paper/README.md §Génération automatique.

Prend en entrée :
- une liste de paires `<path/results.json>:<oracle_id>` (un par Oracle)
- optionnellement les prédictions YAML pré-enregistrées

Génère vers `--output` :
- signatures_table.{json,md}     (cross_oracle_synthesis)
- signatures_variance.md
- signatures_per_oracle.{json,md} (partie1_signatures)
- predictions_evaluation.{json,md} (partie1_predictions_vs_measured, si predictions fournies)
- figures/signatures.pdf + figures/predictions.pdf (paper_figures)

Robuste à l'absence partielle de résultats : skip propre + log warning.

Usage :
    python -m livrables.run_all \\
        --results OPS/logs/sprints/C/results.json:OR \\
                  OPS/logs/sprints/S5/results.json:DV \\
                  OPS/logs/sprints/S7/results.json:LL \\
        --predictions DOC/paper/partie1/predictions_a_priori.yaml \\
        --output DOC/paper/partie1/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from shared.logging_helpers import setup_logging, log_checkpoint


def _load_results(specs: list[str], logger: logging.Logger) -> dict[str, dict[str, Any]]:
    """Charge {oracle_id → results.json dict}. Skip silencieusement les absents."""
    out: dict[str, dict] = {}
    for spec in specs:
        if ":" not in spec:
            logger.error("Spec invalide (attendu 'path.json:oracle_id') : %s", spec)
            continue
        path_str, oid = spec.rsplit(":", 1)
        path = Path(path_str)
        if not path.is_file():
            logger.warning("Results file absent — skip oracle %s : %s", oid, path)
            continue
        try:
            with open(path) as f:
                out[oid] = json.load(f)
            logger.info("Loaded %s ← %s", oid, path)
        except Exception as e:
            logger.error("Failed to parse %s : %s", path, e)
    return out


def run_all(
    results_per_oracle: dict[str, dict[str, Any]],
    output_dir: Path,
    predictions_path: Path | None = None,
    n_max_for_signatures: int = 64,
    logger: logging.Logger | None = None,
) -> dict[str, Path]:
    """Génère tous les livrables Partie 1.

    Returns
    -------
    Dict {livrable_name → path produit}
    """
    log = logger or logging.getLogger("livrables.run_all")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}

    if not results_per_oracle:
        raise ValueError("Aucun results.json valide reçu. Abandon.")

    # --- 1. Cross-Oracle synthesis ---
    log_checkpoint(log, "step", name="cross_oracle_synthesis")
    from livrables.cross_oracle_synthesis import (
        build_signatures_table, compute_signature_variance,
    )
    table, table_md = build_signatures_table(results_per_oracle, aggregation="median")
    table_json_path = output_dir / "signatures_table.json"
    table_md_path = output_dir / "signatures_table.md"
    table_json_path.write_text(json.dumps(table, indent=2))
    table_md_path.write_text(table_md)
    artifacts["signatures_table_json"] = table_json_path
    artifacts["signatures_table_md"] = table_md_path
    log.info("[run_all] Table cross-Oracle écrite : %s", table_json_path)

    variances = compute_signature_variance(results_per_oracle)
    var_md = "## Top 30 Properties × metric par variance cross-Oracle\n\n"
    var_md += "| Property.metric | Variance |\n|---|---|\n"
    for key, v in variances[:30]:
        var_md += f"| {key} | {v:.3g} |\n"
    var_md_path = output_dir / "signatures_variance.md"
    var_md_path.write_text(var_md)
    artifacts["signatures_variance_md"] = var_md_path

    # --- 2. Signatures per Oracle (textuel) ---
    log_checkpoint(log, "step", name="signatures_per_oracle")
    from livrables.partie1_signatures import build_all_signatures
    sigs, sigs_md = build_all_signatures(results_per_oracle, N_max=n_max_for_signatures)
    sigs_json_path = output_dir / "signatures_per_oracle.json"
    sigs_md_path = output_dir / "signatures_per_oracle.md"
    sigs_json_path.write_text(json.dumps(sigs, indent=2))
    sigs_md_path.write_text(sigs_md)
    artifacts["signatures_per_oracle_json"] = sigs_json_path
    artifacts["signatures_per_oracle_md"] = sigs_md_path

    # --- 3. Predictions vs measured (optional) ---
    if predictions_path is not None and predictions_path.is_file():
        log_checkpoint(log, "step", name="predictions_evaluation")
        from livrables.partie1_predictions_vs_measured import (
            evaluate_all, load_predictions, render_markdown,
        )
        try:
            preds = load_predictions(predictions_path)
            detailed, summary = evaluate_all(preds, results_per_oracle)
            preds_json = output_dir / "predictions_evaluation.json"
            preds_md = output_dir / "predictions_evaluation.md"
            preds_json.write_text(json.dumps({"detailed": detailed, "summary": summary},
                                              indent=2))
            preds_md.write_text(render_markdown(detailed, summary))
            artifacts["predictions_json"] = preds_json
            artifacts["predictions_md"] = preds_md
            log.info("[run_all] Predictions évaluées : %d validés, %d réfutés, %d N/A",
                     summary["n_validated"], summary["n_refuted"], summary["n_unevaluable"])
        except Exception as e:
            log.error("Predictions step KO : %s", e, exc_info=True)
    elif predictions_path is not None:
        log.warning("Predictions YAML introuvable : %s — skip cette étape", predictions_path)

    # --- 4. Figures ---
    log_checkpoint(log, "step", name="figures")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    try:
        from livrables.paper_figures import (
            generate_figure_signatures,
            generate_figure_predictions_vs_measured,
        )
        sig_fig = generate_figure_signatures(table, figures_dir / "signatures.pdf")
        artifacts["figure_signatures"] = sig_fig
        log.info("[run_all] Figure signatures : %s", sig_fig)
        # Figure predictions si disponible
        if "predictions_json" in artifacts:
            with open(artifacts["predictions_json"]) as f:
                data = json.load(f)
            pred_fig = generate_figure_predictions_vs_measured(
                data["detailed"], figures_dir / "predictions.pdf",
            )
            artifacts["figure_predictions"] = pred_fig
    except Exception as e:
        log.error("Figures step KO : %s", e, exc_info=True)

    # --- 5. Summary index ---
    log_checkpoint(log, "step", name="index")
    index_md = "# Index livrables Partie 1\n\nFichiers générés par `livrables.run_all` :\n\n"
    index_md += "| Livrable | Path |\n|---|---|\n"
    for name, path in artifacts.items():
        rel = path.relative_to(output_dir) if path.is_relative_to(output_dir) else path
        index_md += f"| `{name}` | `{rel}` |\n"
    index_md += f"\nOracles inclus : `{sorted(results_per_oracle.keys())}` ({len(results_per_oracle)} total).\n"
    index_path = output_dir / "INDEX.md"
    index_path.write_text(index_md)
    artifacts["index"] = index_path

    log.info("[run_all] %d artefacts produits dans %s", len(artifacts), output_dir)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(prog="livrables.run_all",
                                     description="Génère tous les livrables Partie 1.")
    parser.add_argument("--results", nargs="+", required=True,
                       help="path1.json:oracle_id1 path2.json:oracle_id2 ...")
    parser.add_argument("--predictions", default=None,
                       help="YAML prédictions a priori (optional)")
    parser.add_argument("--output", required=True, help="Dossier output")
    parser.add_argument("--n-max", type=int, default=64,
                       help="N max pour heuristiques signatures (défaut 64)")
    args = parser.parse_args()

    logger = setup_logging(phase="livrables", prefix="livrables_run_all", to_file=True)
    output_dir = Path(args.output)
    results_per_oracle = _load_results(args.results, logger)
    if not results_per_oracle:
        logger.error("Aucun résultats chargé — abandon")
        sys.exit(1)

    predictions_path = Path(args.predictions) if args.predictions else None
    artifacts = run_all(
        results_per_oracle, output_dir,
        predictions_path=predictions_path,
        n_max_for_signatures=args.n_max,
        logger=logger,
    )
    print(f"=== {len(artifacts)} artefacts dans {output_dir}/ ===", flush=True)


if __name__ == "__main__":
    main()
