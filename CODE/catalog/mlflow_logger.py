"""
mlflow_logger.py — MLflow logging pour Battery runs (deepening #6).

Wrappe shared.mlflow_helpers pour le contexte catalog : tags figés
(phase=catalog, sprint=A1+), log_params Battery + Properties, log_metric
pour chaque scalar retourné par Property, log_artifact pour results.json.

Désactivé silencieusement si MLFLOW_TRACKING_URI absent (graceful fallback
local, cf. feedback_script_robustness §5 mlflow opt-in).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from catalog.batteries.base import BatteryResults


def is_mlflow_active() -> bool:
    """True si MLFLOW_TRACKING_URI est défini (logging activé)."""
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


def log_battery_results(
    results: BatteryResults,
    *,
    output_dir: Path,
    run_name: str,
    sprint: int = 1,
    domain: str = "smnist",
    status: str = "exploratory",
    oracle_id: str | None = None,
    extra_tags: dict[str, str] | None = None,
) -> None:
    """Log results dans MLflow (tags catalog + scalars par régime).

    Silencieux si MLFLOW_TRACKING_URI absent. Erreur de log ≠ crash run.
    """
    if not is_mlflow_active():
        print(
            "[mlflow_logger] MLFLOW_TRACKING_URI absent, skip logging "
            "(résultats JSON dans output_dir disponibles).",
            file=sys.stderr, flush=True,
        )
        return

    try:
        import mlflow
        from shared.mlflow_helpers import start_run, log_yaml_config
    except ImportError as exc:
        print(f"[mlflow_logger] mlflow indisponible : {exc}. Skip.", file=sys.stderr)
        return

    try:
        with start_run(
            experiment="catalog",
            run_name=run_name,
            phase="catalog",
            sprint=sprint,
            domain=domain,
            status=status,
            oracle_id=oracle_id or results.metadata.get("oracle_id"),
            extra_tags=extra_tags,
        ):
            # Params : metadata Battery
            params = {
                "battery_name": results.metadata.get("battery_name"),
                "n_regimes": results.metadata.get("n_regimes"),
                "n_examples_per_regime": results.metadata.get("n_examples_per_regime"),
                "device": results.metadata.get("device"),
                "dtype": results.metadata.get("dtype"),
                "n_properties": len(results.metadata.get("properties", [])),
            }
            mlflow.log_params({k: str(v) for k, v in params.items() if v is not None})

            # Log liste des Properties activées
            props = results.metadata.get("properties", [])
            mlflow.log_param("property_list", ",".join(props[:30]))  # tronqué

            # Log métriques per-régime : pour chaque scalar numérique
            for regime_key, regime_out in results.per_regime.items():
                regime_tag = str(regime_key).replace(" ", "").replace("(", "").replace(")", "").replace(",", "_")[:80]
                for prop_name, prop_out in regime_out.items():
                    for key, val in prop_out.items():
                        if isinstance(val, (int, float)) and not isinstance(val, bool):
                            metric_name = f"{prop_name}__{regime_tag}__{key}"[:250]
                            try:
                                mlflow.log_metric(metric_name, float(val))
                            except Exception:  # noqa: BLE001
                                pass

            # Cross-régime
            for prop_name, prop_out in results.cross_regime.items():
                for key, val in prop_out.items():
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        metric_name = f"cross__{prop_name}__{key}"[:250]
                        try:
                            mlflow.log_metric(metric_name, float(val))
                        except Exception:
                            pass

            # Artifact : results.json
            results_json = output_dir / "results.json"
            if results_json.is_file():
                mlflow.log_artifact(str(results_json))
    except Exception as exc:  # noqa: BLE001
        print(
            f"[mlflow_logger] Échec MLflow logging (non-bloquant) : "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr, flush=True,
        )
