"""
mlflow_helpers.py — wrappers MLflow standardisés pour le projet ASP.

- `start_run()` : ouvre un run avec tags obligatoires (phase, sprint,
  domain, oracle_id, status).
- `set_status_invalidated()` : retag a posteriori si une config a été
  modifiée après run (cf. règle T.1).
- `log_yaml_config()` : log la config Hydra résolue comme artefact.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow
import yaml


def require_tracking_uri() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI absent. Sur le pod, ouvrir d'abord le tunnel SSH "
            "vers le VPS puis : export MLFLOW_TRACKING_URI=http://localhost:5000"
        )
    return uri


def start_run(
    *,
    experiment: str,
    run_name: str,
    phase: str,
    sprint: int,
    domain: str,
    status: str,
    oracle_id: str | None = None,
    extra_tags: dict[str, str] | None = None,
) -> mlflow.ActiveRun:
    require_tracking_uri()
    mlflow.set_experiment(experiment)
    tags: dict[str, Any] = {
        "phase": str(phase),
        "sprint": str(sprint),
        "domain": domain,
        "status": status,
    }
    if oracle_id is not None:
        tags["oracle_id"] = oracle_id
    if extra_tags:
        tags.update({k: str(v) for k, v in extra_tags.items()})
    return mlflow.start_run(run_name=run_name, tags=tags)


def set_status_invalidated(run_id: str, reason: str) -> None:
    """Marque un run comme invalidé (config modifiée a posteriori)."""
    require_tracking_uri()
    client = mlflow.MlflowClient()
    client.set_tag(run_id, "status", "invalidated")
    client.set_tag(run_id, "invalidation_reason", reason)


def log_yaml_config(config: dict[str, Any], filename: str = "config.yaml") -> None:
    tmp = Path("/tmp") / filename
    tmp.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    mlflow.log_artifact(str(tmp))
