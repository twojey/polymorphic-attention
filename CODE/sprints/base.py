"""
base.py — SprintBase : interface abstraite Sprint.

Spec : DOC/ROADMAP §3.9.

Un Sprint exécute une séquence d'opérations (phases existantes + battery
catalog + génération livrable) avec :
- des **critères go/no-go** explicites (cf. DOC/FALSIFIABILITE)
- du **checkpoint/resume** (utilise shared/checkpoint atomique)
- un **rapport** Markdown structuré (DOC/reports/sprints/)
- une **traçabilité MLflow** opt-in

Pattern : sous-classer SprintBase, override `_run_inner()`, et utiliser
les helpers `_checkpoint_save/load`, `_log_metric`, `_write_report`.
"""

from __future__ import annotations

import json
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from shared.checkpoint import Checkpoint


class SprintStatus(str, Enum):
    """État d'un Sprint."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"  # critère go/no-go pas satisfait


@dataclass
class SprintResult:
    """Résultat sérialisable d'un Sprint."""
    sprint_id: str
    status: SprintStatus
    duration_seconds: float
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)  # path → desc
    go_nogo_decisions: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sprint_id": self.sprint_id,
            "status": self.status.value,
            "duration_seconds": self.duration_seconds,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "go_nogo_decisions": self.go_nogo_decisions,
            "error": self.error,
        }


class SprintBase(ABC):
    """Interface abstraite Sprint.

    Sous-classes :
    - `SprintBReExtract` : re-extraction phase 1 V2
    - `SprintCCatalogFull` : Battery research × dumps
    - `SprintDPhase3V3` : phase 3 V3+ avec Backbone informé
    - `SprintEPhase4Warmup` : phase 4a warm-up Spectromètre
    - `SprintFPhase4Autonomous` : phase 4b autonomous routing
    - `SprintGPhase5Validation` : phase 5 tests 5a-6c
    - `SprintS4SMNISTExtended` : SMNIST seq_len étendu
    - `SprintS5Vision` : Vision Oracle training + adapter
    - `SprintS6Code` : Code Oracle training + adapter
    - `SprintS7LL` : LL Oracle (TinyStories) training + adapter

    Convention metadata :
    - `sprint_id` : identifiant unique (ex "B_re_extract", "C_catalog_full")
    - `expected_duration_hint` : estimation wall-clock (string humain)
    - `expected_compute_cost` : estimation $$ pod (string)
    - `requires_pod` : True si Sprint demande GPU pod (vs VPS-only)
    """

    sprint_id: str = ""
    expected_duration_hint: str = ""
    expected_compute_cost: str = ""
    requires_pod: bool = False

    def __init__(
        self,
        *,
        output_dir: str | Path,
        checkpoint_dir: str | Path | None = None,
        mlflow_uri: str | None = None,
        seed: int = 0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow_uri = mlflow_uri
        self.seed = seed
        self._result = SprintResult(sprint_id=self.sprint_id,
                                     status=SprintStatus.PENDING,
                                     duration_seconds=0.0)
        self._mlflow_run: Any = None
        ckpt_state_dir = self.checkpoint_dir / self.sprint_id
        self._checkpoint, self._checkpoint_resumed = Checkpoint.create_or_resume(
            ckpt_state_dir,
            fingerprint={"sprint_id": self.sprint_id, "seed": seed},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> SprintResult:
        """Exécute le Sprint. Capture errors, gère checkpoint/resume."""
        print(f"=== Sprint {self.sprint_id} : START ===", flush=True)
        t0 = time.time()
        self._result.status = SprintStatus.RUNNING
        if self.mlflow_uri:
            self._setup_mlflow()
        try:
            self._run_inner()
            self._result.status = SprintStatus.SUCCESS
        except _SprintSkipped as e:
            self._result.status = SprintStatus.SKIPPED
            self._result.error = str(e)
            print(f"[{self.sprint_id}] SKIPPED : {e}", file=sys.stderr, flush=True)
        except Exception as e:
            self._result.status = SprintStatus.FAILED
            self._result.error = f"{type(e).__name__}: {e}"
            print(f"[{self.sprint_id}] FAILED : {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
        finally:
            self._result.duration_seconds = time.time() - t0
            self._write_summary()
            self._teardown_mlflow()
        print(f"=== Sprint {self.sprint_id} : {self._result.status.value.upper()} "
              f"({self._result.duration_seconds:.1f}s) ===", flush=True)
        return self._result

    # ------------------------------------------------------------------
    # Subclass API
    # ------------------------------------------------------------------

    @abstractmethod
    def _run_inner(self) -> None:
        """Logique du Sprint. Lever _SprintSkipped si critère pas satisfait."""
        ...

    def _check_go_nogo(self, criterion: str, condition: bool, *,
                       skip_if_failed: bool = False) -> None:
        """Évalue un critère go/no-go et logge la décision."""
        decision = {
            "criterion": criterion,
            "passed": bool(condition),
            "timestamp": time.time(),
        }
        self._result.go_nogo_decisions.append(decision)
        if not condition and skip_if_failed:
            raise _SprintSkipped(f"go/no-go failed: {criterion}")

    def _log_metric(self, key: str, value: Any) -> None:
        """Logge une métrique vers MLflow + dans le résultat."""
        self._result.metrics[key] = value
        if self._mlflow_run is not None:
            try:
                import mlflow  # noqa: PLC0415
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, float(value))
                else:
                    mlflow.log_param(key, str(value))
            except Exception:
                pass

    def _add_artifact(self, path: str | Path, description: str) -> None:
        """Référence un fichier produit par le Sprint."""
        self._result.artifacts[str(path)] = description
        if self._mlflow_run is not None:
            try:
                import mlflow  # noqa: PLC0415
                mlflow.log_artifact(str(path))
            except Exception:
                pass

    def _checkpoint_save(self, key: str, obj: Any) -> None:
        """Persiste un objet checkpoint sous une clé nommée (atomique)."""
        self._checkpoint.save(key, obj)

    def _checkpoint_has(self, key: str) -> bool:
        """True si l'étape `key` a déjà été sauvegardée."""
        return self._checkpoint.has(key)

    def _checkpoint_load(self, key: str) -> Any:
        """Recharge l'état `key`. Lève FileNotFoundError si absent."""
        return self._checkpoint.load(key)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _setup_mlflow(self) -> None:
        try:
            import mlflow  # noqa: PLC0415
            mlflow.set_tracking_uri(self.mlflow_uri)
            mlflow.set_experiment(f"sprint_{self.sprint_id}")
            self._mlflow_run = mlflow.start_run(run_name=self.sprint_id)
        except Exception as e:
            print(f"[{self.sprint_id}] MLflow setup KO : {e}", file=sys.stderr)
            self._mlflow_run = None

    def _teardown_mlflow(self) -> None:
        if self._mlflow_run is not None:
            try:
                import mlflow  # noqa: PLC0415
                mlflow.end_run()
            except Exception:
                pass
            self._mlflow_run = None

    def _write_summary(self) -> None:
        """Écrit un résumé JSON dans output_dir."""
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(self._result.to_dict(), f, indent=2, default=str)
        self._add_artifact(summary_path, "Sprint summary JSON")


class _SprintSkipped(Exception):
    """Levé quand un critère go/no-go bloquant échoue."""
