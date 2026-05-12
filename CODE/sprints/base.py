"""
base.py — SprintBase : interface abstraite Sprint.

Spec : DOC/ROADMAP §3.9.

Un Sprint exécute une séquence d'opérations (phases existantes + battery
catalog + génération livrable) avec :
- des **critères go/no-go** explicites (cf. DOC/FALSIFIABILITE)
- du **checkpoint/resume** (utilise shared/checkpoint atomique)
- un **rapport** Markdown structuré (DOC/reports/sprints/)
- une **traçabilité MLflow** opt-in
- du **logging horodaté** vers fichier + console (shared/logging_helpers)

Pattern : sous-classer SprintBase, override `_run_inner()`, et utiliser
les helpers `_checkpoint_save/load`, `_log_metric`, `_write_report`.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from shared.checkpoint import Checkpoint
from shared.logging_helpers import setup_logging, log_checkpoint


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
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sprint_id": self.sprint_id,
            "status": self.status.value,
            "duration_seconds": self.duration_seconds,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "go_nogo_decisions": self.go_nogo_decisions,
            "error": self.error,
            "manifest": self.manifest,
        }


class SprintBase(ABC):
    """Interface abstraite Sprint.

    Convention metadata (class attributes) :
    - `sprint_id` : identifiant unique
    - `expected_duration_hint` / `expected_compute_cost` : estimations
    - `requires_pod` : True si Sprint demande GPU pod
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
        log_level: int = logging.INFO,
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
        # Logging : un fichier horodaté par run, dans output_dir
        # NB : on configure le root logger via setup_logging, donc tous les
        # sous-modules (catalog.battery, etc.) bénéficient du même handler.
        self.logger = self._setup_logger(log_level)
        self._result.manifest = self._build_manifest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> SprintResult:
        """Exécute le Sprint. Capture errors, gère checkpoint/resume + logging.

        Capture en plus de Exception : MemoryError (OOM Python-level) et
        KeyboardInterrupt sont loggés et propagés au top-level dispatcher
        (run.py main()) pour produire un summary.json même en cas de mort
        violente.
        """
        self.logger.info("=== Sprint %s : START ===", self.sprint_id)
        if self._checkpoint_resumed:
            self.logger.info(
                "[%s] checkpoint reprise (keys déjà sauvegardées : %s)",
                self.sprint_id, self._checkpoint.keys(),
            )
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
            self.logger.warning("[%s] SKIPPED : %s", self.sprint_id, e)
        except MemoryError as e:
            self._result.status = SprintStatus.FAILED
            self._result.error = f"MemoryError: {e}"
            self.logger.exception("[%s] OOM (MemoryError) : %s", self.sprint_id, e)
            # On laisse propager après finally pour que run.py top-level
            # le voie et exit 137 (convention OOM)
            self._safe_finalize(t0)
            raise
        except KeyboardInterrupt as e:
            self._result.status = SprintStatus.FAILED
            self._result.error = f"KeyboardInterrupt: {e}"
            self.logger.warning("[%s] interrupted (Ctrl-C)", self.sprint_id)
            self._safe_finalize(t0)
            raise
        except Exception as e:
            self._result.status = SprintStatus.FAILED
            self._result.error = f"{type(e).__name__}: {e}"
            self.logger.exception("[%s] FAILED : %s", self.sprint_id, e)
        finally:
            self._safe_finalize(t0)
        self.logger.info(
            "=== Sprint %s : %s (%.1fs) ===",
            self.sprint_id, self._result.status.value.upper(),
            self._result.duration_seconds,
        )
        return self._result

    def _safe_finalize(self, t_start: float) -> None:
        """Écrit summary.json + ferme MLflow, en avalant toute exception
        (le finally NE DOIT PAS masquer l'exception originelle)."""
        try:
            self._result.duration_seconds = time.time() - t_start
            self._write_summary()
        except Exception as exc:  # noqa: BLE001
            self.logger.error("[%s] _write_summary KO : %s", self.sprint_id, exc)
        try:
            self._teardown_mlflow()
        except Exception as exc:  # noqa: BLE001
            self.logger.error("[%s] _teardown_mlflow KO : %s", self.sprint_id, exc)

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
        status = "PASS" if condition else "FAIL"
        self.logger.info("[go/no-go] %s : %s", criterion, status)
        if not condition and skip_if_failed:
            raise _SprintSkipped(f"go/no-go failed: {criterion}")

    def _log_metric(self, key: str, value: Any) -> None:
        """Logge une métrique vers MLflow + dans le résultat + logger."""
        self._result.metrics[key] = value
        log_checkpoint(self.logger, "metric", **{key: value})
        if self._mlflow_run is not None:
            try:
                import mlflow  # noqa: PLC0415
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, float(value))
                else:
                    mlflow.log_param(key, str(value))
            except Exception as e:
                self.logger.warning("MLflow log_metric(%s) KO : %s", key, e)

    def _add_artifact(self, path: str | Path, description: str) -> None:
        """Référence un fichier produit par le Sprint."""
        self._result.artifacts[str(path)] = description
        self.logger.info("[artifact] %s : %s", path, description)
        if self._mlflow_run is not None:
            try:
                import mlflow  # noqa: PLC0415
                mlflow.log_artifact(str(path))
            except Exception as e:
                self.logger.warning("MLflow log_artifact(%s) KO : %s", path, e)

    def _checkpoint_save(self, key: str, obj: Any) -> None:
        """Persiste un objet checkpoint sous une clé nommée (atomique)."""
        self._checkpoint.save(key, obj)
        self.logger.info("[ckpt] save key=%s", key)

    def _checkpoint_has(self, key: str) -> bool:
        return self._checkpoint.has(key)

    def _checkpoint_load(self, key: str) -> Any:
        return self._checkpoint.load(key)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _setup_logger(self, level: int) -> logging.Logger:
        """Configure le root logger + retourne un logger nommé sprint.<id>.

        Fichier log persistent : <output_dir>/sprint.log (append mode).
        Console (stderr) : tous les niveaux ≥ INFO.
        """
        # On utilise shared.logging_helpers en lui pointant le fichier voulu
        # via ASP_LOG_FILE env var, puis on appelle setup_logging.
        log_file = self.output_dir / "sprint.log"
        os.environ["ASP_LOG_FILE"] = str(log_file)
        # On force ASP_REPO_ROOT pour fallback path
        if "ASP_REPO_ROOT" not in os.environ:
            for parent in [Path.cwd(), *Path.cwd().parents]:
                if (parent / "OPS").is_dir() and (parent / "CODE").is_dir():
                    os.environ["ASP_REPO_ROOT"] = str(parent)
                    break
        setup_logging(phase=self.sprint_id, prefix=f"sprint_{self.sprint_id}",
                      level=level, to_file=True, reuse_bash_log=True)
        return logging.getLogger(f"sprint.{self.sprint_id}")

    def _build_manifest(self) -> dict[str, Any]:
        """Construit le manifest de run : git, env, machine."""
        manifest: dict[str, Any] = {
            "sprint_id": self.sprint_id,
            "seed": self.seed,
            "timestamp_start": time.time(),
            "requires_pod": self.requires_pod,
            "expected_duration_hint": self.expected_duration_hint,
            "expected_compute_cost": self.expected_compute_cost,
        }
        # Git hash
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL, timeout=2,
            ).decode().strip()
            manifest["git_hash"] = commit
        except Exception:
            manifest["git_hash"] = "unknown"
        # Git dirty flag
        try:
            dirty = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL, timeout=2,
            ).decode().strip()
            manifest["git_dirty"] = bool(dirty)
        except Exception:
            manifest["git_dirty"] = None
        # Python + torch version
        import platform
        manifest["python_version"] = platform.python_version()
        manifest["torch_version"] = torch.__version__
        manifest["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            manifest["cuda_device"] = torch.cuda.get_device_name(0)
        return manifest

    def _setup_mlflow(self) -> None:
        try:
            import mlflow  # noqa: PLC0415
            mlflow.set_tracking_uri(self.mlflow_uri)
            mlflow.set_experiment(f"sprint_{self.sprint_id}")
            self._mlflow_run = mlflow.start_run(run_name=self.sprint_id)
            # Log manifest comme params MLflow
            for k, v in self._result.manifest.items():
                try:
                    mlflow.log_param(k, str(v))
                except Exception:
                    pass
        except Exception as e:
            self.logger.warning("[%s] MLflow setup KO : %s", self.sprint_id, e)
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
