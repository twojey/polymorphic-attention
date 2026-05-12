"""
sprint_e_phase4_warmup.py — Sprint E : phase 4a warm-up Spectromètre.

Spec : DOC/04_phase_routage_budget §4a.

Objectif : entraîner le Spectromètre conjointement avec ASPLayer en mode
warm-up (FrozenAlphaSpectrometer + distillation V3.5 p75 asymétrique).
Vérifier transition 4a → 4b (loss converge + Spearman > 0.50 + variance OK).

Pipeline :
1. Charger checkpoint Sprint D (phase 3 V3+)
2. Lancer phase4_routage_budget.run --max-epochs-4a N --max-epochs-4b 0
3. Vérifier transition_ok dans results.json
4. Critère go : transition_4a_to_4b == True

Compute : ~1 jour pod GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from shared.retry import retry_call
from sprints.base import SprintBase


class SprintEPhase4Warmup(SprintBase):
    """Sprint E — phase 4a warm-up Spectromètre + distillation V3.5."""

    sprint_id = "E_phase4_warmup"
    expected_duration_hint = "1 jour pod GPU"
    expected_compute_cost = "$5-10"
    requires_pod = True

    def __init__(
        self,
        *,
        sprint_d_checkpoint: str | Path,
        config_path: str | Path,
        oracle_checkpoint: str | Path,
        signals: str = "S_Spectral",
        max_epochs_4a: int = 30,
        steps_per_epoch: int = 200,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sprint_d_checkpoint = Path(sprint_d_checkpoint)
        self.config_path = Path(config_path)
        self.oracle_checkpoint = Path(oracle_checkpoint)
        self.signals = signals
        self.max_epochs_4a = max_epochs_4a
        self.steps_per_epoch = steps_per_epoch
        self.device = device

    def _run_phase4_warmup(self) -> dict[str, Any]:
        phase4_out = self.output_dir / "phase4_4a"
        cmd = [
            sys.executable, "-m", "phase4_routage_budget.run",
            "--config", str(self.config_path),
            "--oracle-checkpoint", str(self.oracle_checkpoint),
            "--signals", self.signals,
            "--output", str(phase4_out),
            "--max-epochs-4a", str(self.max_epochs_4a),
            "--max-epochs-4b", "0",  # 4a only
            "--steps-per-epoch", str(self.steps_per_epoch),
            "--device", self.device,
            "--no-mlflow",
        ]
        self.logger.info("[%s] cmd: %s", self.sprint_id, " ".join(cmd))

        env = {"PYTHONPATH": "CODE"}
        env.update(os.environ)

        def _exec():
            return subprocess.run(
                cmd, env=env, check=True, capture_output=True, text=True,
                timeout=12 * 3600,
            )

        try:
            proc = retry_call(
                _exec, args=(), max_attempts=2, base_delay=10.0,
                jitter=1.0, logger=self.logger,
            )
            (self.output_dir / "phase4_stdout.log").write_text(proc.stdout)
            (self.output_dir / "phase4_stderr.log").write_text(proc.stderr)
            results_path = phase4_out / "results.json"
            if results_path.is_file():
                return {"status": "ok",
                        "results": json.loads(results_path.read_text())}
            return {"status": "ok_no_results"}
        except subprocess.CalledProcessError as e:
            self.logger.error(
                "phase4 returncode=%s stderr=%s",
                e.returncode, e.stderr[-1000:] if e.stderr else "",
            )
            return {"status": "failed", "returncode": e.returncode}

    def _run_inner(self) -> None:
        self._check_go_nogo(
            "sprint_d_checkpoint_exists",
            self.sprint_d_checkpoint.is_file() or self.sprint_d_checkpoint.is_dir(),
            skip_if_failed=False,
        )
        self._check_go_nogo(
            "oracle_checkpoint_exists",
            self.oracle_checkpoint.is_file(),
            skip_if_failed=True,
        )

        result = self._run_phase4_warmup()
        self._log_metric("phase4a_status", result["status"])

        if "results" in result:
            data = result["results"].get("phase_4a", {})
            transition_ok = bool(data.get("transition_4a_to_4b", False))
            self._log_metric("transition_4a_to_4b", transition_ok)
            diag = data.get("transition_diagnostics", {})
            for k, v in diag.items():
                self._log_metric(f"diag_{k}", v)

            self._check_go_nogo(
                "transition_4a_to_4b_passed",
                transition_ok,
                skip_if_failed=False,
            )

        self._add_artifact(
            self.output_dir / "phase4_4a",
            "Phase 4a warm-up Spectromètre training",
        )
