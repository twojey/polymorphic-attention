"""
sprint_d_phase3_v3.py — Sprint D : phase 3 V3+ avec Backbone informé.

Spec : DOC/ROADMAP §3.9 + DOC/03_phase_kernel_asp.

Objectif : entraîner ASPLayer en phase 3 V3 avec backbone identifié par
Sprint C (Butterfly, Banded, Cauchy, etc.). Vérifier val_acc ≥ 0.90 ×
Oracle baseline (0.580 ≥ 0.580).

Pipeline :
1. Parse Sprint C report → backbone class
2. Configurer ASPLayer + smart init via phase3_kernel_asp
3. Lancer phase3_kernel_asp.run_train (subprocess avec stderr capture)
4. Vérifier val_acc ASP ≥ target

Compute : 1-2 sem dev + ~$5-10 sur pod GPU.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from shared.retry import retry_call
from sprints.base import SprintBase


class SprintDPhase3V3(SprintBase):
    """Sprint D — phase 3 V3+ avec backbone informé Sprint C."""

    sprint_id = "D_phase3_v3"
    expected_duration_hint = "1-2 semaines dev + run"
    expected_compute_cost = "$5-10 (GPU pod requis)"
    requires_pod = True

    def __init__(
        self,
        *,
        sprint_c_report: str | Path,
        backbone_class_override: str | None = None,
        oracle_baseline_acc: float = 0.645,
        target_acc_ratio: float = 0.90,
        n_epochs: int = 50,
        device: str = "cuda",
        oracle_checkpoint: str | Path | None = None,
        config_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sprint_c_report = Path(sprint_c_report)
        self.backbone_class_override = backbone_class_override
        self.oracle_baseline_acc = oracle_baseline_acc
        self.target_acc_ratio = target_acc_ratio
        self.n_epochs = n_epochs
        self.device = device
        self.oracle_checkpoint = (
            Path(oracle_checkpoint) if oracle_checkpoint else None
        )
        self.config_path = Path(config_path) if config_path else None

    def _identify_backbone_class(self) -> str:
        """Parse Sprint C report ou utilise override."""
        if self.backbone_class_override is not None:
            return self.backbone_class_override
        if not self.sprint_c_report.is_file():
            raise FileNotFoundError(
                f"Sprint C report introuvable : {self.sprint_c_report}. "
                "Spécifier backbone_class_override pour bypass."
            )
        text = self.sprint_c_report.read_text()
        candidates = [
            "butterfly", "monarch", "banded", "block_diag", "cauchy",
            "toeplitz", "pixelfly", "sparse_lowrank",
        ]
        for cand in candidates:
            if cand in text.lower():
                return cand
        return "dense"

    def _run_phase3(self, backbone: str) -> dict[str, Any]:
        """Appelle phase3_kernel_asp.run_train via subprocess avec retry."""
        cmd = [
            sys.executable, "-m", "phase3_kernel_asp.run_train",
            "--backbone-class", backbone,
            "--n-epochs", str(self.n_epochs),
            "--device", self.device,
            "--output", str(self.output_dir / "phase3_v3"),
        ]
        if self.oracle_checkpoint:
            cmd.extend(["--oracle-checkpoint", str(self.oracle_checkpoint)])
        if self.config_path:
            cmd.extend(["--config", str(self.config_path)])
        cmd.append("--no-mlflow")
        self.logger.info("[%s] cmd: %s", self.sprint_id, " ".join(cmd))

        env = {"PYTHONPATH": "CODE"}
        import os
        env.update(os.environ)

        def _exec():
            return subprocess.run(
                cmd, env=env, check=True, capture_output=True, text=True,
                timeout=24 * 3600,
            )

        try:
            proc = retry_call(
                _exec, args=(), max_attempts=2, base_delay=10.0,
                jitter=1.0, logger=self.logger,
            )
            (self.output_dir / "phase3_stdout.log").write_text(proc.stdout)
            (self.output_dir / "phase3_stderr.log").write_text(proc.stderr)
            return {"status": "ok", "stdout_tail": proc.stdout[-2000:]}
        except subprocess.CalledProcessError as e:
            self.logger.error("phase3 returncode=%s stderr=%s",
                              e.returncode, e.stderr[-1000:])
            return {"status": "failed", "returncode": e.returncode,
                    "stderr_tail": e.stderr[-1000:] if e.stderr else ""}
        except FileNotFoundError:
            # Module pas exécutable en sub : driver squelette uniquement
            self.logger.warning(
                "phase3_kernel_asp.run_train module non disponible, fallback skeleton"
            )
            return {"status": "skeleton", "reason": "phase3_run_train_unavailable"}

    def _run_inner(self) -> None:
        backbone = self._identify_backbone_class()
        self._log_metric("backbone_class", backbone)
        self.logger.info("[%s] backbone identifié : %s",
                         self.sprint_id, backbone)

        self._check_go_nogo(
            "backbone_class_recognized",
            backbone != "dense",
            skip_if_failed=False,
        )

        result = self._run_phase3(backbone)
        self._log_metric("phase3_status", result["status"])

        # Parse val_acc dans stdout si présent
        val_acc = None
        if result.get("stdout_tail"):
            import re
            m = re.search(r"val_acc[=:\s]+([0-9.]+)", result["stdout_tail"])
            if m:
                val_acc = float(m.group(1))
                self._log_metric("val_acc_asp", val_acc)

        target = self.oracle_baseline_acc * self.target_acc_ratio
        if val_acc is not None:
            self._check_go_nogo(
                f"val_acc_asp_above_{target:.3f}",
                val_acc >= target,
                skip_if_failed=False,
            )
        else:
            self._log_metric("val_acc_status",
                             "not_extracted_from_stdout")

        self._add_artifact(
            self.output_dir / "phase3_v3",
            f"Phase 3 V3+ run avec backbone={backbone}",
        )
