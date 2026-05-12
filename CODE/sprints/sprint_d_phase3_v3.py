"""
sprint_d_phase3_v3.py — Sprint D : phase 3 V3+ avec Backbone informé.

Spec : DOC/ROADMAP §3.9 + DOC/03_phase_kernel_asp.

Objectif : architecture ASPLayer informée par la classe identifiée dans
Sprint C (Butterfly ? Banded ? Cauchy ? Block-diag ?), re-test val_acc
vs Oracle 0.645.

Pipeline :
1. Lire DOC/reports/sprint_c.md → classe dominante identifiée
2. Configurer ASPLayer avec le projector approprié (catalog/projectors/)
3. Lancer phase3_kernel_asp.run_train avec smart_init informé
4. Évaluer val_acc vs Oracle baseline 0.645
5. Critère go : val_acc ASP ≥ 0.60 (90 % Oracle)

Compute : 1-2 semaines dev + ~$5 compute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sprint_c_report = Path(sprint_c_report)
        self.backbone_class_override = backbone_class_override
        self.oracle_baseline_acc = oracle_baseline_acc
        self.target_acc_ratio = target_acc_ratio
        self.n_epochs = n_epochs
        self.device = device

    def _identify_backbone_class(self) -> str:
        """Parse Sprint C report ou utilise override."""
        if self.backbone_class_override is not None:
            return self.backbone_class_override
        if not self.sprint_c_report.is_file():
            raise FileNotFoundError(
                f"Sprint C report introuvable : {self.sprint_c_report}. "
                "Spécifier backbone_class_override pour bypass."
            )
        # V1 : parse manuel attendu — TODO Sprint D inner work
        text = self.sprint_c_report.read_text()
        # Heuristique simple : chercher des clés dans le rapport
        candidates = ["butterfly", "monarch", "banded", "block_diag",
                      "cauchy", "toeplitz", "pixelfly", "sparse_lowrank"]
        for cand in candidates:
            if cand in text.lower():
                return cand
        return "dense"  # fallback

    def _run_inner(self) -> None:
        backbone = self._identify_backbone_class()
        self._log_metric("backbone_class", backbone)
        self.logger.info("[%s] Backbone identifié : %s", self.sprint_id, backbone)

        self._check_go_nogo(
            "backbone_class_recognized",
            backbone != "dense",
            skip_if_failed=False,
        )

        # Lazy import phase 3
        try:
            from phase3_kernel_asp import run_train as phase3_run
        except ImportError as e:
            raise RuntimeError(f"phase3_kernel_asp introuvable : {e}")

        # Stub : appel pipeline phase 3 V3+
        # TODO Sprint D : configurer ASPLayer avec backbone + smart_init,
        # lancer training pipeline phase3_kernel_asp.run_train avec
        # config Hydra dédiée.
        self._log_metric("status", "phase3_v3_training_pending_pod")
        self._add_artifact(self.output_dir / "TODO_phase3_v3.md",
                          "Stub Sprint D — à compléter avec pod GPU")

        # Placeholder val_acc check (sera rempli après training réel)
        # val_acc = self._train_and_eval()  # noqa: ERA001
        # self._log_metric("val_acc_asp", val_acc)
        # target = self.oracle_baseline_acc * self.target_acc_ratio
        # self._check_go_nogo(
        #     f"val_acc_asp >= {target:.3f}",
        #     val_acc >= target,
        # )
