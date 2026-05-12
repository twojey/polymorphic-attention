"""
sprint_s5_vision.py — Sprint S5 : Vision Oracle training + adapter.

Spec : DOC/CATALOGUE §3.3 + §S5.

Objectif : entraîner un VisionOracle (DINOv2-base) sur MNIST patches /
ImageNet val sample, puis appliquer la Battery research pour mesurer
les signatures Vision.

Pipeline :
1. Soit utiliser DINOv2 pré-entraîné (`facebook/dinov2-base`) via
   HFVisionBackend
2. Soit fine-tune MinimalViT sur MNIST patches local
3. Battery research × régimes patch_size × n_classes
4. Comparer signatures Vision vs SMNIST (cross-Oracle table)

Compute : 1-2 jours pod GPU (training DINOv2 fine-tune ou 5h forward HF).
"""

from __future__ import annotations

from pathlib import Path

from sprints.base import SprintBase


class SprintS5Vision(SprintBase):
    """Sprint S5 — Vision Oracle training + adapter."""

    sprint_id = "S5_vision"
    expected_duration_hint = "1-2 jours pod GPU"
    expected_compute_cost = "$10-20"
    requires_pod = True

    def __init__(
        self,
        *,
        use_hf_dinov2: bool = True,
        hf_model_name: str = "facebook/dinov2-base",
        local_checkpoint: str | Path | None = None,
        dataset: str = "mnist_patches",
        n_examples_per_regime: int = 64,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.use_hf_dinov2 = use_hf_dinov2
        self.hf_model_name = hf_model_name
        self.local_checkpoint = Path(local_checkpoint) if local_checkpoint else None
        self.dataset = dataset
        self.n_examples = n_examples_per_regime
        self.device = device

    def _run_inner(self) -> None:
        # Build VisionOracle
        try:
            from catalog.oracles import VisionOracle, VisionModelSpec
        except ImportError as e:
            raise RuntimeError(f"catalog.oracles.vision : {e}")

        if self.use_hf_dinov2:
            try:
                from catalog.oracles import HFVisionBackend
            except ImportError as e:
                raise RuntimeError(f"HFVisionBackend (transformers requis) : {e}")
            try:
                backend = HFVisionBackend(self.hf_model_name, device=self.device)
            except Exception as e:
                self._check_go_nogo(
                    f"hf_load_{self.hf_model_name}", False, skip_if_failed=True,
                )
                raise
            oracle = VisionOracle(backend=backend, oracle_id=f"vision_{self.hf_model_name}")
        else:
            self._check_go_nogo(
                "local_checkpoint_exists",
                self.local_checkpoint is not None and self.local_checkpoint.is_file(),
                skip_if_failed=True,
            )
            oracle = VisionOracle(checkpoint_path=self.local_checkpoint, device=self.device)

        self._log_metric("oracle_id", oracle.oracle_id)
        self._log_metric("n_layers", oracle.n_layers)

        # Battery
        from catalog.batteries import level_research
        battery = level_research(device=self.device)
        results = battery.run(oracle, n_examples_per_regime=self.n_examples)
        import json
        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        self._add_artifact(results_path, "results JSON Vision")

        from catalog.report import render_markdown_report
        md = render_markdown_report(results.to_dict())
        rep_path = self.output_dir / "report.md"
        with open(rep_path, "w") as f:
            f.write(md)
        self._add_artifact(rep_path, "Rapport Vision")
        self._log_metric("n_regimes", len(results.per_regime))
