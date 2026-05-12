"""
sprint_s7_ll.py — Sprint S7 : LL Oracle (TinyStories) training + adapter.

Spec : DOC/CATALOGUE §3.3 + §S7.

Objectif : entraîner / utiliser un Language Oracle (Llama-3.2-1B ou
MinimalLM fine-tuned sur TinyStories nested-clauses), mesurer signatures.

Pipeline :
1. Soit Llama-3.2-1B via HFLanguageBackend
2. Soit MinimalLM fine-tuned sur TinyStories template
3. Battery research × régimes depth × seq_len
4. Comparer signatures LL vs SMNIST + Vision + Code

Compute : 2 jours pod GPU (Llama 1B fits A100, sinon training MinimalLM).
"""

from __future__ import annotations

from pathlib import Path

from sprints.base import SprintBase


class SprintS7LL(SprintBase):
    """Sprint S7 — LL Oracle (TinyStories) training + adapter."""

    sprint_id = "S7_ll"
    expected_duration_hint = "2 jours pod GPU"
    expected_compute_cost = "$10-20"
    requires_pod = True

    def __init__(
        self,
        *,
        use_hf_llama: bool = True,
        hf_model_name: str = "meta-llama/Llama-3.2-1B",
        local_checkpoint: str | Path | None = None,
        n_examples_per_regime: int = 64,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.use_hf_llama = use_hf_llama
        self.hf_model_name = hf_model_name
        self.local_checkpoint = Path(local_checkpoint) if local_checkpoint else None
        self.n_examples = n_examples_per_regime
        self.device = device

    def _run_inner(self) -> None:
        try:
            from catalog.oracles import LLOracle, LLModelSpec
        except ImportError as e:
            raise RuntimeError(f"catalog.oracles.language : {e}")

        if self.use_hf_llama:
            try:
                from catalog.oracles import HFLanguageBackend
            except ImportError as e:
                raise RuntimeError(f"HFLanguageBackend : {e}")
            try:
                backend = HFLanguageBackend(self.hf_model_name, device=self.device)
            except Exception as e:
                self._check_go_nogo(
                    f"hf_load_{self.hf_model_name}", False, skip_if_failed=True,
                )
                raise
            oracle = LLOracle(backend=backend, oracle_id=f"ll_{self.hf_model_name}")
        else:
            self._check_go_nogo(
                "local_checkpoint_exists",
                self.local_checkpoint is not None and self.local_checkpoint.is_file(),
                skip_if_failed=True,
            )
            oracle = LLOracle(checkpoint_path=self.local_checkpoint, device=self.device)

        self._log_metric("oracle_id", oracle.oracle_id)

        from catalog.batteries import level_research
        battery = level_research(device=self.device)
        results = battery.run(oracle, n_examples_per_regime=self.n_examples)
        import json
        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        self._add_artifact(results_path, "results JSON LL")
        from catalog.report import render_markdown_report
        md = render_markdown_report(results.to_dict())
        rep_path = self.output_dir / "report.md"
        with open(rep_path, "w") as f:
            f.write(md)
        self._add_artifact(rep_path, "Rapport LL")
        self._log_metric("n_regimes", len(results.per_regime))
