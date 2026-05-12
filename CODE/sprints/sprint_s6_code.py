"""
sprint_s6_code.py — Sprint S6 : Code Oracle training + adapter.

Spec : DOC/CATALOGUE §3.3 + §S6.

Objectif : entraîner / utiliser un Code Oracle (StarCoder-2-3B ou
MinimalCodeBackend fine-tuned sur Dyck-k extended), mesurer signatures.

Pipeline :
1. Soit StarCoder-2-3B via HFLanguageBackend (causal LM)
2. Soit MinimalCodeBackend trained sur Dyck-k generator
3. Battery research × régimes depth × seq_len
4. Comparer signatures Code vs LL (cross-Oracle table)

Compute : 1-2 jours pod GPU.
"""

from __future__ import annotations

from pathlib import Path

from sprints.base import SprintBase


class SprintS6Code(SprintBase):
    """Sprint S6 — Code Oracle training + adapter."""

    sprint_id = "S6_code"
    expected_duration_hint = "1-2 jours pod GPU"
    expected_compute_cost = "$10-20"
    requires_pod = True

    def __init__(
        self,
        *,
        use_hf_starcoder: bool = False,
        hf_model_name: str = "bigcode/starcoder2-3b",
        local_checkpoint: str | Path | None = None,
        n_bracket_types: int = 4,
        n_examples_per_regime: int = 64,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.use_hf_starcoder = use_hf_starcoder
        self.hf_model_name = hf_model_name
        self.local_checkpoint = Path(local_checkpoint) if local_checkpoint else None
        self.k = n_bracket_types
        self.n_examples = n_examples_per_regime
        self.device = device

    def _run_inner(self) -> None:
        try:
            from catalog.oracles import CodeOracle, CodeModelSpec
        except ImportError as e:
            raise RuntimeError(f"catalog.oracles.code : {e}")

        if self.use_hf_starcoder:
            # StarCoder est causal LM ; on l'utilise via HFLanguageBackend
            # mais comme Code Oracle (avec input Dyck-k tokenisé via tokenizer HF)
            try:
                from catalog.oracles import HFLanguageBackend
            except ImportError as e:
                raise RuntimeError(f"HFLanguageBackend : {e}")
            backend = HFLanguageBackend(self.hf_model_name, device=self.device)
            # Wrap dans CodeOracle (utilise backend forward_with_attn générique)
            oracle = CodeOracle(backend=backend, n_bracket_types=self.k,
                                oracle_id=f"code_{self.hf_model_name}")
        else:
            oracle = CodeOracle(checkpoint_path=self.local_checkpoint,
                                n_bracket_types=self.k, device=self.device)

        self._log_metric("oracle_id", oracle.oracle_id)
        self._log_metric("n_bracket_types", self.k)

        # Battery research
        from catalog.batteries import level_research
        battery = level_research(device=self.device)
        results = battery.run(oracle, n_examples_per_regime=self.n_examples)
        import json
        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        self._add_artifact(results_path, "results JSON Code")
        from catalog.report import render_markdown_report
        md = render_markdown_report(results.to_dict())
        rep_path = self.output_dir / "report.md"
        with open(rep_path, "w") as f:
            f.write(md)
        self._add_artifact(rep_path, "Rapport Code")
        self._log_metric("n_regimes", len(results.per_regime))
