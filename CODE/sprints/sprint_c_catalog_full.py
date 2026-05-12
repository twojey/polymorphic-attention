"""
sprint_c_catalog_full.py — Sprint C : Battery level_research sur dumps Sprint B.

Spec : DOC/ROADMAP §3.9 + DOC/CATALOGUE §Battery.

Objectif : exécuter `level_research` (131 Properties) sur les dumps de
Sprint B, agréger les 23 familles, produire un rapport Markdown
"signatures SMNIST détaillées".

Robustesse :
- Mem check via shared.mem_guard avant chaque régime (abort si critique).
- Logs INFO par régime + par propriété pour suivre l'avancement.
- Checkpoint par régime via progress_callback de Battery : si crash en
  milieu de batch, le resume reprend au régime suivant (pas tout
  recommencer).
- `n_workers=1` par défaut (parallèle activable mais multiplierait la RAM).

Pipeline :
1. Charger dumps depuis Sprint B output_dir
2. Pour chaque dump : reconstruire un Oracle de type "frozen" qui replay
   le dump (pas de re-extraction)
3. Battery research × régimes → results.json (incrémental)
4. catalog.report → rapport Markdown
5. Critère go : ≥ 50 % des Properties produisent une valeur non-skip

Compute : ~1 semaine compute selon dumps (CPU possible).
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch

from shared.mem_guard import check_memory
from sprints.base import SprintBase


class _FrozenDumpOracle:
    """Oracle replay-only : lit des dumps pré-extraits depuis le disque."""

    def __init__(self, dumps_dir: Path, oracle_id: str = "frozen_dumps") -> None:
        self.dumps_dir = Path(dumps_dir)
        self.oracle_id = oracle_id
        self.domain = "frozen"
        self._dump_files = sorted(self.dumps_dir.glob("dump_omega*.pt"))
        if not self._dump_files:
            raise FileNotFoundError(f"Aucun dump dans {self.dumps_dir}")
        # Inférer n_layers / n_heads à partir du 1er dump (puis libérer)
        first = torch.load(str(self._dump_files[0]), weights_only=False)
        self.n_layers = len(first["attn"])
        self.n_heads = first["attn"][0].shape[1]
        del first
        gc.collect()

    def regime_grid(self):
        from catalog.oracles.base import RegimeSpec
        grid = []
        for fpath in self._dump_files:
            stem = fpath.stem  # "dump_omega0_delta16"
            parts = stem.split("_")
            omega = int(parts[1].removeprefix("omega"))
            delta = int(parts[2].removeprefix("delta"))
            grid.append(RegimeSpec(omega=omega, delta=delta, entropy=0.0))
        return grid

    def extract_regime(self, regime, n_examples):
        from catalog.oracles.base import AttentionDump
        omega = regime.omega
        delta = regime.delta
        key = f"dump_omega{omega}_delta{delta}"
        path = self.dumps_dir / f"{key}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Dump manquant : {path}")
        d = torch.load(str(path), weights_only=False)
        n_avail = d["attn"][0].shape[0]
        n_take = min(n_examples, n_avail)
        return AttentionDump(
            attn=[a[:n_take] for a in d["attn"]],
            omegas=d["omegas"][:n_take],
            deltas=d["deltas"][:n_take],
            entropies=d["entropies"][:n_take],
            tokens=d["tokens"][:n_take] if d.get("tokens") is not None else None,
            query_pos=d["query_pos"][:n_take] if d.get("query_pos") is not None else None,
            metadata=d.get("metadata", {}),
        )


class SprintCCatalogFull(SprintBase):
    """Sprint C — Battery level_research sur dumps."""

    sprint_id = "C_catalog_full"
    expected_duration_hint = "1 semaine compute + 5j analyse"
    expected_compute_cost = "$5-10 (GPU optionnel)"
    requires_pod = False  # CPU suffit (Battery sur ~9 dumps)

    def __init__(
        self,
        *,
        dumps_dir: str | Path,
        battery_level: str = "research",
        n_examples_per_regime: int = 32,
        device: str = "cpu",
        n_workers: int = 1,
        min_available_gb: float = 4.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dumps_dir = Path(dumps_dir)
        self.battery_level = battery_level
        self.n_examples_per_regime = n_examples_per_regime
        self.device = device
        self.n_workers = n_workers
        self.min_available_gb = min_available_gb

    def _run_inner(self) -> None:
        self._check_go_nogo(
            "dumps_dir_exists",
            self.dumps_dir.is_dir() and any(self.dumps_dir.glob("dump_omega*.pt")),
            skip_if_failed=True,
        )

        check_memory(
            min_available_gb=self.min_available_gb,
            label="sprint-C start", logger=self.logger, abort=True,
        )

        oracle = _FrozenDumpOracle(self.dumps_dir)
        n_dumps = len(oracle._dump_files)
        self._log_metric("oracle_id", oracle.oracle_id)
        self._log_metric("n_layers", oracle.n_layers)
        self._log_metric("n_heads", oracle.n_heads)
        self._log_metric("n_dumps", n_dumps)

        # Battery
        from catalog.batteries import (
            level_minimal, level_principal, level_extended,
            level_full, level_research,
        )
        levels = {
            "minimal": level_minimal, "principal": level_principal,
            "extended": level_extended, "full": level_full,
            "research": level_research,
        }
        battery_fn = levels.get(self.battery_level, level_research)
        battery = battery_fn(device=self.device)
        battery.n_workers = self.n_workers  # override depuis CLI
        self._log_metric("battery_level", self.battery_level)
        self._log_metric("n_properties", len(battery.properties))
        self._log_metric("n_workers", battery.n_workers)
        self.logger.info(
            "[%s] Battery '%s' : %d properties × %d dumps, n_workers=%d, "
            "n_examples_per_regime=%d",
            self.sprint_id, self.battery_level, len(battery.properties),
            n_dumps, battery.n_workers, self.n_examples_per_regime,
        )

        # Si on a déjà les résultats Battery complets en checkpoint, on saute.
        if self._checkpoint_has("battery_results"):
            self.logger.info("[%s] battery_results : déjà calculés (resume)",
                             self.sprint_id)
            results_dict = self._checkpoint_load("battery_results")
        else:
            # Checkpoint par régime via callback (granulaire, résume après crash)
            per_regime_partial: dict[Any, Any] = {}
            if self._checkpoint_has("per_regime_partial"):
                per_regime_partial = self._checkpoint_load("per_regime_partial")
                self.logger.info(
                    "[%s] reprise checkpoint partiel : %d régimes déjà traités",
                    self.sprint_id, len(per_regime_partial),
                )

            def _on_regime(regime_key: Any, regime_out: dict) -> None:
                per_regime_partial[str(regime_key)] = regime_out
                self._checkpoint_save("per_regime_partial", per_regime_partial)
                self.logger.info(
                    "[%s] checkpoint régime %s : %d régimes total saved",
                    self.sprint_id, regime_key, len(per_regime_partial),
                )

            results = battery.run(
                oracle,
                n_examples_per_regime=self.n_examples_per_regime,
                progress_callback=_on_regime,
                min_available_gb=self.min_available_gb,
            )
            results_dict = results.to_dict()
            self._checkpoint_save("battery_results", results_dict)
            gc.collect()

        # Sauvegarder JSON
        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results_dict, f, indent=2, default=str)
        self._add_artifact(results_path, "results JSON Battery × dumps")

        # Stats
        n_regimes = len(results_dict["per_regime"])
        n_props_with_values = 0
        n_skip = 0
        for regime_out in results_dict["per_regime"].values():
            for prop_out in regime_out.values():
                if isinstance(prop_out, dict) and "skip_reason" in prop_out:
                    n_skip += 1
                else:
                    n_props_with_values += 1
        self._log_metric("n_regimes", n_regimes)
        self._log_metric("n_property_outputs_valid", n_props_with_values)
        self._log_metric("n_property_outputs_skipped", n_skip)

        # Rapport Markdown
        from catalog.report import render_markdown_report
        markdown = render_markdown_report(results_dict)
        report_path = self.output_dir / "report.md"
        with open(report_path, "w") as f:
            f.write(markdown)
        self._add_artifact(report_path, "Rapport Markdown signatures")

        # Critère go/no-go : ≥ 50 % des Properties produisent une valeur
        n_total = n_props_with_values + n_skip
        valid_frac = n_props_with_values / max(n_total, 1)
        self._log_metric("valid_property_fraction", valid_frac)
        self._check_go_nogo(
            "min_50pct_properties_produce_value",
            valid_frac >= 0.5,
        )
