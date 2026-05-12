"""
base.py — Battery : orchestre Oracle × Properties → résultats.

Spec : DOC/00_FONDATIONS.md §Battery.

Une Battery prend en entrée :
- un Oracle (qui fournit AttentionDump par régime)
- une liste de Properties (qui calculent des mesures sur chaque dump)
- un PropertyContext template (device, dtype)

Et orchestre :
1. Pour chaque régime de la grille Oracle, extract_regime(regime)
2. Pour chaque Property dans la liste, compute(attn, ctx)
3. Agrège cross-régime en RegimeStats (V3.5 distributional view)
4. Retourne un BatteryResults sérialisable

Les Properties scope="cross_regime" reçoivent toutes les attentions
collectées d'un coup (au lieu d'un dump par appel).

**Parallélisation** : `n_workers > 1` active un ThreadPoolExecutor pour
dispatcher les régimes en parallèle. Adapté pour les workloads dominés
par les SVD batchées (qui libèrent le GIL via les bindings BLAS/LAPACK
de PyTorch). Désactivé par défaut (n_workers=1) car certains Oracles
HuggingFace ne sont pas thread-safe pendant le forward.
"""

from __future__ import annotations

import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import torch

from catalog.oracles.base import AbstractOracle, AttentionDump, RegimeSpec
from catalog.properties.base import Property, PropertyContext


@dataclass
class BatteryResults:
    """Résultats d'une Battery exécutée sur un Oracle.

    Structure :
    - `per_regime[regime_key][property_name]` : dict de scalaires retournés
      par Property.compute pour ce régime
    - `cross_regime[property_name]` : dict de scalaires pour Properties à
      scope cross_regime
    - `metadata` : oracle_id, n_examples_per_regime, durations, etc.
    """

    per_regime: dict[Any, dict[str, dict[str, float | int | str | bool]]] = field(
        default_factory=dict
    )
    cross_regime: dict[str, dict[str, float | int | str | bool]] = field(
        default_factory=dict
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Sérialisation JSON-compatible (les clés régime tuple → str)."""
        return {
            "per_regime": {
                str(k): v for k, v in self.per_regime.items()
            },
            "cross_regime": self.cross_regime,
            "metadata": self.metadata,
        }


class Battery:
    """Composition de Properties exécutée sur un Oracle.

    Args
    ----
    properties : liste de Property activées
    name : identifiant Battery (ex "research")
    device, dtype : politique exec (cf. PropertyContext)
    n_workers : nombre de threads pour dispatch régimes parallèle. Défaut 1
        (séquentiel). > 1 → ThreadPoolExecutor. Idéal pour CPU + SVD ;
        ATTENTION avec Oracles HuggingFace non thread-safe.
    """

    def __init__(
        self,
        properties: list[Property],
        *,
        name: str = "custom",
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        n_workers: int = 1,
    ) -> None:
        if not properties:
            raise ValueError("Battery requiert au moins une Property")
        if n_workers < 1:
            raise ValueError(f"n_workers ≥ 1, reçu {n_workers}")
        self.properties = properties
        self.name = name
        self.device = device
        self.dtype = dtype
        self.n_workers = n_workers
        self._stderr_lock = threading.Lock()

    def run(
        self,
        oracle: AbstractOracle,
        *,
        regimes: list[RegimeSpec] | None = None,
        n_examples_per_regime: int = 32,
    ) -> BatteryResults:
        """Exécute la Battery sur l'Oracle.

        Si `regimes` est None, utilise `oracle.regime_grid()`. Les Properties
        scope=per_regime tournent par régime ; les Properties cross_regime
        reçoivent toutes les attentions collectées en fin de boucle.

        Avec n_workers > 1, les régimes sont dispatchés en parallèle via
        ThreadPoolExecutor. Les Properties cross_regime tournent toujours
        en série après collecte (elles dépendent de tous les dumps).
        """
        if regimes is None:
            regimes = oracle.regime_grid()
        if not regimes:
            raise ValueError("Battery : pas de régimes (oracle.regime_grid() vide)")

        results = BatteryResults()
        results.metadata.update({
            "battery_name": self.name,
            "oracle_id": oracle.oracle_id,
            "domain": oracle.domain,
            "n_regimes": len(regimes),
            "n_examples_per_regime": n_examples_per_regime,
            "device": self.device,
            "dtype": str(self.dtype),
            "properties": [p.name for p in self.properties],
            "n_workers": self.n_workers,
        })

        per_regime_props = [p for p in self.properties if p.scope == "per_regime"]
        per_regime_layers_props = [
            p for p in self.properties if p.scope == "per_regime_layers"
        ]
        cross_regime_props = [p for p in self.properties if p.scope == "cross_regime"]
        all_dumps: dict[tuple, AttentionDump] = {}

        # --- Per-regime dispatch (séquentiel ou parallèle) ---
        if self.n_workers == 1:
            for regime in regimes:
                res = self._process_one_regime(
                    regime, oracle, n_examples_per_regime,
                    per_regime_props, per_regime_layers_props,
                )
                if res is None:
                    continue
                regime_key, regime_out, dump = res
                results.per_regime[regime_key] = regime_out
                all_dumps[regime_key] = dump
        else:
            with ThreadPoolExecutor(max_workers=self.n_workers,
                                    thread_name_prefix="battery") as ex:
                futures = {
                    ex.submit(
                        self._process_one_regime,
                        regime, oracle, n_examples_per_regime,
                        per_regime_props, per_regime_layers_props,
                    ): regime for regime in regimes
                }
                for future in as_completed(futures):
                    regime = futures[future]
                    try:
                        res = future.result()
                    except Exception as exc:  # noqa: BLE001
                        self._log_err(
                            f"[battery] worker crash sur régime {regime.key} : "
                            f"{type(exc).__name__}: {exc}"
                        )
                        continue
                    if res is None:
                        continue
                    regime_key, regime_out, dump = res
                    results.per_regime[regime_key] = regime_out
                    all_dumps[regime_key] = dump

        # --- Cross-regime properties (toujours séquentiel) ---
        for prop in cross_regime_props:
            ctx = PropertyContext(
                device=self.device, dtype=self.dtype,
                metadata={"oracle_id": oracle.oracle_id},
            )
            try:
                out = prop.compute(all_dumps, ctx)
            except Exception as exc:  # noqa: BLE001
                self._log_err(
                    f"[battery] cross_regime {prop.name} échec : "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            results.cross_regime[prop.name] = out

        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _process_one_regime(
        self,
        regime: RegimeSpec,
        oracle: AbstractOracle,
        n_examples_per_regime: int,
        per_regime_props: list[Property],
        per_regime_layers_props: list[Property],
    ) -> tuple[tuple, dict, AttentionDump] | None:
        """Traite un régime : extract + computes + return (key, out, dump).

        Retourne None si extract_regime échoue (régime skipé). Thread-safe :
        chaque appel a son propre PropertyContext (cache local).
        """
        try:
            dump = oracle.extract_regime(regime, n_examples_per_regime)
            dump.validate()
        except Exception as exc:  # noqa: BLE001
            self._log_err(
                f"[battery] échec extract_regime({regime.key}) : "
                f"{type(exc).__name__}: {exc}. SKIP régime."
            )
            return None

        regime_out: dict[str, dict] = {}
        for ell, A in enumerate(dump.attn):
            ctx = PropertyContext(
                device=self.device, dtype=self.dtype,
                regime={"layer": ell, **{
                    "omega": regime.omega, "delta": regime.delta,
                    "entropy": regime.entropy,
                }},
                metadata={
                    "oracle_id": oracle.oracle_id,
                    "tokens": dump.tokens,
                    "query_pos": dump.query_pos,
                    "omegas": dump.omegas,
                    "deltas": dump.deltas,
                    "entropies": dump.entropies,
                },
            )
            for prop in per_regime_props:
                try:
                    out = prop.compute(A, ctx)
                except Exception as exc:  # noqa: BLE001
                    self._log_err(
                        f"[battery] {prop.name} échec sur régime "
                        f"{regime.key} layer {ell} : {type(exc).__name__}: {exc}"
                    )
                    continue
                layered_out = {f"layer{ell}_{k}": v for k, v in out.items()}
                regime_out.setdefault(prop.name, {}).update(layered_out)

        if per_regime_layers_props:
            ctx_layers = PropertyContext(
                device=self.device, dtype=self.dtype,
                regime={
                    "omega": regime.omega, "delta": regime.delta,
                    "entropy": regime.entropy, "n_layers": len(dump.attn),
                },
                metadata={"oracle_id": oracle.oracle_id},
            )
            for prop in per_regime_layers_props:
                try:
                    out = prop.compute(dump.attn, ctx_layers)
                except Exception as exc:  # noqa: BLE001
                    self._log_err(
                        f"[battery] {prop.name} (cross-layer) échec sur régime "
                        f"{regime.key} : {type(exc).__name__}: {exc}"
                    )
                    continue
                regime_out[prop.name] = out

        return regime.key, regime_out, dump

    def _log_err(self, msg: str) -> None:
        """Log thread-safe vers stderr (lock pour éviter interleaving)."""
        with self._stderr_lock:
            print(msg, file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr, limit=3)
