"""
base.py — Battery : orchestre Oracle × Properties → résultats.

Spec : DOC/CONTEXT.md §Battery.

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
"""

from __future__ import annotations

import sys
import traceback
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
    """Composition de Properties exécutée sur un Oracle."""

    def __init__(
        self,
        properties: list[Property],
        *,
        name: str = "custom",
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if not properties:
            raise ValueError("Battery requiert au moins une Property")
        self.properties = properties
        self.name = name
        self.device = device
        self.dtype = dtype

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
        })

        per_regime_props = [p for p in self.properties if p.scope == "per_regime"]
        cross_regime_props = [p for p in self.properties if p.scope == "cross_regime"]
        all_dumps: dict[tuple, AttentionDump] = {}

        # --- Per-regime properties ---
        for regime in regimes:
            try:
                dump = oracle.extract_regime(regime, n_examples_per_regime)
                dump.validate()
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[battery] échec extract_regime({regime.key}) : "
                    f"{type(exc).__name__}: {exc}. SKIP régime.",
                    file=sys.stderr, flush=True,
                )
                traceback.print_exc(file=sys.stderr, limit=3)
                continue

            all_dumps[regime.key] = dump
            regime_out: dict[str, dict] = {}
            for ell, A in enumerate(dump.attn):
                # On itère sur les couches : chaque Property s'applique à
                # chaque couche, on agrège ensuite par layer en suffixe nom
                ctx = PropertyContext(
                    device=self.device, dtype=self.dtype,
                    regime={"layer": ell, **{
                        "omega": regime.omega, "delta": regime.delta,
                        "entropy": regime.entropy,
                    }},
                    metadata={"oracle_id": oracle.oracle_id},
                )
                for prop in per_regime_props:
                    try:
                        out = prop.compute(A, ctx)
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"[battery] {prop.name} échec sur régime "
                            f"{regime.key} layer {ell} : {type(exc).__name__}: {exc}",
                            file=sys.stderr, flush=True,
                        )
                        traceback.print_exc(file=sys.stderr, limit=3)
                        continue
                    # On préfixe les clés par layer pour traçabilité
                    layered_out = {f"layer{ell}_{k}": v for k, v in out.items()}
                    regime_out.setdefault(prop.name, {}).update(layered_out)

            results.per_regime[regime.key] = regime_out

        # --- Cross-regime properties ---
        for prop in cross_regime_props:
            ctx = PropertyContext(
                device=self.device, dtype=self.dtype,
                metadata={"oracle_id": oracle.oracle_id},
            )
            try:
                out = prop.compute(all_dumps, ctx)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[battery] cross_regime {prop.name} échec : "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr, flush=True,
                )
                traceback.print_exc(file=sys.stderr, limit=3)
                continue
            results.cross_regime[prop.name] = out

        return results
