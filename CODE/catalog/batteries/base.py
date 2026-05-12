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

**Robustesse (V2)** :
- Logs INFO par régime et par propriété (avec compteur i/N) pour suivre
  l'avancement en temps réel.
- Mem check via shared.mem_guard avant chaque régime, abort si critique.
- Si aucune Property scope=cross_regime n'est présente, on libère le dump
  immédiatement après traitement → évite l'OOM observé Sprint C v1.
- `progress_callback(regime_key, regime_out)` opt-in pour checkpoint
  granulaire côté Sprint.
"""

from __future__ import annotations

import gc
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from catalog.oracles.base import AbstractOracle, AttentionDump, RegimeSpec
from catalog.properties.base import Property, PropertyContext
from shared.mem_guard import MemoryGuardAbort, check_memory

_LOGGER = logging.getLogger("catalog.battery")


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
        progress_callback: Callable[[Any, dict], None] | None = None,
        min_available_gb: float = 4.0,
    ) -> BatteryResults:
        """Exécute la Battery sur l'Oracle.

        Si `regimes` est None, utilise `oracle.regime_grid()`. Les Properties
        scope=per_regime tournent par régime ; les Properties cross_regime
        reçoivent toutes les attentions collectées en fin de boucle.

        Avec n_workers > 1, les régimes sont dispatchés en parallèle via
        ThreadPoolExecutor. Les Properties cross_regime tournent toujours
        en série après collecte (elles dépendent de tous les dumps).

        Args
        ----
        progress_callback(regime_key, regime_out): hook opt-in appelé après
            chaque régime terminé. Sprint C l'utilise pour checkpointer
            granulairement (résume si crash en milieu de boucle).
        min_available_gb: seuil RAM disponible vérifié AVANT chaque régime.
            Sous le seuil, on abort proprement avant que l'OOM-killer ne
            tue le process silencieusement.
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
        # On ne garde les dumps en RAM que si au moins une Property cross_regime
        # va les utiliser. Sinon on libère après chaque régime → critique pour
        # éviter l'OOM observé Sprint C v1 (5 dumps × ~10 GB).
        keep_all_dumps = bool(cross_regime_props)
        all_dumps: dict[tuple, AttentionDump] = {}
        n_total = len(regimes)
        _LOGGER.info(
            "[battery] run %s : %d régimes × %d properties (per_regime=%d, "
            "per_regime_layers=%d, cross_regime=%d), keep_all_dumps=%s",
            self.name, n_total, len(self.properties),
            len(per_regime_props), len(per_regime_layers_props),
            len(cross_regime_props), keep_all_dumps,
        )

        # --- Per-regime dispatch (séquentiel ou parallèle) ---
        if self.n_workers == 1:
            for idx, regime in enumerate(regimes, start=1):
                _LOGGER.info(
                    "[battery] régime %d/%d : %s", idx, n_total, regime.key,
                )
                try:
                    check_memory(
                        min_available_gb=min_available_gb,
                        label=f"battery régime {idx}/{n_total} {regime.key}",
                        logger=_LOGGER, abort=True,
                    )
                except MemoryGuardAbort as exc:
                    _LOGGER.error(
                        "[battery] abort RAM insuffisante avant régime %s : %s",
                        regime.key, exc,
                    )
                    break
                res = self._process_one_regime(
                    regime, oracle, n_examples_per_regime,
                    per_regime_props, per_regime_layers_props,
                    regime_idx=idx, n_total=n_total,
                )
                if res is None:
                    continue
                regime_key, regime_out, dump = res
                results.per_regime[regime_key] = regime_out
                if keep_all_dumps:
                    all_dumps[regime_key] = dump
                else:
                    # Libère immédiatement — économie ~10 GB par régime SMNIST
                    del dump
                if progress_callback is not None:
                    try:
                        progress_callback(regime_key, regime_out)
                    except Exception as exc:  # noqa: BLE001
                        _LOGGER.warning(
                            "[battery] progress_callback levé %s sur régime %s",
                            type(exc).__name__, regime_key,
                        )
                gc.collect()
        else:
            with ThreadPoolExecutor(max_workers=self.n_workers,
                                    thread_name_prefix="battery") as ex:
                futures = {
                    ex.submit(
                        self._process_one_regime,
                        regime, oracle, n_examples_per_regime,
                        per_regime_props, per_regime_layers_props,
                        regime_idx=i, n_total=n_total,
                    ): regime for i, regime in enumerate(regimes, start=1)
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
                    if keep_all_dumps:
                        all_dumps[regime_key] = dump
                    else:
                        del dump
                    if progress_callback is not None:
                        try:
                            progress_callback(regime_key, regime_out)
                        except Exception as exc:  # noqa: BLE001
                            _LOGGER.warning(
                                "[battery] progress_callback levé %s régime %s",
                                type(exc).__name__, regime_key,
                            )
                    gc.collect()

        # --- Cross-regime properties (toujours séquentiel) ---
        for idx, prop in enumerate(cross_regime_props, start=1):
            _LOGGER.info(
                "[battery] cross_regime %d/%d : %s",
                idx, len(cross_regime_props), prop.name,
            )
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

        _LOGGER.info(
            "[battery] run %s done : %d régimes traités, %d cross_regime "
            "properties calculées",
            self.name, len(results.per_regime), len(results.cross_regime),
        )
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
        *,
        regime_idx: int = 0,
        n_total: int = 0,
    ) -> tuple[tuple, dict, AttentionDump] | None:
        """Traite un régime : extract + computes + return (key, out, dump).

        Retourne None si extract_regime échoue (régime skipé). Thread-safe :
        chaque appel a son propre PropertyContext (cache local).
        """
        tag = f"[regime {regime_idx}/{n_total} {regime.key}]"
        try:
            dump = oracle.extract_regime(regime, n_examples_per_regime)
            dump.validate()
        except Exception as exc:  # noqa: BLE001
            self._log_err(
                f"[battery] {tag} échec extract_regime : "
                f"{type(exc).__name__}: {exc}. SKIP."
            )
            return None

        n_layers = len(dump.attn)
        n_props_per_regime = len(per_regime_props)
        n_props_layers = len(per_regime_layers_props)
        _LOGGER.info(
            "[battery] %s extract OK (n_layers=%d), props per_regime=%d × layers, "
            "props per_regime_layers=%d",
            tag, n_layers, n_props_per_regime, n_props_layers,
        )

        regime_out: dict[str, dict] = {}
        n_props_total = n_props_per_regime * n_layers
        prop_idx = 0
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
                prop_idx += 1
                _LOGGER.debug(
                    "[battery] %s layer=%d prop=%s (%d/%d)",
                    tag, ell, prop.name, prop_idx, n_props_total,
                )
                try:
                    out = prop.compute(A, ctx)
                except Exception as exc:  # noqa: BLE001
                    self._log_err(
                        f"[battery] {tag} layer={ell} prop={prop.name} échec : "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                layered_out = {f"layer{ell}_{k}": v for k, v in out.items()}
                regime_out.setdefault(prop.name, {}).update(layered_out)

        if per_regime_layers_props:
            ctx_layers = PropertyContext(
                device=self.device, dtype=self.dtype,
                regime={
                    "omega": regime.omega, "delta": regime.delta,
                    "entropy": regime.entropy, "n_layers": n_layers,
                },
                metadata={"oracle_id": oracle.oracle_id},
            )
            for i_p, prop in enumerate(per_regime_layers_props, start=1):
                _LOGGER.debug(
                    "[battery] %s per_regime_layers prop=%s (%d/%d)",
                    tag, prop.name, i_p, n_props_layers,
                )
                try:
                    out = prop.compute(dump.attn, ctx_layers)
                except Exception as exc:  # noqa: BLE001
                    self._log_err(
                        f"[battery] {tag} cross-layer prop={prop.name} échec : "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                regime_out[prop.name] = out

        return regime.key, regime_out, dump

    def _log_err(self, msg: str) -> None:
        """Log thread-safe via logger (lock pour éviter interleaving)."""
        with self._stderr_lock:
            _LOGGER.error(msg, exc_info=True)
