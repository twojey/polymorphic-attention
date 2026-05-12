"""
cross_oracle.py — Harness cross-Oracle (SMNIST × LL × Vision × Code).

Lance la même Battery sur plusieurs Oracles et compare les signatures.
Cœur du livrable scientifique Partie 1 — "Mathematical Signatures of
Attention" cross-domain.

Pipeline :
1. Pour chaque Oracle dans la liste :
   - Construire l'Oracle (lazy)
   - Lancer Battery(level)
   - Sauvegarder BatteryResults
2. Comparer les résultats par Property (corrélations, distances de
   signatures, ranking par Oracle)
3. Sortie : JSON avec matrice (Property × Oracle) + analyse comparative

Cohérent feedback_script_robustness : chaque Oracle peut échouer
indépendamment (skip + log), checkpoint atomic par Oracle.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

from infra.machine import MachineProfile
from shared.checkpoint import Checkpoint


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def build_oracle_from_spec(spec: dict[str, Any]):
    """Construit un Oracle à partir de sa spec (dict).

    spec = {
      "kind": "smnist" | "synthetic" | "ll" | "vision" | "code",
      "checkpoint": "path/to/.ckpt" (si applicable),
      ...
    }
    """
    kind = spec["kind"]
    if kind == "synthetic":
        from catalog.oracles import SyntheticOracle
        return SyntheticOracle(
            structure=spec.get("structure", "random"),
            seq_len=spec.get("seq_len", 16),
            n_layers=spec.get("n_layers", 2),
            n_heads=spec.get("n_heads", 4),
            seed=spec.get("seed", 0),
        )
    if kind == "smnist":
        from catalog.oracles import SMNISTOracle
        return SMNISTOracle(
            checkpoint_path=spec["checkpoint"],
            device=spec.get("device", "cpu"),
        )
    if kind == "ll":
        from catalog.oracles import LLOracle, LLModelSpec
        ms = spec.get("model_spec", {})
        return LLOracle(
            checkpoint_path=spec["checkpoint"],
            model_spec=LLModelSpec(**ms) if ms else None,  # type: ignore
            device=spec.get("device", "cpu"),
        )
    if kind == "vision":
        from catalog.oracles.vision import VisionOracle
        return VisionOracle(
            checkpoint_path=spec["checkpoint"],
            device=spec.get("device", "cpu"),
        )
    if kind == "code":
        from catalog.oracles.code import CodeOracle
        return CodeOracle(
            checkpoint_path=spec["checkpoint"],
            device=spec.get("device", "cpu"),
        )
    raise ValueError(f"Oracle kind inconnu : {kind}")


def compare_signatures(
    results_by_oracle: dict[str, "BatteryResults"],
    *,
    property_summary_key: str = "median",
) -> dict[str, Any]:
    """Compare les signatures Property × Oracle.

    Pour chaque Property × régime, agrège un scalaire (médiane d'une métrique
    'representative') et construit une matrice de comparaison.
    """
    if not results_by_oracle:
        return {}

    # Trouver l'union de Properties et de régimes représentés
    properties_set: set[str] = set()
    regimes_set: set[Any] = set()
    for results in results_by_oracle.values():
        for regime_key, regime_out in results.per_regime.items():
            regimes_set.add(regime_key)
            for prop_name in regime_out.keys():
                properties_set.add(prop_name)

    matrix: dict[str, dict[str, dict[str, float]]] = {}
    for oracle_name, results in results_by_oracle.items():
        matrix[oracle_name] = {}
        for prop_name in properties_set:
            # Pour chaque (oracle, property), agréger sur les régimes
            scalars: list[float] = []
            for regime_key, regime_out in results.per_regime.items():
                prop_out = regime_out.get(prop_name, {})
                # Chercher la clé qui contient property_summary_key
                for k, v in prop_out.items():
                    if property_summary_key in k and isinstance(v, (int, float)):
                        scalars.append(float(v))
                        break
            if scalars:
                matrix[oracle_name][prop_name] = {
                    "median": float(sum(scalars) / len(scalars)),
                    "n_regimes": len(scalars),
                }

    return {
        "n_oracles": len(results_by_oracle),
        "n_properties_compared": len(properties_set),
        "n_regimes_total": len(regimes_set),
        "matrix": matrix,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="catalog.cross_oracle",
        description="Lance Battery sur plusieurs Oracles + comparaison signatures.",
    )
    parser.add_argument("--oracles-spec", type=str, required=True,
                        help="Path JSON spec liste d'Oracles à comparer")
    parser.add_argument("--level", type=str, default="principal",
                        choices=("minimal", "principal", "extended", "full", "research"))
    parser.add_argument("--n-examples", type=int, default=32)
    parser.add_argument("--regime-cap", type=int, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    profile = MachineProfile.detect()
    if args.device:
        profile = MachineProfile.fake(
            device=args.device, gpu_arch=profile.gpu_arch,
            precision_svd=profile.precision_svd,
            n_blas_threads=profile.n_blas_threads, batch_cap=profile.batch_cap,
        )
    profile.apply_blas_env()
    print(f"=== {profile.summary()} ===", flush=True)

    with open(args.oracles_spec) as f:
        specs = json.load(f)
    if not isinstance(specs, list) or not specs:
        raise SystemExit("oracles-spec doit être une liste JSON non-vide")
    print(f"=== {len(specs)} Oracles à évaluer ===", flush=True)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        "level": args.level,
        "n_examples": args.n_examples,
        "n_oracles": len(specs),
        "oracle_names": [s.get("name", s["kind"]) for s in specs],
    }
    cp, resumed = Checkpoint.create_or_resume(
        output_dir / "state", fingerprint=fingerprint
    )
    print(f"=== Checkpoint: resumed={resumed} ===", flush=True)

    # Composer Battery (lazy import pour éviter side effects)
    from catalog.batteries import (
        level_minimal, level_principal, level_extended, level_full, level_research,
    )
    LEVELS = {
        "minimal": level_minimal, "principal": level_principal,
        "extended": level_extended, "full": level_full, "research": level_research,
    }
    battery_factory = LEVELS[args.level]
    battery = battery_factory(device=profile.device, dtype=profile.dtype_svd)
    print(
        f"=== Battery: {args.level} ({len(battery.properties)} properties) ===",
        flush=True,
    )

    # Lancer sur chaque Oracle
    results_by_oracle: dict[str, Any] = {}
    for spec in specs:
        oracle_name = spec.get("name", spec["kind"])
        cp_key = f"oracle_{oracle_name}_results"
        if cp.has(cp_key):
            print(f"[checkpoint] SKIP Oracle {oracle_name}", flush=True)
            results_by_oracle[oracle_name] = cp.load(cp_key)
            continue

        print(f"\n--- Oracle {oracle_name} ({spec['kind']}) ---", flush=True)
        try:
            oracle = build_oracle_from_spec(spec)
            regimes = oracle.regime_grid()
            if args.regime_cap is not None:
                regimes = regimes[: args.regime_cap]
            results = battery.run(
                oracle, regimes=regimes, n_examples_per_regime=args.n_examples,
            )
            cp.save(cp_key, results)
            results_by_oracle[oracle_name] = results
            print(f"  ✓ {oracle_name} terminé ({len(results.per_regime)} régimes)", flush=True)
        except Exception as exc:  # noqa: BLE001
            _stderr(f"[cross_oracle] {oracle_name} a échoué : {type(exc).__name__}: {exc}")
            traceback.print_exc(file=sys.stderr, limit=5)
            continue

    # Comparaison signatures
    comparison = compare_signatures(results_by_oracle)
    comparison_json = output_dir / "comparison.json"
    with open(comparison_json, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\n=== Comparaison écrite : {comparison_json} ===", flush=True)

    # Per-oracle results
    for oracle_name, results in results_by_oracle.items():
        if hasattr(results, "to_dict"):
            per_oracle_json = output_dir / f"results_{oracle_name}.json"
            with open(per_oracle_json, "w") as f:
                json.dump(results.to_dict(), f, indent=2, default=str)

    print(
        f"=== Récap : {len(results_by_oracle)} / {len(specs)} Oracles évalués ===",
        flush=True,
    )


if __name__ == "__main__":
    main()
