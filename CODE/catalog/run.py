"""
run.py — Driver Catalog : lance une Battery sur un Oracle.

Usage :
    PYTHONPATH=CODE uv run python -m catalog.run \\
        --oracle smnist \\
        --level principal \\
        --checkpoint OPS/checkpoints/oracle_e2f0b5e.ckpt \\
        --output OPS/logs/catalog_2026-05-12/ \\
        --n-examples 32

Orchestre :
- Auto-detect machine via infra.MachineProfile (device, dtype, BLAS threads)
- Instancie Oracle selon `--oracle` (smnist | synthetic)
- Compose Battery selon `--level` (minimal | principal | extended | full | research)
- Exécute Battery.run(oracle) avec checkpoint persistance
- Écrit results JSON dans `--output`

Robuste (cf. feedback_script_robustness §1-5) :
- stderr + traceback partout, no silent failures
- Checkpoint atomic via shared.checkpoint
- Caps configurables (--n-examples, --regime-cap, --device)
- Resume automatique si fingerprint compat
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

import catalog.properties  # noqa: F401 — déclenche auto-discovery
from catalog.batteries import (
    Battery,
    BatteryResults,
    level_extended,
    level_full,
    level_minimal,
    level_principal,
    level_research,
)
from catalog.oracles import AbstractOracle, RegimeSpec, SyntheticOracle
from infra.machine import MachineProfile
from shared.checkpoint import Checkpoint


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


LEVEL_FACTORIES = {
    "minimal": level_minimal,
    "principal": level_principal,
    "extended": level_extended,
    "full": level_full,
    "research": level_research,
}


def build_oracle(args: argparse.Namespace) -> AbstractOracle:
    if args.oracle == "smnist":
        if not args.checkpoint:
            raise SystemExit(
                "--checkpoint requis pour --oracle=smnist "
                "(ex: OPS/checkpoints/oracle_e2f0b5e.ckpt)"
            )
        from catalog.oracles.smnist import SMNISTOracle
        return SMNISTOracle(
            checkpoint_path=args.checkpoint,
            device=args.device or "cpu",
        )
    if args.oracle == "synthetic":
        return SyntheticOracle(
            structure=args.synthetic_structure,
            seq_len=args.synthetic_seq_len,
            n_layers=args.synthetic_n_layers,
            n_heads=args.synthetic_n_heads,
            seed=args.synthetic_seed,
        )
    raise SystemExit(f"Oracle inconnu : {args.oracle}")


def build_battery(level: str, profile: MachineProfile) -> Battery:
    factory = LEVEL_FACTORIES.get(level)
    if factory is None:
        raise SystemExit(
            f"Niveau inconnu : {level}. Disponibles : "
            f"{sorted(LEVEL_FACTORIES.keys())}"
        )
    return factory(device=profile.device, dtype=profile.dtype_svd)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="catalog.run",
        description="Lance une Battery de Properties sur un Oracle.",
    )
    parser.add_argument(
        "--oracle", choices=("smnist", "synthetic"), required=True,
    )
    parser.add_argument(
        "--level", choices=tuple(LEVEL_FACTORIES.keys()), default="principal",
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path Oracle checkpoint (requis pour --oracle=smnist)")
    parser.add_argument("--output", type=str, required=True,
                        help="Répertoire de sortie (results.json, state/...)")
    parser.add_argument("--n-examples", type=int, default=32,
                        help="Examples par régime")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cpu | cuda). Défaut : auto MachineProfile.")
    parser.add_argument("--regime-cap", type=int, default=None,
                        help="Limite le nombre de régimes (debug/smoke)")
    # Synthetic-only
    parser.add_argument("--synthetic-structure", default="random",
                        choices=("random", "low_rank", "toeplitz", "hankel"))
    parser.add_argument("--synthetic-seq-len", type=int, default=16)
    parser.add_argument("--synthetic-n-layers", type=int, default=2)
    parser.add_argument("--synthetic-n-heads", type=int, default=4)
    parser.add_argument("--synthetic-seed", type=int, default=0)
    # MLflow
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Désactive MLflow logging (override default opt-in)")
    parser.add_argument("--mlflow-sprint", type=int, default=1)
    parser.add_argument("--mlflow-status", default="exploratory",
                        choices=("exploratory", "pre_registered", "invalidated"))

    args = parser.parse_args()

    # --- Machine profile ---
    profile = MachineProfile.detect()
    if args.device:
        # Override explicite : on construit un profile partiel
        profile = MachineProfile.fake(
            device=args.device,  # type: ignore[arg-type]
            gpu_arch=profile.gpu_arch, precision_svd=profile.precision_svd,
            n_blas_threads=profile.n_blas_threads, batch_cap=profile.batch_cap,
        )
    profile.apply_blas_env()
    print(f"=== {profile.summary()} ===", flush=True)

    # --- Output dir + checkpoint ---
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = output_dir / "state"
    fingerprint = {
        "oracle": args.oracle,
        "level": args.level,
        "n_examples": args.n_examples,
        "device": profile.device,
        "precision": profile.precision_svd,
    }
    cp, resumed = Checkpoint.create_or_resume(state_dir, fingerprint=fingerprint)
    print(f"=== Checkpoint: state_dir={state_dir} resumed={resumed} ===", flush=True)

    # --- Build Oracle + Battery ---
    oracle = build_oracle(args)
    battery = build_battery(args.level, profile)
    print(
        f"=== Oracle: {oracle} | Battery: {battery.name} "
        f"({len(battery.properties)} properties) ===",
        flush=True,
    )
    print("Properties :")
    for p in battery.properties:
        print(f"  - {p.name:35s} family={p.family} cost={p.cost_class}")

    # --- Skip si déjà run ---
    if cp.has("battery_results"):
        print("[checkpoint] SKIP run (battery_results déjà calculé)", flush=True)
        results = cp.load("battery_results")
    else:
        regimes = oracle.regime_grid()
        if args.regime_cap is not None:
            regimes = regimes[: args.regime_cap]
            print(f"=== Régimes cap à {len(regimes)} ===", flush=True)

        t0 = time.perf_counter()
        try:
            results = battery.run(
                oracle,
                regimes=regimes,
                n_examples_per_regime=args.n_examples,
            )
        except Exception as exc:  # noqa: BLE001
            _stderr(f"[catalog.run] battery.run a échoué : {type(exc).__name__}: {exc}")
            traceback.print_exc(file=sys.stderr, limit=10)
            sys.exit(1)
        dt = time.perf_counter() - t0
        print(f"=== Battery.run terminé en {dt:.1f}s ===", flush=True)
        cp.save("battery_results", results)

    # --- Sortie JSON ---
    results_json = output_dir / "results.json"
    with open(results_json, "w") as f:
        json.dump(results.to_dict(), f, indent=2, default=str)
    print(f"=== Résultats écrits : {results_json} ===", flush=True)

    # --- MLflow logging (opt-in si MLFLOW_TRACKING_URI défini) ---
    if not args.no_mlflow:
        from catalog.mlflow_logger import log_battery_results
        log_battery_results(
            results,
            output_dir=output_dir,
            run_name=f"{args.oracle}_{args.level}_{args.n_examples}ex",
            sprint=args.mlflow_sprint,
            domain=args.oracle,
            status=args.mlflow_status,
        )

    # --- Résumé console ---
    n_regimes = len(results.per_regime)
    n_props = len(results.metadata.get("properties", []))
    print(
        f"\n=== Récap : {n_regimes} régimes × {n_props} properties — "
        f"{results.metadata.get('battery_name')} sur {results.metadata.get('oracle_id')} ==="
    )


if __name__ == "__main__":
    main()
