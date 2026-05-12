"""
run.py — Driver phase 5 : dispatcher tests 5a-5e + 6c sur ASPLayer entraîné.

Spec : DOC/05_phase_pareto.md.

Lance séquentiellement (avec checkpoint reprise) :
- 5a Identifiabilité (anti-fraud + différentiel)
- 5b Élasticité
- 5c SE/HT (semi-empirique / heavy-tail)
- 5e OOD (out-of-distribution)
- 6c R_max = r_med / 2 (test final)

Chaque test produit un AntiFraudResult / ElasticityResult / etc. ; le
driver agrège en un JSON final + log MLflow.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

from infra.machine import MachineProfile
from shared.checkpoint import Checkpoint


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


TESTS_AVAILABLE = ("5a", "5b", "5c", "5e", "6c")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="phase5_pareto.run",
        description="Dispatcher tests phase 5 identifiabilité / élasticité / OOD.",
    )
    parser.add_argument("--asp-checkpoint", type=str, required=True,
                        help="Path checkpoint ASPLayer entraîné phase 4")
    parser.add_argument("--oracle-checkpoint", type=str, required=True,
                        help="Path checkpoint Oracle phase 1")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tests", type=str, default=",".join(TESTS_AVAILABLE),
                        help=f"Liste tests à lancer (défaut : tous). Options : {TESTS_AVAILABLE}")
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

    requested = [t.strip() for t in args.tests.split(",") if t.strip()]
    unknown = set(requested) - set(TESTS_AVAILABLE)
    if unknown:
        _stderr(f"Tests inconnus : {unknown}. Options : {TESTS_AVAILABLE}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        "asp_checkpoint": str(args.asp_checkpoint),
        "oracle_checkpoint": str(args.oracle_checkpoint),
        "tests": sorted(requested),
        "device": profile.device,
    }
    cp, resumed = Checkpoint.create_or_resume(
        output_dir / "state", fingerprint=fingerprint
    )
    print(f"=== Checkpoint: resumed={resumed}, tests = {requested} ===", flush=True)

    all_results: dict[str, dict] = {}

    # Dispatch tests
    test_runners = {
        "5a": "_run_test_5a",
        "5b": "_run_test_5b",
        "5c": "_run_test_5c",
        "5e": "_run_test_5e",
        "6c": "_run_test_6c",
    }

    for test_id in requested:
        cp_key = f"test_{test_id}_done"
        if cp.has(cp_key):
            print(f"[checkpoint] SKIP test {test_id}", flush=True)
            all_results[test_id] = cp.load(cp_key)
            continue
        print(f"=== Test {test_id} ===", flush=True)
        runner_name = test_runners[test_id]
        try:
            runner = globals()[runner_name]
            result = runner(args, profile)
            cp.save(cp_key, result)
            all_results[test_id] = result
        except Exception as exc:  # noqa: BLE001
            _stderr(f"[phase5.run] test {test_id} a échoué : {type(exc).__name__}: {exc}")
            traceback.print_exc(file=sys.stderr, limit=10)
            all_results[test_id] = {"status": "failed", "error": str(exc)}

    # Sortie JSON final
    results_json = output_dir / "results.json"
    with open(results_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"=== Résultats écrits : {results_json} ===", flush=True)

    # MLflow (opt-in)
    if not args.no_mlflow:
        try:
            from catalog.mlflow_logger import is_mlflow_active
            if is_mlflow_active():
                _log_to_mlflow(all_results, output_dir, args)
        except ImportError:
            pass

    print(f"=== Récap : {len(all_results)} tests exécutés ===", flush=True)


def _run_test_5a(args, profile) -> dict:
    """Test 5a : Identifiabilité (anti-fraud + différentiel). Squelette."""
    from phase5_pareto.test_5a_identifiability import run_anti_fraud
    print(
        "  [TODO Sprint E] Charger ASPLayer + Oracle, "
        "construire datasets (noise, real, distillé), appeler run_anti_fraud.",
        flush=True,
    )
    return {"status": "skeleton", "test": "5a"}


def _run_test_5b(args, profile) -> dict:
    """Test 5b : Élasticité r_max-shift. Squelette."""
    print("  [TODO Sprint E] phase5_pareto.test_5b_elasticity.run_elasticity_test", flush=True)
    return {"status": "skeleton", "test": "5b"}


def _run_test_5c(args, profile) -> dict:
    """Test 5c : SE/HT (semi-empirique / heavy-tail). Squelette."""
    print("  [TODO Sprint E] phase5_pareto.test_5c_se_ht", flush=True)
    return {"status": "skeleton", "test": "5c"}


def _run_test_5e(args, profile) -> dict:
    """Test 5e : OOD. Squelette."""
    print("  [TODO Sprint E] phase5_pareto.test_5e_ood", flush=True)
    return {"status": "skeleton", "test": "5e"}


def _run_test_6c(args, profile) -> dict:
    """Test 6c : R_max = r_med / 2 final. Squelette."""
    print("  [TODO Sprint E] phase5_pareto.test_6c_rmax_half", flush=True)
    return {"status": "skeleton", "test": "6c"}


def _log_to_mlflow(results: dict, output_dir: Path, args) -> None:
    """Loggue les verdicts agrégés à MLflow."""
    try:
        import mlflow
        from shared.mlflow_helpers import start_run
        with start_run(
            experiment="phase5",
            run_name=f"phase5_{Path(args.asp_checkpoint).stem}",
            phase="5", sprint=4, domain="smnist",
            status="exploratory",
        ):
            mlflow.log_params({
                "n_tests": len(results),
                "tests_list": ",".join(results.keys()),
                "asp_checkpoint": Path(args.asp_checkpoint).stem,
            })
            mlflow.log_artifact(str(output_dir / "results.json"))
    except Exception as exc:  # noqa: BLE001
        _stderr(f"[phase5.run] MLflow log échec (non-bloquant) : {exc}")


if __name__ == "__main__":
    main()
