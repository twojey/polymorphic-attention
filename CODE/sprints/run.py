"""
run.py — CLI dispatcher Sprint.

Usage :
    python -m sprints.run --sprint B --output OPS/logs/sprint_B
    python -m sprints.run --sprint C --output OPS/logs/sprint_C \\
        --dumps-dir OPS/logs/sprint_B/dumps
    python -m sprints.run --sprint S5 --use-hf-dinov2 --device cuda

Sprints disponibles : B, C, D, E, F, G, S4, S5, S6, S7.

Robustesse top-level :
- Toute exception non-attrapée est loggée via logger.exception puis exit 2.
- Signaux SIGTERM/SIGINT (incl. OOM-killer, runpod stop) loggués avant exit.
- Pré-condition mémoire vérifiée avant de lancer le Sprint.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import traceback
from pathlib import Path

# Bootstrap logging avant tout import lourd : si l'import explose, on a
# au moins une trace lisible sur stderr.
_BOOTSTRAP_FORMAT = "%(asctime)sZ [%(levelname)s] %(name)s :: %(message)s"
logging.basicConfig(level=logging.INFO, format=_BOOTSTRAP_FORMAT,
                    stream=sys.stderr, force=True)
logging.Formatter.converter = __import__("time").gmtime
_LOGGER = logging.getLogger("sprints.run")


def _install_signal_handlers() -> None:
    """Loggue SIGTERM/SIGINT avec stack trace avant de quitter.

    Sans ça, un kill (RunPod stop, OOM-killer) tue le process sans rien
    écrire dans sprint.log → diagnostic impossible.
    """

    def _handler(signum: int, frame) -> None:
        name = signal.Signals(signum).name
        _LOGGER.error("[sprints.run] signal reçu : %s (signum=%d)", name, signum)
        stack = "".join(traceback.format_stack(frame))
        _LOGGER.error("[sprints.run] stack au moment du signal :\n%s", stack)
        # Exit code 128 + signum (convention Unix)
        sys.exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _handler)
        except (OSError, ValueError):
            # SIGHUP indisponible sur certaines plateformes
            pass


def _precheck_memory(threshold_gb: float = 2.0) -> None:
    """Vérifie qu'il y a au moins `threshold_gb` GB de RAM dispo avant de
    lancer le Sprint. Loggue WARNING sinon (n'abort pas — laisse le Sprint
    decider de son propre seuil)."""
    try:
        from shared.mem_guard import check_memory  # noqa: PLC0415

        check_memory(
            min_available_gb=threshold_gb,
            label="sprint-precheck",
            logger=_LOGGER,
            abort=False,
        )
    except Exception as e:  # noqa: BLE001
        _LOGGER.warning("[sprints.run] precheck memory KO (non-fatal) : %s", e)


def _import_sprints() -> dict:
    """Import des classes Sprint après le bootstrap logging.

    Si un import explose, on loggue ET on remonte (sortie 3). Avant ce
    patch, l'erreur d'import remontait silencieusement en cas de stderr
    redirigé.
    """
    from sprints.sprint_b_re_extract import SprintBReExtract  # noqa: PLC0415
    from sprints.sprint_c_catalog_full import SprintCCatalogFull  # noqa: PLC0415
    from sprints.sprint_d_phase3_v3 import SprintDPhase3V3  # noqa: PLC0415
    from sprints.sprint_e_phase4_warmup import SprintEPhase4Warmup  # noqa: PLC0415
    from sprints.sprint_f_phase4_autonomous import SprintFPhase4Autonomous  # noqa: PLC0415
    from sprints.sprint_g_phase5_validation import SprintGPhase5Validation  # noqa: PLC0415
    from sprints.sprint_s4_smnist_extended import SprintS4SMNISTExtended  # noqa: PLC0415
    from sprints.sprint_s5_vision import SprintS5Vision  # noqa: PLC0415
    from sprints.sprint_s6_code import SprintS6Code  # noqa: PLC0415
    from sprints.sprint_s7_ll import SprintS7LL  # noqa: PLC0415

    return {
        "B": SprintBReExtract,
        "C": SprintCCatalogFull,
        "D": SprintDPhase3V3,
        "E": SprintEPhase4Warmup,
        "F": SprintFPhase4Autonomous,
        "G": SprintGPhase5Validation,
        "S4": SprintS4SMNISTExtended,
        "S5": SprintS5Vision,
        "S6": SprintS6Code,
        "S7": SprintS7LL,
    }


def main() -> int:
    _install_signal_handlers()

    parser = argparse.ArgumentParser(prog="sprints.run",
                                     description="CLI Sprint dispatcher")
    parser.add_argument("--sprint", required=True)
    parser.add_argument("--output", required=True, help="Output dir")
    parser.add_argument("--mlflow-uri", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mem-precheck-gb", type=float, default=2.0,
                        help="RAM dispo minimale (GB) pour démarrer le Sprint")
    # Sprint-specific (optional)
    parser.add_argument("--oracle-checkpoint", default=None,
                       help="Sprint B/S4/D/S5/S7")
    parser.add_argument("--dumps-dir", default=None, help="Sprint C")
    parser.add_argument("--sprint-c-report", default=None, help="Sprint D")
    parser.add_argument("--backbone-class", default=None, help="Sprint D override")
    parser.add_argument("--sprint-d-checkpoint", default=None, help="Sprint E")
    parser.add_argument("--sprint-e-checkpoint", default=None, help="Sprint F")
    parser.add_argument("--sprint-f-checkpoint", default=None, help="Sprint G")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-examples", type=int, default=None)
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Sprint C : workers Battery parallèle (défaut 1)")
    parser.add_argument("--use-hf-dinov2", action="store_true")
    parser.add_argument("--use-hf-llama", action="store_true")
    parser.add_argument("--use-hf-starcoder", action="store_true")
    parser.add_argument("--hf-model", default=None)
    args = parser.parse_args()

    _LOGGER.info("[sprints.run] args : %s", vars(args))
    _precheck_memory(threshold_gb=args.mem_precheck_gb)

    try:
        registry = _import_sprints()
    except Exception as exc:  # noqa: BLE001
        _LOGGER.exception("[sprints.run] échec import Sprint registry : %s", exc)
        return 3

    if args.sprint not in registry:
        _LOGGER.error("[sprints.run] sprint inconnu : %s (dispo: %s)",
                      args.sprint, sorted(registry))
        return 1

    cls = registry[args.sprint]
    kwargs = dict(
        output_dir=args.output,
        mlflow_uri=args.mlflow_uri,
        seed=args.seed,
    )
    sprint_args = _build_sprint_args(args)

    try:
        sprint = cls(**kwargs, **sprint_args)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.exception("[sprints.run] échec construction Sprint %s : %s",
                          args.sprint, exc)
        return 4

    try:
        result = sprint.run()
    except MemoryError as exc:
        _LOGGER.exception("[sprints.run] MemoryError attrapée : %s", exc)
        return 137  # Convention OOM Linux
    except Exception as exc:  # noqa: BLE001
        # SprintBase.run() attrape déjà, mais filet de sécurité.
        _LOGGER.exception("[sprints.run] exception non-attrapée : %s", exc)
        return 5

    status = result.status.value
    _LOGGER.info("[sprints.run] Sprint %s terminé : %s (%.1fs)",
                 args.sprint, status, result.duration_seconds)
    if status == "success":
        return 0
    if status == "skipped":
        return 0  # skip n'est pas un échec
    return 1


def _build_sprint_args(args: argparse.Namespace) -> dict:
    """Construit le kwargs spécifique au Sprint sélectionné."""
    s = args.sprint
    out: dict = {"device": args.device}
    if s == "B":
        if args.oracle_checkpoint:
            out["oracle_checkpoint"] = args.oracle_checkpoint
        if args.n_examples:
            out["n_examples_per_regime"] = args.n_examples
    elif s == "C":
        if args.dumps_dir:
            out["dumps_dir"] = args.dumps_dir
        if args.n_examples:
            out["n_examples_per_regime"] = args.n_examples
        if args.n_workers is not None:
            out["n_workers"] = args.n_workers
    elif s == "D":
        if args.sprint_c_report:
            out["sprint_c_report"] = args.sprint_c_report
        if args.backbone_class:
            out["backbone_class_override"] = args.backbone_class
    elif s == "E":
        if args.sprint_d_checkpoint:
            out["sprint_d_checkpoint"] = args.sprint_d_checkpoint
    elif s == "F":
        if args.sprint_e_checkpoint:
            out["sprint_e_checkpoint"] = args.sprint_e_checkpoint
    elif s == "G":
        if args.sprint_f_checkpoint:
            out["sprint_f_checkpoint"] = args.sprint_f_checkpoint
    elif s == "S4":
        if args.oracle_checkpoint:
            out["oracle_checkpoint"] = args.oracle_checkpoint
    elif s == "S5":
        out["use_hf_dinov2"] = args.use_hf_dinov2
        if args.hf_model:
            out["hf_model_name"] = args.hf_model
    elif s == "S6":
        out["use_hf_starcoder"] = args.use_hf_starcoder
        if args.hf_model:
            out["hf_model_name"] = args.hf_model
    elif s == "S7":
        out["use_hf_llama"] = args.use_hf_llama
        if args.hf_model:
            out["hf_model_name"] = args.hf_model
        if args.oracle_checkpoint:
            out["local_checkpoint"] = args.oracle_checkpoint
    return out


if __name__ == "__main__":
    sys.exit(main())
