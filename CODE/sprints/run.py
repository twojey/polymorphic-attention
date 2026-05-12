"""
run.py — CLI dispatcher Sprint.

Usage :
    python -m sprints.run --sprint B --output OPS/logs/sprint_B
    python -m sprints.run --sprint C --output OPS/logs/sprint_C \\
        --dumps-dir OPS/logs/sprint_B/dumps
    python -m sprints.run --sprint S5 --use-hf-dinov2 --device cuda

Sprints disponibles : B, C, D, E, F, G, S4, S5, S6, S7.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sprints.sprint_b_re_extract import SprintBReExtract
from sprints.sprint_c_catalog_full import SprintCCatalogFull
from sprints.sprint_d_phase3_v3 import SprintDPhase3V3
from sprints.sprint_e_phase4_warmup import SprintEPhase4Warmup
from sprints.sprint_f_phase4_autonomous import SprintFPhase4Autonomous
from sprints.sprint_g_phase5_validation import SprintGPhase5Validation
from sprints.sprint_s4_smnist_extended import SprintS4SMNISTExtended
from sprints.sprint_s5_vision import SprintS5Vision
from sprints.sprint_s6_code import SprintS6Code
from sprints.sprint_s7_ll import SprintS7LL


REGISTRY = {
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


def main() -> None:
    parser = argparse.ArgumentParser(prog="sprints.run",
                                     description="CLI Sprint dispatcher")
    parser.add_argument("--sprint", required=True, choices=sorted(REGISTRY.keys()))
    parser.add_argument("--output", required=True, help="Output dir")
    parser.add_argument("--mlflow-uri", default=None)
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument("--use-hf-dinov2", action="store_true")
    parser.add_argument("--use-hf-llama", action="store_true")
    parser.add_argument("--use-hf-starcoder", action="store_true")
    parser.add_argument("--hf-model", default=None)
    args = parser.parse_args()

    cls = REGISTRY[args.sprint]
    kwargs = dict(
        output_dir=args.output,
        mlflow_uri=args.mlflow_uri,
        seed=args.seed,
    )

    # Map sprint-specific args
    sprint_args = _build_sprint_args(args)
    sprint = cls(**kwargs, **sprint_args)
    result = sprint.run()
    if result.status.value != "success":
        sys.exit(1)


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
    main()
