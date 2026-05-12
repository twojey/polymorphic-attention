"""
run.py — Driver phase 4 : entraînement Spectrometer + ASPLayer concret.

Spec : DOC/04_phase_routage_budget.md + ROADMAP §3.5.

Pipeline :
1. Charger Oracle phase 1 + signaux retenus phase 1.5 (config)
2. Instancier Spectrometer(input_dim = L × n_signaux retenus)
3. Phase 4a : warm-up FrozenAlphaSpectrometer (α=1) + distillation V3.5
   asymétrique (p75 percentile r_eff Oracle)
4. Phase 4b : λ_distil → 0, λ_budget · L_sparsity + L_task seul
5. Construire Diagramme de Phase + Pareto λ

Driver V1 : squelette structuré, le training inner loop nécessite
intégration avec phase 3 (ASPTransformer concret) — à compléter dans
Sprint D quand les signaux retenus phase 1.5 sont actés.

Cohérent feedback_script_robustness :
- stderr + traceback
- shared.checkpoint atomic
- mlflow opt-in
- caps configurables (epochs, steps_per_epoch)
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import torch

from infra.machine import MachineProfile
from shared.checkpoint import Checkpoint


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="phase4_routage_budget.run",
        description="Training Spectrometer + ASPLayer phase 4a/4b.",
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path config Hydra (ex: OPS/configs/phase4/spectrometer.yaml)")
    parser.add_argument("--oracle-checkpoint", type=str, required=True,
                        help="Path checkpoint Oracle phase 1 (.ckpt)")
    parser.add_argument("--signals", type=str, required=True,
                        help="Liste signaux retenus phase 1.5, ex: 'S_Spectral,S_KL'")
    parser.add_argument("--output", type=str, required=True,
                        help="Répertoire de sortie (state/, results.json, models/)")
    parser.add_argument("--max-epochs-4a", type=int, default=10,
                        help="Cap epochs phase 4a warm-up")
    parser.add_argument("--max-epochs-4b", type=int, default=20,
                        help="Cap epochs phase 4b learning autonome")
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

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        "config": args.config,
        "oracle_checkpoint": str(args.oracle_checkpoint),
        "signals": args.signals,
        "max_epochs_4a": args.max_epochs_4a,
        "max_epochs_4b": args.max_epochs_4b,
        "device": profile.device,
    }
    cp, resumed = Checkpoint.create_or_resume(
        output_dir / "state", fingerprint=fingerprint
    )
    print(f"=== Checkpoint: resumed={resumed} ===", flush=True)

    # === Squelette d'orchestration phase 4 ===
    # Cette section est l'entry point qui prépare tous les composants ;
    # le training inner loop est laissé à compléter Sprint D, quand l'API
    # ASPLayer concrète + Spectrometer concret + DataLoader signaux retenus
    # sont stabilisés.
    signals_list = [s.strip() for s in args.signals.split(",") if s.strip()]

    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(args.config)
        print(f"=== Config chargée : {args.config} ===", flush=True)

        # Phase 4a — Warm-up (FrozenAlphaSpectrometer + distillation V3.5 p75)
        if cp.has("phase_4a_done"):
            print("[checkpoint] SKIP phase 4a", flush=True)
        else:
            print(
                f"=== Phase 4a (warm-up, α=1, distillation p75) — "
                f"max_epochs={args.max_epochs_4a} ===", flush=True
            )
            print(
                "  [TODO Sprint D] Implémenter inner loop avec :\n"
                "    - FrozenAlphaSpectrometer + ASPLayer (R_max depuis cfg)\n"
                "    - Loss = L_task + λ_distil · L_distill_p75_asymmetric\n"
                "    - Curriculum 3 étages (cfg.curriculum.stages)\n"
                "    - TransitionMonitor pour 4a → 4b\n"
                "  Composants déjà disponibles : Spectrometer, FrozenAlphaSpectrometer,\n"
                "  asymmetric_distillation_loss, default_curriculum, CurriculumScheduler.",
                flush=True,
            )
            cp.save("phase_4a_done", {"status": "skeleton", "epochs": 0})

        # Phase 4b — Autonome (λ_distil=0, λ_budget · L_sparsity + L_task)
        if cp.has("phase_4b_done"):
            print("[checkpoint] SKIP phase 4b", flush=True)
        else:
            print(
                f"=== Phase 4b (autonome, λ_distil=0) — "
                f"max_epochs={args.max_epochs_4b} ===", flush=True
            )
            print(
                "  [TODO Sprint D] Inner loop :\n"
                "    - Spectrometer learnable (déboucle des warmup weights)\n"
                "    - Sweep λ_budget : 7 valeurs log-spaced (cfg.lambda_budget)\n"
                "    - Mesure dérive aux N premiers steps\n"
                "  Sortie attendue : Diagramme de Phase + Pareto λ.",
                flush=True,
            )
            cp.save("phase_4b_done", {"status": "skeleton", "epochs": 0})

        # Diagramme + Pareto
        if cp.has("diagrams_done"):
            print("[checkpoint] SKIP diagrammes", flush=True)
        else:
            from phase4_routage_budget.diagram_phase import (
                build_phase_diagram, is_phase_diagram_increasing,
            )
            print("=== Diagramme + Pareto λ ===", flush=True)
            print(
                "  [TODO Sprint D] Appeler build_phase_diagram(R_target_per_lambda)\n"
                "  + build_pareto_curve sur résultats phase 4b.",
                flush=True,
            )
            cp.save("diagrams_done", {"status": "skeleton"})

    except Exception as exc:  # noqa: BLE001
        _stderr(f"[phase4.run] échec : {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stderr, limit=10)
        sys.exit(1)

    print(f"=== Phase 4 squelette terminé — output: {output_dir} ===", flush=True)


if __name__ == "__main__":
    main()
