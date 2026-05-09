"""
run_distillability.py — driver phase 1.5b (Distillabilité de S_Spectral).

Spec : DOC/01b §5.

Pipeline :
1. Charger embeddings tokens + S_Spectral teacher (calculés en run.py et
   sauvés comme MLflow artifact, ou recalculés ici).
2. Train/eval split.
3. Entraîner StudentMLP, mesurer ρ et MSE relative.
4. Verdict ρ > 0.85 ET MSE_rel < seuil.

Usage : `PYTHONPATH=CODE uv run python -m phase1b_calibration_signal.run_distillability ...`
"""

from __future__ import annotations

import time
from pathlib import Path

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf

from phase1b_calibration_signal.bench.distillability import train_student
from shared.mlflow_helpers import log_yaml_config, start_run
from shared.runner import finalize_manifest, make_manifest, write_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]


@hydra.main(version_base=None, config_path="../../OPS/configs/phase1b", config_name="signals")
def main(cfg: DictConfig) -> None:
    """Driver distillabilité.

    Sur le pod : récupère les embeddings + S_Spectral teacher depuis MLflow
    artifacts produits par run.py phase 1.5. Pour V1, on passe les chemins
    via override Hydra.
    """
    print(OmegaConf.to_yaml(cfg))

    embeddings_path = OmegaConf.select(cfg, "embeddings_artifact")
    teacher_path = OmegaConf.select(cfg, "teacher_artifact")
    if embeddings_path is None or teacher_path is None:
        raise SystemExit(
            "Override requis : `embeddings_artifact=...` `teacher_artifact=...` "
            "(produits par phase1b_calibration_signal.run)"
        )

    embeddings = torch.load(str(embeddings_path), map_location="cpu", weights_only=False)
    teacher = torch.load(str(teacher_path), map_location="cpu", weights_only=False)
    assert embeddings.ndim == 2 and embeddings.size(0) == teacher.size(0), \
        "Mismatch entre embeddings (N, d) et teacher (N,)"

    n_total = embeddings.size(0)
    n_train = int(0.8 * n_total)
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(cfg.bench.seed))
    train_idx, eval_idx = perm[:n_train], perm[n_train:]

    manifest = make_manifest(
        phase="1.5", sprint=cfg.sprint, domain=cfg.domain, seed=cfg.bench.seed,
        config_path="OPS/configs/phase1b/signals.yaml",
        short_description="distillab",
        repo=REPO_ROOT,
    )
    write_manifest(manifest, REPO_ROOT)

    t0 = time.perf_counter()
    with start_run(
        experiment="phase1.5",
        run_name=manifest.run_id,
        phase="1.5", sprint=cfg.sprint, domain=cfg.domain,
        status=manifest.status,
        extra_tags={"subphase": "1.5b"},
    ):
        log_yaml_config(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
        result = train_student(
            embeddings_train=embeddings[train_idx],
            targets_train=teacher[train_idx],
            embeddings_eval=embeddings[eval_idx],
            targets_eval=teacher[eval_idx],
            d_model=embeddings.size(1),
            hidden=cfg.thresholds_phase1b.distillability.student.hidden,
            epochs=cfg.thresholds_phase1b.distillability.student.epochs,
            batch_size=cfg.thresholds_phase1b.distillability.student.batch_size,
            lr=cfg.thresholds_phase1b.distillability.student.lr,
            rho_threshold=cfg.thresholds_phase1b.distillability.rho_threshold,
            mse_relative_threshold=cfg.thresholds_phase1b.distillability.mse_relative_threshold,
        )
        mlflow.log_metric("distill_rho_spearman", result.rho_spearman)
        mlflow.log_metric("distill_mse", result.mse)
        mlflow.log_metric("distill_mse_relative", result.mse_relative)
        mlflow.log_metric("distill_passed", float(result.passed))
        finalize_manifest(manifest, duration_s=time.perf_counter() - t0, repo_root=REPO_ROOT)
        verdict = "PASS" if result.passed else "FAIL"
        print(f"Distillabilité : {verdict} (ρ={result.rho_spearman:.3f}, MSE_rel={result.mse_relative:.3f})")


if __name__ == "__main__":
    main()
