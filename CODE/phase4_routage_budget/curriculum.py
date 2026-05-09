"""
curriculum.py — Curriculum de Stress 3 étages.

Spec : DOC/04 §2, ROADMAP 4.2.

Étages V1 :
1. **Stage 1** : régimes faciles (ω, Δ bas, ℋ bas)
2. **Stage 2** : régimes intermédiaires
3. **Stage 3** : régimes durs (ω, Δ hauts, ℋ varié)

Transition : critère de qualité par stage (ex. val_acc ≥ 0.9 sur le stage
courant) avant de passer au suivant.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CurriculumStage:
    name: str
    omega_max: int
    delta_max: int
    entropy_max: float
    transition_acc: float = 0.90    # acc requis pour passer au stage suivant
    min_steps: int = 100            # minimum d'itérations avant transition possible


@dataclass
class CurriculumConfig:
    stages: list[CurriculumStage]


def default_curriculum() -> CurriculumConfig:
    return CurriculumConfig(stages=[
        CurriculumStage(name="easy",         omega_max=2,  delta_max=16,   entropy_max=0.2,
                         transition_acc=0.95, min_steps=200),
        CurriculumStage(name="intermediate", omega_max=6,  delta_max=256,  entropy_max=0.5,
                         transition_acc=0.90, min_steps=300),
        CurriculumStage(name="hard",         omega_max=12, delta_max=4096, entropy_max=1.0,
                         transition_acc=0.0, min_steps=500),  # dernier stage = pas de transition
    ])


class CurriculumScheduler:
    """État machine de progression dans le curriculum."""

    def __init__(self, cfg: CurriculumConfig) -> None:
        self.cfg = cfg
        self.current_idx = 0
        self.steps_in_stage = 0

    def current_stage(self) -> CurriculumStage:
        return self.cfg.stages[self.current_idx]

    def step(self, val_acc: float | None = None) -> None:
        """Avance d'une itération. Si val_acc fourni, vérifie la transition."""
        self.steps_in_stage += 1
        if self.current_idx >= len(self.cfg.stages) - 1:
            return  # dernier stage atteint
        stage = self.cfg.stages[self.current_idx]
        if (
            val_acc is not None
            and val_acc >= stage.transition_acc
            and self.steps_in_stage >= stage.min_steps
        ):
            self.current_idx += 1
            self.steps_in_stage = 0

    def is_final_stage(self) -> bool:
        return self.current_idx >= len(self.cfg.stages) - 1
