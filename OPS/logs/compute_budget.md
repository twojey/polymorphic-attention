# Budget compute — projet ASP

Suivi humain des GPU-hours consommées par phase. Édition manuelle après chaque run de référence (cf. ROADMAP T.2).

**Alerte** : si la consommation cumulée d'une phase dépasse 2× le budget initial estimé, suspendre les runs et réviser.

## Estimations initiales (Sprint 1, RTX 5090)

| Phase | Estimé (GPU-h) | Justification |
|---|---|---|
| 1 (Oracle SMNIST) | 30 | balayages monovariés ω/Δ/ℋ + 3 croisés, plateau de loss validation |
| 1.5 (3 signaux) | 6 | inférence sur banc hybride, pas d'entraînement Oracle |
| 1.5b (Distillabilité) | 4 | MLP léger student, S_Spectral teacher |

Total Sprint 1 estimé : **~40 GPU-h**.

## Consommation réelle

| Date | Phase | Run ID | GPU-h | Cumul phase | Cumul total | Notes |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | Aucun run effectué à ce jour |

## Plafonds par phase (à mettre à jour avant chaque sprint)

| Phase | Plafond hard (GPU-h) | Action si dépassement |
|---|---|---|
| 1 | 60 | Pause + review |
| 1.5 | 12 | Pause + review |
| 1.5b | 8 | Pause + review |
