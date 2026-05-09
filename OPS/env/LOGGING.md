# LOGGING.md — Outil de logging d'expérience

**Statut** : décision prise (Stage 0.4, ROADMAP). W&B comme outil principal, complété par fichiers locaux pour la traçabilité offline.

## Décision

| Composant | Choix |
|---|---|
| Logging principal | **Weights & Biases (W&B)** |
| Manifest de run local | Dump YAML dans `OPS/logs/manifests/<run_id>.yaml` |
| Suivi compute (humain) | Markdown manuel dans `OPS/logs/compute_budget.md` (cf. T.2) |
| Stockage artefacts (poids, matrices A, dictionnaire SCH) | W&B Artifacts |

## Justifications

### Pourquoi W&B

- **Topologie VPS + RunPod éphémère** : le pod est détruit régulièrement. Les fichiers TensorBoard/MLflow self-hosted seraient à copier manuellement à chaque arrêt. W&B sync vers le cloud en continu → consultable depuis le VPS sans intervention.
- **Sweeps natifs** : phase 4 prévoit 5–7 valeurs de `λ_budget` log-spaced ; phase 5 prévoit 3 seeds × 4 tailles `N` × plusieurs comparateurs. Hydra + W&B sweep gèrent ça en une commande.
- **Artifacts versionnés** : poids Oracle (par domaine), matrices d'attention extraites, dictionnaire SCH, checkpoints ASPLayer. Versionnés et liés au run qui les a produits → aucun doute sur la provenance d'un artefact.
- **Comparaison cross-phase** : le projet produit ~6 phases × multiples runs × 4 sprints. W&B permet de filtrer/grouper/comparer sur des centaines de runs.
- **Intégration Lightning Fabric** : `WandbLogger` natif, pas de glue code.
- **Free tier académique** suffisant pour le volume estimé.

### Pourquoi pas TensorBoard

Pas d'artifact management, pas de sweeps, pas de collaboration, fichiers à copier manuellement depuis le pod. Dispo *en plus* via Fabric pour debug local rapide, jamais comme source de vérité.

### Pourquoi pas MLflow self-hosted

Demanderait un service hébergé sur le VPS, plus de surface à maintenir. Aucun gain par rapport à W&B sur ce projet.

### Pourquoi pas fichiers locaux uniquement

Incompatible avec la topologie pod éphémère. Et aucune comparaison/sweep automatique.

## Ce qui est loggé où

| Information | Cible |
|---|---|
| Métriques scalaires (loss, accuracy, ρ Spearman, etc.) | W&B `log` |
| Histogrammes (distribution `r_eff`, `R_target`, etc.) | W&B `log` (mode histogram) |
| Heatmaps (Stress-Rank Map, Diagramme de Phase) | W&B `log` (mode image) |
| Poids Oracle (par domaine) | W&B Artifact, type `model` |
| Matrices d'attention extraites (sous-échantillon) | W&B Artifact, type `dataset` |
| Dictionnaire SCH (table régime → r_eff → classe) | W&B Artifact, type `dataset` |
| Checkpoints ASPLayer | W&B Artifact, type `model` |
| Manifest run (seed, commit, fingerprint hardware) | YAML local + W&B `config` |
| Budget compute consolidé (GPU-hours par phase) | `OPS/logs/compute_budget.md` (édition humaine) |
| Rapports de phase | `DOC/reports/phaseX_report.md` (commités) |

## Conventions de nommage des runs W&B

Format : `<phase>_<sprint>_<domaine>_<short_description>_<short_hash>`

Exemples :
- `phase1_s1_smnist_oracle_baseline_a1b2c3d`
- `phase1b_s1_smnist_signals_3sig_e4f5g6h`
- `phase2_s2_smnist_audit_battA_i7j8k9l`
- `phase4_s3_smnist_lambda_sweep_m1n2o3p`

Le `short_hash` est les 7 premiers caractères du commit git du repo au moment du run. Sans commit, pas de run (règle T.1).

## Tags W&B obligatoires

Chaque run doit porter au minimum :

- `phase:1` / `phase:1.5` / `phase:2` / `phase:3` / `phase:4` / `phase:5`
- `sprint:1` / `sprint:2` / `sprint:3` / `sprint:4`
- `domain:smnist` / `domain:code` / `domain:vision`
- `oracle:<oracle_id>` (phases 2+)
- `status:exploratory` ou `status:registered` — un run `registered` correspond à une config pré-enregistrée non modifiée (cf. règle T.1). Les runs `exploratory` ne comptent jamais comme preuve dans un rapport de phase.

## Articulation avec `OPS/logs/`

- `OPS/logs/` est **gitignored** sauf `compute_budget.md` qui est commité.
- `OPS/logs/manifests/<run_id>.yaml` : copie locale du manifest, utile en cas de perte d'accès W&B.
- `OPS/logs/runs_index.csv` : index humain des runs majeurs avec colonnes `run_id, phase, sprint, date, status, verdict`. Édité à la main après chaque run de référence.

## Pré-enregistrement et W&B

Règle T.1 (pré-enregistrement) impose que tous les seuils soient fixés avant la phase. Articulation avec W&B :

1. La config Hydra du run est commitée **avant** le run (commit dédié, message `pre-register: phaseX <description>`).
2. Le `short_hash` du nom de run pointe vers ce commit.
3. Le run est lancé avec `tags=[..., "status:registered"]`.
4. Toute modification post-run → nouveau commit, nouveau run, ancien run annulé (jamais supprimé, marqué `status:invalidated` via `wandb.run.tags.add(...)` ou re-tag manuel).

## Sécurité

- Token W&B injecté via env var `WANDB_API_KEY` au démarrage du pod. Jamais commité.
- Pas de données utilisateur, pas de PII dans les datasets → mode cloud public W&B acceptable.
- Si plus tard un dataset sensible est introduit (peu probable : SSG + corpus synthétiques), basculer sur mode privé W&B ou self-hosted.
