# Phase 1 — Métrologie (RCP) via le SSG

Spec : [DOC/01_phase_metrologie.md](../../DOC/01_phase_metrologie.md).

## Modules attendus

- **ssg/** — Structural Stress Generator. Producteur de Structure-MNIST paramétré par (ω, Δ, ℋ), avec balayages monovariés puis croisés. Cœur du protocole.
- **domains/** — générateurs de données par domaine pour le multi-Oracle (cf. DOC/01 section 7b) :
  - `structure_mnist/` (cœur, équivalent au SSG)
  - `code_synth/` (Dyck-k étendu, mini-DSL — code synthétique)
  - `vision/` (MNIST/CIFAR par patches, optionnel V1)
- **dataset_split/** — découpage **par domaine** en `train_oracle` / `audit_svd` / `init_phase3` (règle des trois sets disjoints, cf. DOC/01 section 8.6). Vérification d'absence de fuite par hashing, par domaine.
- **oracle/** — Oracle (Transformer dense pur, pas de GQA, pas de sliding window, context length ≥ max(Δ)). Quality gates dans `quality_gates.py` (cf. DOC/01 section 8). Entraînement BF16 mixed-precision avec arrêt par plateau de loss validation par régime. **Modifications post-entraînement interdites**. Architecture identique entre Oracles, seules les adaptations strictement nécessaires au domaine (taille embedding, patches vs tokens) sont autorisées.
- **extract/** — extraction des matrices d'attention par tête × couche × exemple, cast FP64 au moment de la sortie. Pas d'agrégation pré-extraction.
- **metrics/** — rang de Hankel numérique, entropie spectrale, agrégation par régime (post-extraction).
- **report/** — génération des courbes monovariées et des cartes 2D rang(ω, Δ), H(ω, Δ), etc., **par domaine**.

## Entrées

- Spécification du balayage (ω, Δ, ℋ) (config OPS/).
- Hyperparamètres de l'Oracle.

## Sorties

- Matrices d'attention exportées par (Oracle/domaine, couche, tête, exemple).
- Poids de chaque Oracle (consommés en phase 2 et utilisés comme baselines en phase 5).
- Statistiques agrégées par domaine (csv ou parquet).
- Courbes monovariées + cartes 2D par domaine (figures + données brutes).
- Rapport phase 1 avec recommandation préliminaire de R_max pour la phase 3 (par domaine si différents).

## Critère de fin

Le go/no-go de la phase 1 (cf. DOC/falsifiabilite.md) est tranché et documenté. La stack est arrêtée à ce moment-là.

## Lancement (V1 — Sprint 1, Structure-MNIST seul)

```bash
# Sur le pod, après setup_env.sh + tunnel SSH MLflow :
PYTHONPATH=CODE uv run python -m phase1_metrologie.run \
    --config-path=../../OPS/configs/phase1 \
    --config-name=oracle_smnist
```

Le driver enchaîne SSG → Oracle → extraction FP64 → métriques → MLflow + manifest.

## Modules implémentés (V1)

- `ssg/structure_mnist.py` — SSG token-level avec axes (ω, Δ, ℋ), tri-partition
- `oracle/transformer.py` — Transformer dense, hooks capture A
- `oracle/train.py` — boucle Fabric BF16 + plateau detection
- `oracle/extract.py` — extraction A par (couche, tête, exemple) FP64
- `metrics/hankel.py` — rang Hankel numérique
- `metrics/spectral.py` — entropie spectrale H = -Σ p_k log p_k
- `tests/` — 11 tests SSG + 12 tests métriques

## Limites V1 documentées

- **Token-level**, pas pixel-level. Les "digits" sont des tokens d'identité 0-9. La structure (ω, Δ, ℋ) est préservée. Bascule pixel-level prévue post-Sprint 1 si nécessaire (cf. ssg/structure_mnist.py docstring).
- **Domaine SMNIST seul** pour Sprint 1. Multi-Oracle (code synthétique, vision) reporté à Sprint 4 conformément à la stratification de la ROADMAP.
- **Extraction sur 1 batch** dans le driver V1. À étendre pour couvrir tout audit_svd avant rapport phase 1.
