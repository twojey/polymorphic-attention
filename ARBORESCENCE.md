# Arborescence du Projet : Attention Superlinéaire Polymorphe (ASP)

**Convention de base** : Arborescence racine figée en trois dossiers majuscules `CODE/`, `DOC/`, `OPS/`.

---

## 📋 Vue d'ensemble

```
polymorphic-attention/
├── CODE/                    # Code source (Python)
├── DOC/                     # Documentation (Markdown)
├── OPS/                     # Configuration opérationnelle & scripts
├── pyproject.toml           # Configuration UV + dépendances Python
├── uv.lock                  # Lockfile dépendances
├── ROADMAP.md               # Feuille de route du projet
├── ARBORESCENCE.md          # Ce fichier
└── [.git, .venv, .claude]   # Infrastructure (non-documentée)
```

---

## 📁 CODE/ — Code source (6 phases + infrastructure)

Structure : une branche par phase du pipeline ASP, plus les modules partagés.

### Structure générale

```
CODE/
├── shared/                  # Modules réutilisables
├── infra/                   # Profils machine & infrastructure
├── catalog/                 # Catalogue exhaustif de propriétés & oracles [PRIORITAIRE 2026-05-12]
├── phase1_metrologie/       # Phase 1 : Métrologie & observations
├── phase1b_calibration_signal/  # Phase 1B : Calibration du signal
├── phase2_audit_spectral/   # Phase 2 : Audit spectral
├── phase3_kernel_asp/       # Phase 3 : Kernel ASP (architecture)
├── phase4_routage_budget/   # Phase 4 : Routage & gestion budget
└── phase5_pareto/           # Phase 5 : Front de Pareto & tests d'évaluation
```

### `CODE/shared/` — Modules partagés

Primitives réutilisables par tous les phases :

```
shared/
├── __init__.py
├── checkpoint.py            # Sérialisation checkpoints (reprise après crash)
├── logging_helpers.py        # Utilitaires de journalisation avec horodatage
├── mlflow_helpers.py         # Intégration MLflow (logging artefacts)
├── plotting.py               # Visualisations (matplotlib/plotly)
├── aggregation.py            # Agrégation métriques cross-batteries
├── runner.py                 # Runner générique pour phases
└── tests/                    # Tests unitaires
```

**Responsabilité** : Zéro code métier, uniquement infrastructure réutilisable.

---

### `CODE/infra/` — Infrastructure machine & profils

```
infra/
├── __init__.py
├── machine.py               # MachineProfile : spécifications CPU/GPU/RAM/VRAM
└── tests/
```

**Responsabilité** : Abstraction hardware (RunPod, VPS local, CPU-only).

---

### `CODE/catalog/` — [PRIORITAIRE] Catalogue de propriétés & oracles

Point d'entrée principal depuis le pivot stratégique 2026-05-12.

```
catalog/
├── __init__.py
├── run.py                   # CLI orchestrateur : lance batteries & oracles

├── properties/              # 131 propriétés classifiées (DOC/00b)
│   └── [propriétés du domaine ASP]

├── oracles/                 # Oracles cross-Oracle (SMNIST → LL, etc.)
│   ├── smnist_oracle.py     # Oracle SMNIST : classifieur sur SMNIST 28×28
│   └── ...

├── batteries/               # Batteries de tests (B1–B6)
│   ├── b1_morphology.py     # B1 : Morphologie & géométrie
│   ├── b2_concentration.py  # B2 : Concentration des normes
│   ├── b3_separation.py     # B3 : Séparation classes
│   ├── b4_robustness.py     # B4 : Robustesse adversariale
│   ├── b5_block_diag.py     # B5 : Block-diagonalité (headwise)
│   ├── b6_banded.py         # B6 : Propriétés Banded
│   └── battery_orchestrator.py

├── projectors/              # Projection multi-niveaux (layer, head, token)
│   └── ...

├── reports/                 # Génération rapports comparatifs
│   └── ...

└── tests/                   # Tests intégration & end-to-end
    └── test_catalog_*.py
```

**Priorité** : Catalogue DOC/00b (131 propriétés) mappées dans le code, cross-Oracle (SMNIST→LL).

**État** : Batteries 1–6 implémentées, adapter déployé, CLI fonctionnel (depuis 2026-05-11).

---

### `CODE/phase1_metrologie/` — Phase 1 : Observations & métrologie

Extraction de signatures brutes depuis un modèle entraîné.

```
phase1_metrologie/
├── __init__.py
├── run.py                   # Runner : exécute métrologie phase 1
├── run_extract.py           # Extraction de signatures brutes
├── report.py                # Génération rapports phase 1
├── sweeps.py                # Configurations hyperparamètres (Hydra)

├── metrics/                 # Métriques & agrégation
│   └── ...

├── oracle/                  # Oracles phase 1 (si spécifiques)
│   └── ...

├── ssg/                     # Signature Stochastic Generator
│   └── ...

└── tests/                   # Tests phase 1
    └── test_phase1_*.py
```

**Responsabilité** : Mesurer & enregistrer métriques brutes depuis réseau forward.

---

### `CODE/phase1b_calibration_signal/` — Phase 1B : Calibration du signal

Validation que les signaux mesurés en phase 1 sont robustes & reproductibles.

```
phase1b_calibration_signal/
├── __init__.py
├── run.py                   # Runner : exécute calibration
├── run_distillability.py    # Test distillabilité du signal

├── signals/                 # Générateurs signaux
│   └── ...

├── bench/                   # Benchmarks robustesse
│   └── ...

└── tests/
    └── test_calibration_*.py
```

**Responsabilité** : Confirmer reproductibilité avant passage à phase 2.

---

### `CODE/phase2_audit_spectral/` — Phase 2 : Audit spectral & décomposition

Décomposer signatures en composantes spectrales indépendantes.

```
phase2_audit_spectral/
├── __init__.py
├── run.py                   # Runner phase 2
├── checkpoint.py            # Sérialisation états intermédiaires

├── svd_pipeline.py          # Pipeline SVD/décomposition
├── head_specialization.py   # Analyse spécialisation par head
├── signal_decoupling.py     # Découplage signaux indépendants
├── stress_rank_map.py       # Mapping stress → rang matriciel
├── transfer_law.py          # Lois transfert entre architectures

├── batteries/               # Batteries spécifiques phase 2
│   └── ...

└── tests/
    └── test_phase2_*.py
```

**Responsabilité** : Décomposer & analyser structure spectrale, valider hypothèses transfert.

---

### `CODE/phase3_kernel_asp/` — Phase 3 : Kernel ASP (architecture)

Implémentation mécanisme ASP core : allocation dynamique capacité par head/token.

```
phase3_kernel_asp/
├── __init__.py
├── run_train.py             # Runner entraînement phase 3
├── checkpoint.py            # Checkpoints entraînement

├── asp_layer.py             # Couche ASP core
├── backbone.py              # Interface backbone générique
├── backbone_concrete.py     # Implémentations concrètes (Attention polymorphe, etc.)
├── transformer.py           # Wrapper Transformer avec backbone ASP
├── losses.py                # Fonctions loss (ASP-spécifiques)

├── matriochka.py            # Nested representations (multi-scale)
├── smart_init.py            # Initialisation intelligente poids
├── soft_mask.py             # Masques progressifs (soft vs hard)
├── sanity.py                # Tests sanité forward/backward

└── tests/
    └── test_phase3_*.py
```

**Responsabilité** : Implémentation architecture ASP & entraînement.

---

### `CODE/phase4_routage_budget/` — Phase 4 : Routage & gestion budget

Optimiser allocation capacité via curriculum & sparsity.

```
phase4_routage_budget/
├── __init__.py
├── curriculum.py            # Curriculum learning (progression difficulté)
├── sparsity_loss.py          # Constraints sparsité activations
├── distillation.py           # Distillation connaissances
├── diagram_phase.py          # Diagrammes flow & bottlenecks
├── spectrometer.py           # Analyse spectrale budgets

└── tests/
    └── test_phase4_*.py
```

**Responsabilité** : Optimiser allocation ressources & apprentissage.

---

### `CODE/phase5_pareto/` — Phase 5 : Front de Pareto & validation

Tests d'évaluation & caractérisation trade-offs.

```
phase5_pareto/
├── __init__.py
├── abstract.py              # Classe abstraite test
├── pareto.py                # Calcul & visualisation front Pareto

├── test_5a_identifiability.py    # Test 5A : Identifiabilité
├── test_5b_elasticity.py         # Test 5B : Élasticité
├── test_5c_se_ht.py              # Test 5C : Stabilité énergétique
├── test_5e_ood.py                # Test 5E : Out-of-Distribution
├── test_6c_rmax_half.py          # Test 6C : Capacité maximale

└── tests/
    └── test_phase5_*.py
```

**Responsabilité** : Valider hypothèses & caractériser front Pareto.

---

## 📚 DOC/ — Documentation (Markdown, français)

Architecture : 
- Fichiers numérotés : **00_** (pré-requis), **01–05** (phases)
- **CONTEXT.md** : Langage de domaine actuel
- **carnet_de_bord.md** : Journal chronologique (hypothèses, décisions, surprises)
- **adr/** : Architecture Decision Records

```
DOC/
├── README.md                        # Introduction & guide lecture
├── CONTEXT.md                       # Langage de domaine, variables clés
├── glossaire.md                     # Définitions termes techniques
├── falsifiabilite.md                # Méthodologie falsifiabilité-first

├── 00_vision.md                     # Vue d'ensemble & motivation
├── 00b_classification_proprietes.md # [PRIORITAIRE] Catalogue 131 propriétés
├── 00c_predictions_signatures.md    # Prédictions signatures
├── 00d_oracles_battery.md           # Oracles & batteries mappées
├── 01_phase_metrologie.md           # Phase 1 : Métrologie
├── 01b_phase_calibration_signal.md  # Phase 1B : Calibration signal
├── 02_phase_audit_spectral.md       # Phase 2 : Audit spectral
├── 03_phase_kernel_asp.md           # Phase 3 : Kernel ASP
├── 04_phase_routage_budget.md       # Phase 4 : Routage budget
├── 05_phase_pareto.md               # Phase 5 : Pareto

├── carnet_de_bord.md                # [CRUCIAL] Journal projet (hypothèses, chronologie)

├── adr/                             # Architecture Decision Records
│   ├── 001_stack_tech.md            # Stack PyTorch + Fabric + Hydra + uv
│   ├── 002_hardware.md              # Hardware RunPod RTX 5090 éphémère
│   ├── 003_logging.md               # Logging MLflow self-hosted
│   └── ...

└── reports/                         # Rapports générés
    ├── phase1_results.md
    ├── phase2_results.md
    └── ...
```

**Fichiers clés** :
- `00b_classification_proprietes.md` : Catalogue exhaustif 131 propriétés (mappé en CODE/catalog)
- `carnet_de_bord.md` : Source de vérité pour hypothèses & décisions chronologiques

---

## ⚙️ OPS/ — Configuration & infrastructure opérationnelle

Dossier de configuration, checkpoints, logs, secrets.

```
OPS/
├── configs/                 # Configuration Hydra par phase
│   ├── phase1/
│   │   ├── default.yaml
│   │   ├── smnist_oracle.yaml
│   │   └── ...
│   ├── phase1b/
│   ├── phase2/
│   ├── phase3/
│   ├── phase4/
│   └── phase5/

├── scripts/                 # Scripts shell et Python
│   ├── setup_pod.sh                    # Setup initial pod RunPod
│   ├── setup_pod_cpu.sh                # Setup variante CPU-only
│   ├── setup_env.sh                    # Setup environnement local
│   ├── install_uv.sh                   # Installation UV
│   ├── install_python_deps.sh          # Installation dépendances Python
│   ├── blackwell_env.sh                # Configuration Blackwell GPU
│   ├── start_mlflow_server.sh          # Démarrage serveur MLflow
│   ├── start_mlflow_tunnel.sh          # Tunnel SSH pour MLflow
│   ├── run_phase1b_skl.sh              # Runner scikit-learn phase 1B
│   ├── monitor_run.sh                  # Monitoring exécution
│   ├── verify_torch.sh                 # Vérification PyTorch
│   ├── validate_primitives.py          # Validation primitives
│   └── lib/                            # Librairie shell réutilisable
│       └── common.sh

├── env/                     # Fichiers environnement (secrets, clés API)
│   ├── SETUP.md            # [LU EN PREMIER] Guide setup pod
│   └── runpod_env.sh       # Variables d'environnement RunPod

├── checkpoints/            # Checkpoints modèles
│   ├── phase1/
│   │   └── *.pt
│   ├── phase2/
│   │   └── *.pt
│   ├── phase3/
│   │   └── *.pt
│   └── ...

├── logs/                    # Logs exécution (persistent)
│   ├── mlflow/             # MLflow artifacts
│   │   ├── 0/              # Expérience 0
│   │   │   ├── 1/          # Run 1
│   │   │   │   ├── metrics/
│   │   │   │   ├── params/
│   │   │   │   └── artifacts/
│   │   │   └── ...
│   │   └── ...
│   ├── pod_2026_05_12/     # Logs pod sesssion 2026-05-12
│   │   ├── setup.log
│   │   ├── phase1.log
│   │   └── ...
│   └── manifests/          # Manifests d'exécution
│       └── phase1_manifest.json

└── secrets/                 # Secrets (API keys, tokens)
    ├── mlflow_token
    ├── runpod_api_key
    └── [.gitignored]
```

**Structure clés** :
- `OPS/env/SETUP.md` : **À LIRE EN PREMIER** pour setup pod
- `OPS/logs/` : Logs persistent avec horodatage (obligatoire robustesse scripts)
- `OPS/configs/` : Configuration Hydra centralisée par phase
- `OPS/scripts/` : Scripts modulaires (une responsabilité = un script)

---

## 📊 Fichiers racine

```
polymorphic-attention/
├── pyproject.toml           # Configuration UV + dépendances
├── uv.lock                  # Lockfile (déterministe)
├── ROADMAP.md               # Feuille de route détaillée (phases, jalons)
├── ARBORESCENCE.md          # Ce fichier
├── .gitignore               # Exclusions git
├── CLAUDE.md                # [À créer] Documentation CLAUDE Code
└── .claude/                 # Configuration Claude Code (local)
```

---

## 🔄 Convention arborescence - Immuabilité racine

Les trois dossiers `CODE/`, `DOC/`, `OPS/` sont **figés** au niveau racine :

- ✅ Ajouter modules *à l'intérieur* de `CODE/phase*/`
- ✅ Ajouter documentation *à l'intérieur* de `DOC/`
- ✅ Ajouter configs/scripts *à l'intérieur* de `OPS/`
- ❌ Créer nouveaux dossiers au niveau racine
- ❌ Déplacer CODE/DOC/OPS de la racine

**Exception** : Fichiers de configuration racine (pyproject.toml, ROADMAP.md, etc.)

---

## 🎯 État du projet (2026-05-12)

| Phase | État | Notes |
|-------|------|-------|
| **Catalog** | ✅ Prioritaire | 131 propriétés, batteries 1–6, oracle adapter, CLI |
| **Phase 1** | ✅ Complet | Métrologie, 119 tests |
| **Phase 1B** | ✅ Complet | Calibration signal |
| **Phase 2** | ✅ En cours | Audit spectral, décomposition |
| **Phase 3** | 🔄 Dev | Kernel ASP, en attente pod RTX 5090 |
| **Phase 4** | 📋 Planifié | Routage budget |
| **Phase 5** | 📋 Planifié | Front Pareto |

**Blocker** : Attente accès pod RunPod RTX 5090 pour entraînement phases 3–5.

---

## 📖 Guide lecture documentation

1. **Commencer par** : `DOC/README.md` → `DOC/CONTEXT.md`
2. **Contexte scientifique** : `DOC/00_vision.md` → `DOC/falsifiabilite.md`
3. **Catalogue** (prioritaire) : `DOC/00b_classification_proprietes.md`
4. **Phases** : `DOC/01_phase_metrologie.md` → ... → `DOC/05_phase_pareto.md`
5. **Chronologie & décisions** : `DOC/carnet_de_bord.md`
6. **Patterns architecturaux** : `DOC/adr/`

---

**Document généré** : 2026-05-12 | Pivot stratégique : Catalogue prioritaire sur ASP itérations
