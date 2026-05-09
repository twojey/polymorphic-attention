# LOGGING.md — Outil de logging d'expérience

**Statut** : décision prise (Stage 0.4, ROADMAP). MLflow self-hosted sur le VPS, complété par fichiers locaux pour la traçabilité offline.

**Aucun compte externe requis. Aucune donnée ne sort des deux machines.**

## Décision

| Composant | Choix |
|---|---|
| Logging principal | **MLflow self-hosted sur le VPS** |
| Backend store | SQLite (`OPS/logs/mlflow/mlflow.db`) |
| Artifact store | Filesystem local sur VPS (`OPS/logs/mlflow/artifacts/`) |
| Manifest de run local | Dump YAML dans `OPS/logs/manifests/<run_id>.yaml` |
| Suivi compute (humain) | Markdown manuel dans `OPS/logs/compute_budget.md` (cf. T.2) |

## Topologie

```
[Pod RunPod RTX 5090 — éphémère]            [VPS — persistant]
        │                                          │
   training script                            MLflow server
   mlflow.log_metric()                        bind 127.0.0.1:5000
        │                                          │
        │  HTTP via tunnel SSH                     │  Backend  : sqlite
        │  ssh -L 5000:localhost:5000 vps          │  Artifacts: filesystem
        ▼                                          │
   localhost:5000 ───────────────────────────────► │
                                                   │
                                            Browser local sur VPS
                                            (ou tunnel SSH depuis poste)
```

Le pod ouvre un tunnel SSH vers le VPS au démarrage du training. MLflow client sur le pod cible `http://localhost:5000` qui est forwardé vers le serveur MLflow tournant sur le VPS.

**Sécurité** : MLflow OSS n'a pas d'authentification. Le serveur **bind sur `127.0.0.1` exclusivement**, jamais exposé publiquement. L'accès passe par SSH (déjà authentifié).

## Justifications

### Pourquoi MLflow self-hosted

- **Pas de compte externe.** Pas de signup, pas de tracking, pas de dépendance cloud.
- **Topologie VPS + pod éphémère** résolue : le pod log vers le VPS persistant via SSH tunnel. Au reboot du pod, l'historique est intact côté VPS.
- **Maturité** : MLflow est l'outil le plus utilisé en open-source pour ce cas. Doc large, communauté grande.
- **Intégration Lightning Fabric** : `MLFlowLogger` fourni par Lightning, branchable en 3 lignes.
- **Sweeps** : `Hydra --multirun` lance N runs séquentiels ou parallèles, chacun ouvre un `mlflow.start_run()` distinct. Suffisant pour les 5–7 valeurs de `λ_budget` phase 4.
- **Artifacts versionnés** : `mlflow.log_artifact()` et `mlflow.log_model()` couvrent poids Oracle, matrices d'attention, dictionnaire SCH, checkpoints ASPLayer.

### Pourquoi pas Aim

UI plus moderne mais écosystème plus petit, intégration Lightning moins documentée. Moins risqué de partir sur MLflow.

### Pourquoi pas TensorBoard seul

Pas d'artifact management, pas de comparaison cross-runs riche, fichiers à synchroniser manuellement depuis le pod éphémère.

### Pourquoi pas W&B

Compte externe requis, données sortent des machines. Refusé par l'utilisateur.

## Lancement du serveur (sur le VPS)

```bash
bash OPS/scripts/start_mlflow_server.sh
```

Le script :
1. Crée `OPS/logs/mlflow/{,artifacts}/` si absent
2. Lance `mlflow server --host 127.0.0.1 --port 5000` avec backend SQLite et artifact store filesystem
3. Bloque en foreground (utiliser `tmux`/`screen`/systemd pour lancer en arrière-plan)

Le serveur tourne en permanence sur le VPS pendant le projet.

## Configuration du pod (au démarrage du training)

```bash
# 1. Tunnel SSH vers le VPS (en arrière-plan)
ssh -N -f -L 5000:127.0.0.1:5000 user@vps

# 2. Pointer MLflow client vers le tunnel local
export MLFLOW_TRACKING_URI=http://localhost:5000

# 3. Lancer le training
uv run python -m asp.phase1.train_oracle --config-name=oracle_smnist
```

`setup_env.sh` vérifie que `MLFLOW_TRACKING_URI` est set, mais n'ouvre pas le tunnel automatiquement (clé SSH = responsabilité utilisateur).

## Conventions de nommage des runs MLflow

- **Experiment name** = `phase<N>` (ex. `phase1`, `phase1.5`, `phase2`)
- **Run name** = `<sprint>_<domaine>_<short_description>_<git_short_hash>`
  - Exemple : `s1_smnist_oracle_baseline_a1b2c3d`
- Le `git_short_hash` est les 7 premiers caractères du commit du repo au moment du run. Sans commit clean, pas de run "registered" (règle T.1).

## Tags MLflow obligatoires

À chaque run, via `mlflow.set_tags({...})` :

- `phase` : `1` / `1.5` / `2` / `3` / `4` / `5`
- `sprint` : `1` / `2` / `3` / `4`
- `domain` : `smnist` / `code` / `vision`
- `oracle_id` : id de l'Oracle utilisé (phases 2+, sinon vide)
- `status` :
  - `exploratory` — run libre, jamais cité comme preuve dans un rapport de phase
  - `registered` — run reproductible à partir d'un commit clean, citable comme preuve
  - `invalidated` — run pré-enregistré dont la config a été modifiée a posteriori → invalidé

## Articulation avec `OPS/logs/`

```
OPS/logs/
├── compute_budget.md         # commité, suivi humain GPU-h
├── runs_index.csv            # commité, index humain des runs majeurs
├── manifests/                # commité (.gitkeep)
│   └── <run_id>.yaml         # un par run, copie locale du manifest
└── mlflow/                   # gitignored
    ├── mlflow.db
    └── artifacts/
```

- `OPS/logs/mlflow/` : gitignored, vit sur le VPS exclusivement.
- `OPS/logs/manifests/<run_id>.yaml` : commité, copie statique du manifest. Sauvegarde si la base MLflow est corrompue.
- `OPS/logs/runs_index.csv` : édité à la main après chaque run de référence (`run_id, phase, sprint, date, status, verdict`).

## Pré-enregistrement et MLflow

Règle T.1 (pré-enregistrement) impose que tous les seuils soient fixés avant la phase. Articulation :

1. La config Hydra du run est commitée **avant** le run (commit dédié, message `pre-register: phaseX <description>`).
2. Le `git_short_hash` du nom de run pointe vers ce commit.
3. Le run est lancé avec `tags={"status": "registered"}` ; le code refuse de lancer si `git status` est dirty.
4. Toute modification post-run → nouveau commit, nouveau run, ancien run re-tagué `status:invalidated` via `mlflow.set_tag("status", "invalidated")`.

## Sauvegarde du serveur MLflow

Le VPS persiste mais peut tomber. Sauvegarde manuelle recommandée :

- Snapshot quotidien de `OPS/logs/mlflow/` (rsync vers stockage externe, ou snapshot du VPS).
- Pas de réplication active à ce stade — single-user, projet de recherche, RPO 24h acceptable.

## Évolutions possibles

- Si le projet implique de la collaboration : ajouter Caddy/nginx + basic auth devant MLflow, ou bascule vers Tailscale pour la mesh privée.
- Si le volume d'artifacts dépasse l'espace VPS : déplacer `default-artifact-root` vers un bucket S3-compatible self-hosted (MinIO).
