# SETUP.md — procédure complète setup pod RunPod RTX 5090

Procédure step-by-step pour passer d'**un pod tout neuf** à **un premier run phase 1 lancé**. Tout est modulaire : chaque étape correspond à un script dédié, exécutable séparément. En cas d'échec, on isole vite la couche fautive.

> **Lecture associée** :
> - `STACK.md` — pourquoi PyTorch 2.11+cu128, contraintes Blackwell sm_120
> - `HARDWARE.md` — topologie VPS + pod, ENV vars Blackwell, pièges connus 5090
> - `LOGGING.md` — MLflow self-hosted, conventions de runs
> - `PRIMITIVES.md` — checklist 6/6 primitives mathématiques (post-validation)

## TL;DR — Setup en 5 commandes

```bash
# Sur le VPS (préparation)
bash OPS/scripts/start_mlflow_server.sh                       # MLflow proxy mode

# Sur le pod (depuis le VPS via SSH)
ssh -p $POD_PORT -i ~/.ssh/id_ed25519 root@$POD_IP \
  "git clone <repo> /workspace/polymorphic-attention || rsync ..."  # voir §3
ssh -p $POD_PORT -i ~/.ssh/id_ed25519 root@$POD_IP \
  "cd /workspace/polymorphic-attention && bash OPS/scripts/setup_pod.sh"

# Sur le VPS (tunnel MLflow inverse)
POD_IP=$POD_IP POD_PORT=$POD_PORT bash OPS/scripts/start_mlflow_tunnel.sh

# Sur le pod (lancer phase 1)
ssh ... "cd /workspace/polymorphic-attention && \
  source OPS/scripts/blackwell_env.sh && \
  MLFLOW_TRACKING_URI=http://localhost:5000 PYTHONPATH=CODE \
  uv run python -m phase1_metrologie.run \
    --config-path=../../OPS/configs/phase1 --config-name=oracle_smnist"
```

---

## Architecture des scripts

Chaque script a une responsabilité **unique**. Ne jamais faire de wrapper qui fait tout d'un coup au-delà de `setup_pod.sh` (orchestrateur de référence).

```
OPS/scripts/
├── blackwell_env.sh          # SOURCE-able : exports ENV vars sm_120
├── install_uv.sh             # idempotent : install uv si absent
├── install_python_deps.sh    # uv sync (CPU ou CUDA, auto-détecté)
├── verify_torch.sh           # smoke check : torch importe + cuda capability
├── setup_pod.sh              # orchestrateur : 4 ci-dessus dans l'ordre
├── start_mlflow_server.sh    # VPS : lance MLflow en proxy mode
├── start_mlflow_tunnel.sh    # VPS : tunnel SSH inverse vers le pod
├── validate_primitives.py    # exhaustif : 6 checks mathématiques sur GPU
└── setup_env.sh              # DEPRECATED : wrapper sur setup_pod.sh
```

**Pourquoi cette granularité** : si l'install des deps casse (cu128 indispo), on relance juste `install_python_deps.sh` après bascule cu126. Si les ENV vars Blackwell sont oubliées dans une session SSH, `source blackwell_env.sh` suffit. Pas besoin de relancer une procédure monolithique de 10 minutes.

---

## 1. Prérequis VPS (avant le pod)

### 1.1 — MLflow tournant en proxy mode

MLflow doit être lancé **avec `--serve-artifacts`** sinon le pod ne pourra pas uploader d'artefacts (le client tente d'écrire au filesystem du serveur, qu'il n'a pas).

```bash
bash OPS/scripts/start_mlflow_server.sh   # bind 127.0.0.1:5000, proxy artefacts
```

Vérification :
```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:5000/   # doit afficher 200
```

Pour persistance via tmux :
```bash
tmux new -s mlflow
bash OPS/scripts/start_mlflow_server.sh
# Ctrl-B D pour détacher
```

### 1.2 — Clé SSH du VPS vers le pod

Le tunnel MLflow inverse + rsync utilisent la clé `~/.ssh/id_ed25519` du VPS. Une copie est conservée dans `OPS/secrets/runpod_id_ed25519` (gitignored, cf. § 5).

Si elle manque :
```bash
ssh-keygen -t ed25519 -C "vps@asp" -f ~/.ssh/id_ed25519 -N ""
```

Et ajouter `id_ed25519.pub` dans la console RunPod → Account → Settings → SSH Public Keys (pour qu'elle soit injectée au démarrage du pod).

---

## 2. Création du pod RunPod

Spécifications validées :

| Champ | Valeur |
|---|---|
| GPU | `NVIDIA GeForce RTX 5090` (sm_120) |
| Cloud type | `COMMUNITY` (économique, suffisant) |
| Image | `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel` |
| Container disk | 40 GB |
| Volume disk | 100 GB |
| Ports | 22/tcp |

L'image `pytorch:2.7.0-cuda12.8` n'est pas finale — on installera **PyTorch 2.11.0+cu128** par-dessus via uv (seul build officiel sm_120). L'image fournit CUDA 12.8 toolkit + cuDNN 9 + drivers compatibles. Cf. `STACK.md`.

Une fois le pod up, noter `POD_IP` et `POD_PORT` (visibles dans la console RunPod).

---

## 3. Apport du repo sur le pod

Deux options.

### 3.a — `rsync` depuis le VPS (recommandé pour pod éphémère)

```bash
rsync -az --info=stats2 \
  --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' \
  --exclude='*.pyc' --exclude='mlruns' --exclude='.mypy_cache' \
  --exclude='.ruff_cache' \
  -e "ssh -p $POD_PORT -i ~/.ssh/id_ed25519" \
  /root/polymorphic-attention/ \
  root@$POD_IP:/workspace/polymorphic-attention/
```

Avantages : pas besoin d'un remote git public, le `.git` (avec HEAD aligné) est inclus → la règle T.1 (run `registered`) est immédiatement satisfaite.

### 3.b — `git clone` depuis un remote privé

Si le repo est sur GitHub privé / Gitea / etc., et que l'agent SSH du pod a la bonne clé :
```bash
ssh -p $POD_PORT -i ~/.ssh/id_ed25519 root@$POD_IP \
  "git clone <repo-url> /workspace/polymorphic-attention"
```

---

## 4. Setup de l'environnement Python sur le pod

### 4.a — Setup full automatique

```bash
ssh -p $POD_PORT -i ~/.ssh/id_ed25519 root@$POD_IP \
  "cd /workspace/polymorphic-attention && bash OPS/scripts/setup_pod.sh"
```

`setup_pod.sh` enchaîne :
1. `source blackwell_env.sh` — exports `TORCH_CUDA_ARCH_LIST=12.0`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `MAX_JOBS=4`, etc.
2. `bash install_uv.sh` — installe uv si absent
3. `bash install_python_deps.sh` — `uv sync --extra cuda --extra dev` (auto-détecté)
4. `bash verify_torch.sh` — vérifie `torch.cuda.is_available()` + capability `(12, 0)`

Idempotent : ré-exécutable sans effet de bord.

### 4.b — Setup en mode debug (étape par étape)

Si `setup_pod.sh` casse à une étape, on isole en relançant individuellement :

```bash
# 1. ENV vars (à sourcer dans la session active)
source OPS/scripts/blackwell_env.sh

# 2. uv
bash OPS/scripts/install_uv.sh

# 3. deps Python
bash OPS/scripts/install_python_deps.sh

# 4. smoke check
bash OPS/scripts/verify_torch.sh
```

### 4.c — Validation exhaustive des primitives mathématiques (Stage 0.2)

Le smoke check de `verify_torch.sh` n'est qu'un import + capability check. Pour valider que les primitives utilisées par phase 1-3 (SVD FP64 batchée, FFT longue, vmap, lstsq batché, attention dense + extraction FP64) tournent correctement sur Blackwell :

```bash
PYTHONPATH=CODE uv run python OPS/scripts/validate_primitives.py
```

Sortie attendue : `6/6 checks passés.` JSON détaillé dans `OPS/env/primitives_results.json`. Coller le résumé dans `OPS/env/PRIMITIVES.md`. Cf. ROADMAP § Stage 0.2.

---

## 5. Tunnel MLflow VPS → pod

MLflow tourne sur le VPS (`127.0.0.1:5000`). Le pod doit le voir comme s'il était local. On utilise un **tunnel inverse** depuis le VPS (plus simple que de configurer une clé pod→VPS pour un forward classique).

```bash
# Sur le VPS :
POD_IP=149.36.1.181 POD_PORT=16279 bash OPS/scripts/start_mlflow_tunnel.sh
```

Ce script :
- Vérifie que MLflow répond sur `127.0.0.1:5000`
- Tue un éventuel tunnel existant
- Ouvre `ssh -N -f -R 5000:127.0.0.1:5000 root@$POD_IP -p $POD_PORT -i $SSH_KEY`
- Affiche le PID pour pouvoir le fermer

Vérification depuis le pod :
```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:5000/   # → 200
```

Sur le pod, exporter avant tout run :
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

---

## 6. OPS/secrets/ — clés et secrets

Le dossier `OPS/secrets/` est **gitignored** (règle `secrets/` du `.gitignore` racine).

Contenu attendu :

| Fichier | Rôle |
|---|---|
| `runpod_id_ed25519` | clé privée SSH pour les pods (chmod 600) |
| `runpod_id_ed25519.pub` | clé publique correspondante |

À copier sur un nouveau VPS / poste de travail :
```bash
mkdir -p OPS/secrets
cp ~/.ssh/id_ed25519     OPS/secrets/runpod_id_ed25519
cp ~/.ssh/id_ed25519.pub OPS/secrets/runpod_id_ed25519.pub
chmod 600                OPS/secrets/runpod_id_ed25519
chmod 644                OPS/secrets/runpod_id_ed25519.pub
```

Pour utiliser ces clés depuis les scripts de tunnel/SSH (au lieu de `~/.ssh/id_ed25519`) :
```bash
SSH_KEY=$(pwd)/OPS/secrets/runpod_id_ed25519 \
  POD_IP=... POD_PORT=... bash OPS/scripts/start_mlflow_tunnel.sh
```

> ⚠ La clé reste sensible même si gitignored. Ne pas la pusher sur un branche, ne pas l'uploader dans des outils tiers.

---

## 7. Premier run phase 1

```bash
ssh -p $POD_PORT -i ~/.ssh/id_ed25519 root@$POD_IP "
cd /workspace/polymorphic-attention && \
source OPS/scripts/blackwell_env.sh && \
nohup env \
  MLFLOW_TRACKING_URI=http://localhost:5000 \
  PYTHONPATH=CODE \
  uv run python -m phase1_metrologie.run \
    --config-path=../../OPS/configs/phase1 \
    --config-name=oracle_smnist \
    extraction.batch_size=4 \
  > /tmp/phase1.log 2>&1 &
echo \$! > /tmp/phase1.pid
echo PID=\$(cat /tmp/phase1.pid)
"
```

Suivi :
```bash
ssh ... "tail -f /tmp/phase1.log"
# ou ouvrir MLflow en local :
ssh -L 5000:127.0.0.1:5000 vps   # puis http://localhost:5000
```

---

## 8. Troubleshooting

| Symptôme | Diagnostic | Fix |
|---|---|---|
| `is_blackwell: False` | driver < 555 | `nvidia-smi` → upgrade driver pod, ou changer d'image RunPod |
| `cu128` indisponible dans uv sync | mirror PyPI absent | bascule `cu126` / `cu124` dans `pyproject.toml § [[tool.uv.index]]`, regen `uv.lock`, repush |
| `cuda available: False` | container sans NVIDIA runtime | recréer le pod avec une image `*-cuda*-devel` |
| MLflow upload artifact 4xx | serveur sans `--serve-artifacts` | redémarrer MLflow avec `start_mlflow_server.sh` (proxy mode déjà inclus) |
| `Expected all tensors on same device` | extraction pas alignée sur cuda | déjà corrigé dans `extract.py` (commit fc834a6) |
| OOM lors de l'extraction phase 1 | `extraction.batch_size` × 6 layers × N² > VRAM | `extraction.batch_size=4` en override CLI |
| Run tagué `exploratory` au lieu de `registered` | `git status` dirty | `git stash` ou commit avant relance (règle T.1) |
| ENV vars Blackwell perdues entre sessions SSH | `source` non hérité | re-`source OPS/scripts/blackwell_env.sh` à chaque nouvelle session |

---

## 9. Dépendances entre les scripts

```
                    blackwell_env.sh  (source-only, no deps)
                          │
                          ▼ (sourced)
   install_uv.sh ──► install_python_deps.sh ──► verify_torch.sh
        │                   │                         │
        └───────────────────┴────────┬────────────────┘
                                     │
                              setup_pod.sh  (orchestrateur)
                                     │
                                     ▼
                          [phase 1 run launchable]

   start_mlflow_server.sh  (VPS only, indépendant)
              │
              ▼
   start_mlflow_tunnel.sh  (VPS only, requiert MLflow up + clé pod)
```

Chaque script peut être appelé indépendamment. `setup_pod.sh` est **un raccourci**, pas une nécessité.

---

## 10. Évolution future de cette doc

Quand un nouveau pattern stabilise (ex : nouvelle image RunPod, nouveau wrapper de tunnel, etc.), **ne pas l'ajouter dans `setup_env.sh`** mais créer un nouveau script dédié et documenter ici. Maintenir l'invariant "un script = une responsabilité".
