# HARDWARE.md — Hardware cible du projet ASP

**Statut** : décision prise (Stage 0.3, ROADMAP). Topologie à deux machines, training sur RunPod éphémère.

## Topologie

| Machine | Rôle | Persistance |
|---|---|---|
| **VPS** | Édition code, édition doc, **serveur MLflow** (cf. LOGGING.md), push git, lancements de runs distants | Persistante, contient le repo source et la base MLflow |
| **RunPod RTX 5090** | Training, extraction, audit spectral, calibration spectromètre | **Éphémère** — disque détruit à l'arrêt du pod |

Aucun training sur le VPS. Aucune édition long-terme sur le pod.

## Spécifications RunPod RTX 5090

| Paramètre | Valeur |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 (Blackwell, sm_120) |
| Compute capability | (12, 0) — confirmé par tests Lumis sur pod réel |
| VRAM | 32 GB |
| Précision native | FP64, FP32, TF32, BF16, FP16, FP8 |
| CUDA | 12.8 |
| Driver | ≥ 555 |
| PyTorch | **≥ 2.11.0+cu128** (seul build officiel sm_120) |

## Image Docker recommandée pour le pod

**`pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel`** (validée par les scripts Lumis sur 5090).

Cette image fournit Python + CUDA toolkit 12.8 system-wide. `setup_env.sh`
y crée un `.venv` avec `uv sync --extra cuda --extra dev` qui installe
torch 2.11.0+cu128 dans le venv (override la version system).

Alternative de fallback si l'image ci-dessus est indisponible :
`runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04` — mais elle ne
supporte PAS Blackwell directement, il faut quand même réinstaller torch
via `setup_env.sh`.

## Configuration RunPod (création du pod)

| Paramètre | Valeur |
|---|---|
| GPU type | `NVIDIA GeForce RTX 5090` |
| Cloud type | **`COMMUNITY`** (la 5090 est plus disponible en COMMUNITY que SECURE) |
| Container image | `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel` |
| Container disk | 40 GB |
| Volume | 100 GB |
| Ports exposés | `22/tcp` (SSH) |

## Variables d'environnement Blackwell

Toutes set automatiquement par `OPS/scripts/setup_env.sh`. Pour info :

```bash
export TORCH_CUDA_ARCH_LIST="12.0"             # cible compute capability sm_120
export CUDA_HOME=/usr/local/cuda
export FORCE_CUDA=1
export MAX_JOBS=4                               # anti-freeze sur compilation parallèle
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_P2P_DISABLE=1                       # bénin single-GPU
export TOKENIZERS_PARALLELISM=false
```

## Pièges connus 5090 (issus des scripts Lumis validés)

| Symptôme | Cause | Mitigation |
|---|---|---|
| Erreur kernel CUDA non supporté | torch < 2.11 ou index ≠ cu128 | `setup_env.sh` pinne `torch>=2.11.0` + index `cu128` |
| Compilation freeze | parallélisme excessif | `MAX_JOBS=4` set par `setup_env.sh` |
| OOM intermittent | fragmentation VRAM | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set |
| Latence init CUDA élevée | chargement eager des modules | `CUDA_MODULE_LOADING=LAZY` set |
| `nvidia-smi` reporte un GPU différent au reboot | RunPod réassigne le pod | `setup_env.sh` log le GPU réel ; abandonner et relancer si mismatch |

## ASP n'a PAS besoin de

Pour mémoire — si on regarde la stack Lumis qui a beaucoup plus de deps :

- `xformers`, `bitsandbytes`, `unsloth`, `transformers` HF, `peft`, `trl` :
  toute la stack LLM finetuning. Pas notre cas. On a notre propre Transformer
  dense (cf. `CODE/phase1_metrologie/oracle/transformer.py`).
- Quantization : pas pour Sprint 1. Phase 5.3 (HT) uniquement si nécessaire.
- Flash Attention : équivalent mathématique à attention dense, autorisé à
  l'entraînement Oracle, désactivé à l'extraction (DOC/01 §8.1, §8.4).

## Implications protocole

### Phase 1 — Oracle dense

- Attention dense pure exigée (pas GQA, pas sliding, pas Flash) lors de l'extraction.
- VRAM dominée par `O(N² · n_têtes · n_couches · batch)` au moment de l'extraction.
- Sur Structure-MNIST, `Δ_max` du SSG fixe `N`. **Avant chaque run d'Oracle, vérifier que `N · n_têtes · n_couches · 4 octets (FP32)` × batch tient en 32 GB**, sinon réduire batch et accumuler.
- Flash Attention autorisé pour l'**entraînement** de l'Oracle (équivalent mathématique). Désactivé au moment de l'extraction.

### Phase 2 — Audit Spectral

- SVD batchée en **FP64**. Coût mémoire ≈ `O(N² · 8 octets)` par matrice + workspace. Pour `N = 2¹²`, ~134 MB par matrice → tient largement.
- À `N = 2¹⁶`, une seule matrice FP64 fait 32 GB → SVD obligatoire en streaming (une matrice à la fois) ou downsampling. Anticiper.

### Phase 3 — ASPLayer

- Backbone dérivé du dictionnaire SCH. Convolution causale FFT-based ≈ `O(N log N)` mémoire. Tient confortablement.
- Bases Matriochka `U_max, V_max ∈ ℝ^{d × R_max}` : négligeable.
- L_matriochka avec ≥ 4 tirages de `r` par batch : multiplie le forward par 4. Réduire batch en conséquence.

### Phase 5 — Hardware Throughput

- Mesure `Inference_Time` réelle (pas analytique, cf. ROADMAP 5.3). Le pod RunPod doit être **dédié** au moment de la mesure (pas de partage de GPU). Documenter `nvidia-smi` avant chaque mesure HT.

## Reproductibilité — pod éphémère

Le pod est détruit à chaque arrêt. Conséquences strictes :

1. **`OPS/scripts/setup_env.sh` doit être idempotent** et restaurer l'environnement à zéro depuis le repo + lockfile uv.
2. **Tous les artefacts persistants vivent ailleurs** : poids Oracle, matrices `A`, dictionnaire SCH → MLflow artifacts sur le VPS via `mlflow.log_artifact()` (cf. LOGGING.md). Jamais sur le disque du pod seul.
3. **Pas de modification de l'environnement à la main sur le pod.** Toute modification → édition de `setup_env.sh` ou du `pyproject.toml`, push, re-pull, re-sync.
4. **Le repo est cloné depuis git** au démarrage du pod, pas synchronisé manuellement. Travail en cours non commité = perdu.

## Fingerprint hardware

Chaque manifest de run (cf. T.2 ROADMAP, format défini Stage 0.7) doit inclure :

```yaml
hardware:
  gpu: NVIDIA GeForce RTX 5090
  vram_gb: 32
  cuda: "12.8"
  driver: "555.xx"
  pod_id: <runpod_pod_id>
  pod_started_at: <iso8601>
```

Si `nvidia-smi` reporte un GPU différent (pod réassigné), abandonner le run et relancer.

## Évolutions prévisibles

- **Sprint 4** : multi-Oracle ajoute charge compute. RTX 5090 unique reste suffisant si on entraîne séquentiellement par domaine ; insuffisant si parallélisme requis. Réévaluer alors un upgrade RunPod (multi-GPU station, A100/H100 cloud).
- **Phase 5 OOD + N grand** : `N = 2¹⁶` peut nécessiter sharding ou sous-échantillonnage. Anticiper avant Sprint 4.
- **Aucune contrainte phase 5.3 ne suggère** un upgrade hardware pour HT, puisque la cible est `HT ≥ SSM` *sur le même hardware*.

## Sécurité opérationnelle

- Clés SSH (pour le tunnel pod→VPS), `MLFLOW_TRACKING_URI` : variables d'environnement injectées au démarrage du pod, **jamais commitées**.
- Le `.gitignore` couvre `.env`, `*.key`, `OPS/logs/mlflow/`, `mlruns/`, `OPS/logs/` (sauf compute_budget.md, runs_index.csv, manifests/), `**/checkpoints/`. Vérifier au Stage 0.6.
