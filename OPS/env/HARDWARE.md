# HARDWARE.md — Hardware cible du projet ASP

**Statut** : décision prise (Stage 0.3, ROADMAP). Topologie à deux machines, training sur RunPod éphémère.

## Topologie

| Machine | Rôle | Persistance |
|---|---|---|
| **VPS** | Édition code, édition doc, lecture résultats W&B, push git, lancements de runs distants | Persistante, contient le repo source |
| **RunPod RTX 5090** | Training, extraction, audit spectral, calibration spectromètre | **Éphémère** — disque détruit à l'arrêt du pod |

Aucun training sur le VPS. Aucune édition long-terme sur le pod.

## Spécifications RunPod RTX 5090

| Paramètre | Valeur |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 (Blackwell, sm_120) |
| VRAM | 32 GB |
| Précision native | FP64, FP32, TF32, BF16, FP16, FP8 |
| CUDA | 12.8 (minimum, vérifier `nvidia-smi`) |
| Driver | ≥ 555 |

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
2. **Tous les artefacts persistants vivent ailleurs** : poids Oracle, matrices `A`, dictionnaire SCH → W&B Artifacts (cf. LOGGING.md). Jamais sur le disque du pod seul.
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

- Token W&B, clés SSH RunPod, etc. : variables d'environnement injectées au démarrage du pod, **jamais commitées**.
- Le `.gitignore` couvre `.env`, `*.key`, `wandb/`, `OPS/logs/`, `**/checkpoints/`. Vérifier au Stage 0.6.
