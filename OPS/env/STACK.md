# STACK.md — Stack ML du projet ASP

**Statut** : décision prise (Stage 0.1, ROADMAP). Cette stack vaut pour Sprint 1. Toute évolution doit être justifiée par un besoin émergent et tracée ici.

## Décision

| Composant | Choix | Version cible |
|---|---|---|
| Framework | **PyTorch** | ≥ 2.6 (CUDA 12.8, support Blackwell sm_120) |
| Boucle d'entraînement | **Lightning Fabric** | ≥ 2.4 |
| Configs d'expérience | **Hydra** | ≥ 1.3 |
| Gestion deps | **uv** | dernière stable |
| Format de config | YAML (compatible Hydra natif) | — |
| Python | 3.11 | (3.12 OK si toutes les deps suivent) |

## Justifications

### PyTorch (vs JAX/Flax, vs TF)

- **Hooks natifs** sur les modules `nn.MultiheadAttention` et équivalents → extraction de la matrice `A` par `(couche, tête, exemple)` sans intrusion. Critique pour phase 1.4 (extraction Oracle) et phase 2.1 (SVD batchée).
- **Écosystème SSM/attention** plus large que JAX. Permet de rester agnostique sur la famille structurée jusqu'à la phase 2 (principe Discovery > Reproduction).
- **Support Blackwell (RTX 5090)** stabilisé en PyTorch 2.6 ; JAX/XLA suit avec quelques mois de retard.
- **Précision mixte BF16 ↔ FP64** triviale via `torch.autocast` + cast manuel pour la SVD (cf. règle DOC/01 : BF16 entraînement, FP64 extraction).

### Lightning Fabric (vs Lightning full, vs PyTorch raw)

- **Boucle d'entraînement explicite et visible.** Phase 3 introduit `L_matriochka` (output-based, sommée sur tirages de `r`) et `L_consistency`. Le full Lightning oblige à passer par `LightningModule.training_step`, qui devient illisible avec une loss à plusieurs tirages internes par batch.
- **Récupération des bénéfices d'orchestration** : précision mixte, checkpointing, multi-device, callbacks, sans payer le coût d'abstraction.
- **Écrire la boucle à la main est faisable** mais tu réécriras Fabric en moins bien. Pas de raison de payer ce coût.

### Hydra (vs argparse, vs OmegaConf seul)

- La ROADMAP prévoit `OPS/configs/phaseX/*.yaml` avec un format commun. Hydra le supporte nativement.
- **Multirun sweep natif** indispensable phase 4 (5–7 valeurs de `λ_budget` log-spaced) et phase 5 (3 seeds × N ∈ {2¹⁰, 2¹², 2¹⁴, 2¹⁶}).
- **Composition de configs** : un run = `phase + dataset + oracle + seed`, chaque axe versionné séparément. Cohérent avec la règle T.1 (pré-enregistrement).

### uv (vs poetry, vs conda, vs pip-tools)

- **Lockfile reproductible** + résolution rapide. Critique pour RunPod éphémère : `uv sync` repart d'un lockfile commité, pas d'install à 5 minutes à chaque pod.
- Compatible avec les wheels CUDA 12.8 nightly nécessaires pour Blackwell.
- Alternative acceptable : `pip-tools` + `requirements.txt`. Pas conda (lent, lourd, non standard pour la recherche moderne).

## Choix explicitement écartés

| Écarté | Raison |
|---|---|
| **JAX + Flax/NNX** | Support Blackwell en retard ; écosystème SSM moins riche ; hooks d'instrumentation moins ergonomiques. À reconsidérer si phase 3 nécessite des kernels très custom (Toeplitz/Hankel/Cauchy non triviaux). |
| **Full Lightning (`LightningModule`)** | Trop d'abstraction pour les losses customs phase 3+. |
| **PyTorch raw** | Réécriture inévitable de Fabric. |
| **mamba-ssm en dépendance** | Violation Discovery > Reproduction (DOC/00 §4b). Recompilation Blackwell douloureuse. À utiliser **uniquement** comme baseline phase 5 (sequence-domain), jamais comme backbone. |
| **flash-attn en dépendance dure** | Idem : recompilation, et Flash Attention est "équivalent mathématique" → utilisable uniquement pour l'Oracle phase 1, à packager en extra optionnel. |

## Primitives à valider en Stage 0.2

À tester avant de cocher 0.2 dans la ROADMAP, sur une machine RTX 5090 réelle :

- [ ] `torch.linalg.svd` en FP64, batché — phase 2.1 (audit spectral)
- [ ] `torch.svd_lowrank` (randomisée) — phase 1.5 (signal `S_Spectral` sur fenêtre glissante)
- [ ] `torch.fft.rfft` causale, séquences ≥ 2¹⁶ — phase 3 (convolution Toeplitz longue)
- [ ] `torch.linalg.lstsq` batché — phase 2.6b batterie A (fitting `ε_C` par classe)
- [ ] `torch.func.vmap` — utile pour la batterie de tests structurels phase 2.6b
- [ ] Attention dense pure (pas Flash Attention) avec export `A` complet — phase 1 (Oracle), nécessite VRAM `O(N² · n_têtes · n_couches)` au moment de l'extraction.

Documenter les résultats dans `OPS/env/PRIMITIVES.md` (à créer Stage 0.2).

## Contraintes Blackwell (RTX 5090, sm_120)

- **CUDA 12.8 minimum.** PyTorch 2.6+ fournit des wheels CUDA 12.8.
- **Driver NVIDIA ≥ 555.** Vérifier sur le pod RunPod avant chaque run de référence (`nvidia-smi`).
- **FP8** dispo (Blackwell) mais **interdit phase 1–4**. Réservé à phase 5.3 (Hardware Throughput) si besoin de pousser HT au-dessus du SSM pur.
- **Recompilation des packages C++/CUDA tiers nécessaire** si on en introduit (raison supplémentaire de les éviter).

## Pinning et reproductibilité

- `pyproject.toml` géré par uv, lockfile `uv.lock` commité.
- Versions exactes pinnées pour : `torch`, `lightning`, `hydra-core`, `numpy`, `mlflow`. Le reste : ranges majeurs.
- Une expérience commitée doit être rejouable à `uv sync && python -m ...` sans intervention manuelle.

## Évolutions prévisibles

Sprint 1 : on s'en tient à la stack ci-dessus.
Réévaluation possible (et tracée ici) à :

- **Sortie phase 1** — si les hooks PyTorch ne suffisent pas pour l'extraction massive de `A`, envisager un mode "trace + dump" custom.
- **Sortie phase 2** — si la batterie de tests structurels (D notamment) nécessite des opérateurs très custom, JAX redevient candidat.
- **Sortie phase 4** — si le Spectromètre nécessite du fine-grained control flow par token, vérifier que Fabric ne devient pas un goulot d'étranglement.

Aucune migration n'est entreprise sans rapport de phase justifiant le besoin.
