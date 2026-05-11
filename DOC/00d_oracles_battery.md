# 00d — Sélection des Oracles et protocole d'entrées pour la batterie

**But.** Définir précisément QUELS Oracles tester avec la batterie de classification (DOC/00b), et SUR QUELLES DONNÉES, pour que les mesures soient comparables et reproductibles.

> **Cadrage** : Partie 1 — étude des **Oracles denses** uniquement (architectures O(N²)). Le but du projet ASP est de comprendre les attentions denses pour ensuite synthétiser leurs propriétés dans une couche sub-quadratique. Les architectures sub-quadratiques (Mamba, Performer, etc.) sont **hors scope de la batterie**.
>
> Date de fixation : **2026-05-11 ~11:30 UTC**.

---

## 1. Critères de sélection

1. **Architecture dense** uniquement (attention softmax O(N²))
2. **Open source** avec accès aux poids et matrices d'attention (pas d'API propriétaire)
3. **Diversité par domaine d'entraînement** (texte, code, vision, multimodal, biologique, contrôlé)
4. **Taille raisonnable** (≤ 8B pour rester sous ~$50 compute total)
5. **Disponible via HuggingFace** ou téléchargeable directement

## 2. Sélection retenue (6 Oracles, ~$10-20 compute)

| # | Code | Oracle | Domaine | Taille | HF id |
|---|---|---|---|---|---|
| 1 | **OR** | Notre Oracle SMNIST V1 | contrôlé | ~5M | local (artefact MLflow `s1_smnist_oracle_e2f0b5e_oracle.ckpt`) |
| 2 | **LL** | Llama 3.2 1B | texte général (langage naturel) | 1B | `meta-llama/Llama-3.2-1B` |
| 3 | **SC** | StarCoder 2 3B | code (Python, multi-langages) | 3B | `bigcode/starcoder2-3b` |
| 4 | **DV** | DINOv2 ViT-B/14 | vision pure (sans supervision texte) | 86M | `facebook/dinov2-base` |
| 5 | **CL** | CLIP ViT-L/14 | multimodal vision↔texte | 304M | `openai/clip-vit-large-patch14` |
| 6 | **ES** | ESM-2 35M (ou 150M) | biologique (séquences protéines) | 35-150M | `facebook/esm2_t12_35M_UR50D` (35M) ou `facebook/esm2_t30_150M_UR50D` (150M) |

**Total compute estimé** : ~$10-20 sur RunPod GPU 24-48 GB VRAM (L40 ou A6000), ~2-3h par Oracle pour extraction d'attentions sur protocole d'entrées + analyse.

### Extension possible (Palier 2, ~$30-50 compute total)

Si la batterie marche bien sur les 6 et qu'on veut élargir :

| # | Code | Oracle | Pourquoi |
|---|---|---|---|
| 7 | **DC** | DeepSeek-Coder 1.3B | Comparaison intra-domaine code (vs StarCoder) |
| 8 | **WP** | Whisper-small encoder | Audio (modalité différente) |
| 9 | **DM** | DeepSeek-Math 7B | Math/raisonnement (spécialisation logique) |
| 10 | **L8** | Llama 3.1 8B | Test scaling texte (vs Llama 3.2 1B) |

→ **Recommandation : commencer par les 6 du Palier 1, étendre seulement si nécessaire.**

## 3. Protocole d'entrées par domaine (CRITIQUE pour comparaisons valides)

La signature mathématique d'un Oracle dépend des **données d'entrée**. Pour que les mesures soient comparables :
- **Au sein d'un domaine** : tous les Oracles de ce domaine reçoivent les **mêmes données**
- **Cross-domaine** : on accepte que les données diffèrent (vision ≠ texte par nature), mais on documente les choix.

### Datasets standardisés par domaine

| Domaine | Dataset principal | Source | Sample taille | Longueur séquences |
|---|---|---|---|---|
| **Contrôlé** | SSG Structure-MNIST | local (DOC/01) | 2000 ex (déjà en cours) | seq_len max 4371 |
| **Texte général** | WikiText-103 (val) + The Pile sample | HF `Salesforce/wikitext` ou `monology/pile-uncopyrighted` | 1000 séquences | 1024-4096 tokens |
| **Code** | HumanEval + GitHub Python sample | HF `openai_humaneval` + `bigcode/the-stack-smol` | 500 fichiers | 1024-4096 tokens |
| **Vision pure** | ImageNet val sample | HF `imagenet-1k` (val split) | 1000 images | 224×224 → 256 patches |
| **Multimodal** | COCO captions sample | HF `nlphuji/flickr30k` ou `mscoco` | 500 paires (image+caption) | 224×224 + 32-77 tokens |
| **Biologique** | UniRef50 sample | HF `agemagician/uniref50` | 1000 séquences | 100-500 acides aminés |

### Paramètres communs

- **Précision extraction** : FP32 pour calcul des invariants (FP16 pour le forward du modèle)
- **Causalité** : respecter le mode natif du modèle (causal pour LL/SC/OR ; bidirectionnel pour DV/CL/ES)
- **Position des heads/layers extraits** : tous, ou échantillonnage représentatif (couches 0, L/4, L/2, 3L/4, L-1) si calcul trop lourd
- **Random seed** : fixé pré-batterie (seed = 42) pour reproductibilité du sampling

## 4. Procédure d'exécution (par Oracle)

```
Pour chaque Oracle O dans {OR, LL, SC, DV, CL, ES}:
    1. Charger O (HF) + dataset(O.domaine)
    2. Sample N_sample séquences/exemples
    3. Forward avec output_attentions=True (HF) ou hook custom
    4. Extraire matrices A[ℓ, h, t, t'] FP32, batch par batch
    5. Pour chaque catégorie de propriétés (A, B, C, ...):
        a. Calculer les invariants définis dans 00b
        b. Logguer dans MLflow (un sub-run par Oracle)
    6. Sauvegarder agrégats statistiques (médiane, IQR, distribution)
    7. Reporter dans tableau global (cf. format §5)
```

**Coût par Oracle** : ~30-90 min selon taille modèle + nombre de propriétés mesurées.

## 5. Format du rapport batterie (à produire post-exécution)

### 5.a. Tableau global propriété × Oracle

| Propriété (catégorie 00b) | OR | LL | SC | DV | CL | ES | Variance cross-Oracle |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| A1 r_eff (médiane) | val | val | val | val | val | val | range |
| A4 entropie spectrale | val | val | val | val | val | val | range |
| ... | | | | | | | |

### 5.b. Confrontation paris (DOC/00c) vs mesures

| Pari (00c) | Mesure | Validé ? | Note |
|---|---|---|---|
| DV a r_eff le plus faible | val_DV vs autres | ✅/❌ | |
| SC a r_eff le plus haut | val_SC vs autres | ✅/❌ | |
| ... | | | |

Score global : **N/35 paris discriminants validés**.

### 5.c. Signatures par Oracle

Pour chaque Oracle, identifier la (les) classe(s) mathématique(s) la (les) plus proche(s) :
- "Llama 3.2 1B est compatible avec : low-rank effectif (A1), block-diag faible (B5), pas Mercer (R1 ❌)"
- "DINOv2 est compatible avec : H-matrix (Q2 ✅), spectre Laplacien interp. (K1 ✅), low-rank fort (A1 ✅)"
- etc.

## 6. Critères de qualité

- **Reproductibilité** : tous les seeds fixés, datasets versionnés, code commit-référencé
- **Précision numérique** : invariants en FP32, controle des erreurs de quantification (cf. DOC/00b §IV.2 conditionnement)
- **Statistiques significatives** : N_sample ≥ 1000 pour Spearman/bootstrap (sauf 500 pour multimodal/code par contrainte coût)
- **Validation invariance** : pour ≥ 1 Oracle, appliquer transformation (rotation random) et vérifier stabilité des invariants sensés être invariants par similarité

## 7. Ce que la batterie va apprendre (objectifs explicites)

### Objectif 1 — Test de la classification (Partie 1)
- Quelle classe mathématique correspond à chaque Oracle dense diversifié ?
- La classe est-elle stable cross-domaine ou fortement dépendante du domaine d'entraînement ?

### Objectif 2 — Validation des paris a priori (DOC/00c)
- Score de prédiction → mesure de l'**état de l'art théorique** sur les attentions
- Surprises éventuelles → axes de recherche

### Objectif 3 — Préparer la synthèse sub-quadratique (Partie 2)
- Identifier les invariants partagés par TOUS les Oracles denses → propriétés à respecter dans une approximation sub-quad
- Identifier les invariants spécifiques au domaine → axes de spécialisation pour ASP-vision, ASP-code, ASP-bio, etc.

## 8. Ce qui est explicitement EXCLU du scope

- ❌ Architectures sub-quadratiques (Mamba, Performer, Linformer, Hyena, etc.) — hors scope batterie
- ❌ APIs propriétaires (Claude, GPT-4) — pas d'accès aux attentions
- ❌ Modèles > 8B (test scaling reportable Sprint dédié)
- ❌ Étude de phénomènes hors-attention (FFN, normalisation, etc.) — DOC/00 §0 scope ASP

## 9. Calendrier prévisionnel (à dérouler après Sprints dev classification)

| Phase | Oracles | Coût compute | Durée wall-clock |
|---|---|---|---|
| **Validation batterie** | OR (SMNIST) | 0 | ~30 min |
| **Pilote langage** | LL (Llama 3.2 1B) | ~$1 | ~1h |
| **Pilote vision** | DV (DINOv2 ViT-B) | ~$1 | ~1h |
| **Code + multimodal + bio** | SC + CL + ES | ~$5 | ~3h |
| **Rapport** | — | 0 | ~3h dev |
| **Total Palier 1** | 6 Oracles | ~$10-20 | ~10h |

---

## Liens

- **Catalogue propriétés** : [00b_classification_proprietes.md](00b_classification_proprietes.md)
- **Paris a priori** : [00c_predictions_signatures.md](00c_predictions_signatures.md)
- **Cadrage projet** : [00_vision.md](00_vision.md)
- **Roadmap** : [../ROADMAP.md](../ROADMAP.md) section "Stage 1.5+"
- **Carnet** : [carnet_de_bord.md](carnet_de_bord.md) (entrée 2026-05-11 11:30)
