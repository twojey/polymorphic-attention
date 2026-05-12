# CATALOGUE — Classification mathématique des propriétés d'attention

**Livrable Partie 1 — Science fondamentale.** Catalogue exhaustif des propriétés mathématiques candidates à mesurer sur les opérateurs d'attention (Oracles denses + attentions synthétiques), avec protocoles de test et indicateurs d'invariance.

Fusion de :
- **00b** : classification propriétés (23 catégories, 130+ items)
- **00c** : prédictions a priori par Oracle
- **00d** : sélection Oracles et protocole d'entrées

---

## § 1. Cadre théorique : la signature mathématique de la linéarité

### 1.1 Pourquoi une classification mathématique

La réduction de la complexité algorithmique d'un opérateur de boîte noire de O(N²) à O(N) ou O(N log N) **n'est pas une simple optimisation** : c'est le résultat de la découverte et de l'exploitation de **structures algébriques et analytiques sous-jacentes**.

Un opérateur peut être "fondamentalement linéaire" sans qu'on s'en rende compte si on n'identifie pas la structure qui le rend tel. Inversement, on peut chercher en vain à compresser un opérateur qui n'a aucune structure exploitable.

Le but de ce catalogue est de fournir une **batterie exhaustive de tests** pour **identifier la classe d'indistinguabilité minimale** d'un opérateur d'attention donné — c'est-à-dire la signature mathématique qui détermine si (et comment) il peut être implémenté en complexité réduite.

### 1.2 Quatre cadres théoriques unificateurs

#### (1) Rang de déplacement (Kailath, Pan)
Pour un couple d'opérateurs (A, B), le **déplacement** de T est :
```
∇_{A,B}(T) = AT - TB    (Sylvester)
Δ_{A,B}(T) = T - ATB^T  (Stein)
```

Le **rang de déplacement** rang(∇_{A,B}(T)) = r mesure l'écart de T à une structure idéale. Si r ≪ N, T est compressible en O(rN) paramètres.
- Toeplitz : A = B = Z (décalage), r ≤ 2
- Cauchy : A = diag(y)⁻¹, B = diag(x), r ≤ 1
- Vandermonde : A = D(x), B = Z

#### (2) Réalisation d'état (Ho-Kalman)
Si T est un système linéaire invariant dans le temps (LTI) avec paramètres de Markov G_k = C A^{k-1} B, on construit la matrice de Hankel des réponses impulsionnelles.

**Théorème de Ho-Kalman** : T a une réalisation d'état de dimension finie ssi rang(H) = n est fini. n est l'**ordre minimal** du système. Si n ≪ N, on peut récursifier T en O(nN).

#### (3) Matrices hiérarchiques (Hackbusch, Tyrtyshnikov)
Si la structure globale n'est pas apparente, T peut avoir un **rang faible par blocs**. Pour une partition récursive des indices, les blocs hors-diagonaux admissibles sont approximables par rang k ≪ N.

| Format | Stockage | Produit MV |
|--------|----------|-----------|
| **H-matrix** | O(Nk log N) | O(Nk log N) |
| **H²-matrix / HSS** | O(Nk) | **O(Nk)** |

La **nestedness** (emboîtement des bases parent-fils) est la clé du O(N) strict.

#### (4) Théorie des noyaux (Mercer, RKHS, RFF)
Si T représente un noyau K(x, y) symétrique défini positif (Mercer), il admet une décomposition en série tronquée à D ≪ N termes → opérateur O(N²) → O(ND).

**Théorème de Bochner** (noyaux stationnaires) → **Random Fourier Features** : approximation en O(ND).

### 1.3 Critère de "fundamental linearity"

Un opérateur est **fondamentalement linéaire** si l'un de ses invariants structurels est borné **indépendamment de N** :
- rang de déplacement r borné → compression O(rN)
- ordre minimal n du Hankel borné → récursification O(nN)
- nestedness borné → H²/HSS en O(N)
- D termes Mercer suffisants → kernel approx en O(ND)

Si **aucun** de ces invariants n'est borné quand N croît, l'opérateur est intrinsèquement O(N²) et **aucune approximation linéaire fidèle n'existe**. C'est un résultat scientifique fort.

---

## § 2. Catalogue exhaustif des propriétés (23 catégories, 130+ items)

### Vue d'ensemble

| Catégorie | Items | Cadre théorique |
|-----------|-------|-----------------|
| **A** Spectrales | rang, conditionnement, entropie, décroissance | spectral analysis |
| **B** Structurelles | Toeplitz, Hankel, Cauchy, Vandermonde, sparsité, blocs | rang de déplacement |
| **C** Statistiques par token | KL, entropies, variance, gradient, Fisher | information theory |
| **D** Géométriques | cosine sim, Frobenius, angles subspaces | linear algebra |
| **E** Information-théoriques | mutual info, compressibilité | information theory |
| **F** Dynamiques | Lipschitzness, stabilité temporelle | sensitivity |
| **G** Algébriques | trace/det, symétries, idempotence, polynômes | algebraic geometry |
| **H** Cross-layer | résidus, compositions, évolution rang | dynamics |
| **I** Cross-head | diversité, spécialisation, clustering | statistics |
| **J** Stochastiques | Markov-ness, stationnaire, mixing time | stochastic processes |
| **K** Topologiques | Laplacien, persistent homology, centralité | spectral graph / TDA |
| **L** Fréquentielles | FFT 2D, wavelets, quasi-périodicité | harmonic analysis |
| **M** Conditionnelles à l'entrée | sensitivity par type token, variation | functional analysis |
| **N** Comparatives Oracle/student | F-divergence, préservation distill | comparative |
| **O** Rang de déplacement | Toeplitz/Cauchy/Vandermonde-like | Kailath/Pan |
| **P** Réalisation d'état | rang Hankel, HSV, ordre minimal | Ho-Kalman |
| **Q** Hiérarchiques | rang faible blocs, H-matrix, HSS, nestedness | Hackbusch |
| **R** Noyau Mercer/RKHS | Mercer PD, RFF, Bochner, énergie tronquée | Mercer/Bochner |
| **S** Tenseurs | Tucker, Tensor Train, Hierarchical | tensor decomposition |
| **T** Équivariances | G-équivariance, permutation-invariance | representation theory |
| **U** Sparse-structurées | Butterfly, Monarch, Pixelfly | modern fast transforms |
| **V** Opérateurs analytiques | pseudo-différentiels, CZ, compacts | functional analysis / frontier |
| **W** Complexité logique | NIP, NTP₂, dependence measure | model theory |

### Propriétés clés implémentées ou prioritaires

#### **A — Propriétés spectrales**

| ID | Nom | Définition | Priorité |
|----|-----|-----------|----------|
| A1 | **Rang effectif r_eff par fenêtre K** | nb σ_i > τ·σ_max sur sous-matrice K×K | **PRIORITAIRE** |
| A4 | **Entropie spectrale** | H = -Σ p_i log p_i, p_i = σ_i²/Σσ_j² | **PRIORITAIRE** |
| A3 | Conditionnement | κ = σ_max / σ_min | Haute |
| A5 | Décroissance spectrale | régression log(σ_i) ~ -α·i | Haute |
| A6 | Participation ratio | PR = (Σσ_i²)² / Σσ_i⁴ | Médium |

#### **B — Propriétés structurelles**

| ID | Nom | Définition | Priorité |
|----|-----|-----------|----------|
| B2 | **Rang de Hankel** | rang numérique de H(A) | **PRIORITAIRE** |
| B1 | Toeplitz-ness | distance Frobenius à Toeplitz le plus proche | Haute |
| B5 | Block-diagonality | score structure block (Schmidt decomp) | Haute |
| B6 | Bandedness | largeur de bande effective | Haute |
| B4 | Sparsité effective | ratio coefficients > seuil | Médium |

#### **C — Propriétés statistiques par token**

| ID | Nom | Définition | Priorité |
|----|-----|-----------|----------|
| **C1** | **S_KL** | KL local vs baseline empirique | **PRIORITAIRE** |
| **C5** | **S_Spectral** | rang effectif r_eff fenêtre glissante K | **PRIORITAIRE** |
| C6 | S_Grad | ‖∇_x_t L_task‖ | Haute (phase 1.5) |
| C3 | Entropie Shannon par token | H(A[t,:]) | Médium |

#### **Autres catégories** (O-W)
Rang de déplacement, réalisation d'état, matrices hiérarchiques, noyaux Mercer, tenseurs, équivariances, sparse-structured, opérateurs analytiques, complexité logique. À implémenter en Sprint A2+.

---

## § 3. Sélection des Oracles et protocole d'entrées

### 3.1 Critères de sélection

1. **Architecture dense** uniquement (attention softmax O(N²))
2. **Open source** avec accès aux poids et matrices d'attention
3. **Diversité par domaine d'entraînement** (texte, code, vision, multimodal, biologique, contrôlé)
4. **Taille raisonnable** (≤ 8B pour rester sous ~$50 compute)
5. **Disponible via HuggingFace** ou téléchargeable

### 3.2 Sélection retenue (6 Oracles, Palier 1)

| Code | Oracle | Domaine | Taille | HF id |
|------|--------|---------|--------|-------|
| **OR** | Notre Oracle SMNIST V1 | Contrôlé (ω,Δ,ℋ connus) | ~5M | local (MLflow checkpoint) |
| **LL** | Llama 3.2 1B | Texte général | 1B | `meta-llama/Llama-3.2-1B` |
| **SC** | StarCoder 2 3B | Code (Python, multi-langages) | 3B | `bigcode/starcoder2-3b` |
| **DV** | DINOv2 ViT-B/14 | Vision pure (sans texte) | 86M | `facebook/dinov2-base` |
| **CL** | CLIP ViT-L/14 | Multimodal vision↔texte | 304M | `openai/clip-vit-large-patch14` |
| **ES** | ESM-2 35M | Biologique (protéines) | 35M | `facebook/esm2_t12_35M_UR50D` |

**Total compute estimé** : ~$10-20 sur RunPod GPU, ~2-3h par Oracle.

### 3.3 Datasets standardisés par domaine

| Domaine | Dataset | Sample taille | Longueur |
|---------|---------|---------------|----------|
| **Contrôlé** | SSG Structure-MNIST | 2000 ex | seq_len ≤ 4371 |
| **Texte** | WikiText-103 (val) | 1000 séq | 1024-4096 tokens |
| **Code** | HumanEval + GitHub sample | 500 fichiers | 1024-4096 tokens |
| **Vision** | ImageNet val sample | 1000 images | 224×224 → 256 patches |
| **Multimodal** | COCO captions | 500 paires | 224×224 + 32-77 tokens |
| **Biologique** | UniRef50 | 1000 séq | 100-500 acides aminés |

### 3.4 Procédure d'exécution

```
Pour chaque Oracle O dans {OR, LL, SC, DV, CL, ES}:
    1. Charger O + dataset(O.domaine)
    2. Sample N_sample séquences
    3. Forward avec output_attentions=True
    4. Extraire matrices A[ℓ, h, t, t'] FP32
    5. Pour chaque catégorie de propriétés (A, B, C, ...):
        a. Calculer invariants
        b. Logger dans MLflow
    6. Sauvegarder agrégats statistiques
    7. Reporter dans tableau global
```

---

## § 4. Prédictions a priori des signatures

**Document falsifiable** — pré-enregistrement des paris intuitifs confrontés aux mesures réelles.

Date d'enregistrement : **2026-05-11 ~11:00 UTC** (révision ~11:30 UTC)

### 4.1 Légende

- ✅ oui (forte conviction théorique)
- 🟢 oui (probable, indices empiriques)
- 🟡 partiel ou conditionnel
- 🔴 non (probable)
- ❌ non (forte conviction théorique)
- ❓ indécis → valeur scientifique maximale

### 4.2 Prédictions par catégorie

#### **Algébriques (G) — les Oracles denses sont quasi-identiques**

| Propriété | OR | LL | SC | DV | CL | ES |
|-----------|:--:|:--:|:--:|:--:|:--:|:--:|
| Linéarité en x | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Symétrie A=A^T | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Idempotence A²≈A | 🔴 | 🔴 | 🔴 | 🔴 | 🔴 | 🔴 |
| Stochasticité (sum=1) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Causalité (triangulaire) | ✅ | ✅ | ✅ | ❌ (bidir) | ❌ (bidir) | ❌ (bidir) |

**Pattern** : ces propriétés ne discriminent PAS. Seule causalité sépare autorégressifs (OR, LL, SC) vs bidirectionnels (DV, CL, ES).

#### **Spectrales (A) — où la diversité des domaines apparaît**

| Propriété | OR | LL | SC | DV | CL | ES |
|-----------|:--:|:--:|:--:|:--:|:--:|:--:|
| **A1 r_eff faible** | 🟢 | 🟢 | 🟡 | ✅ | 🟢 | 🟢 |
| A3 Conditionnement κ | 🟡 | 🟡 | 🟡 | 🟢 | 🟡 | 🟡 |
| A4 Entropie spectrale | 🟢 | 🟢 | 🟡 | ✅ | 🟢 | 🟢 |
| A5 Décroissance expo | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | 🟡 |
| A6 Participation ratio | 🟢 | 🟢 | 🟡 | ✅ | 🟢 | 🟢 |

**Pattern attendu** :
- **DINOv2** : rang effectif le plus **faible** (pixels voisins corrélés → low-rank)
- **StarCoder** : rang effectif le plus **haut** (logique précise = chaque token compte)
- **Résultat honnête attendu** : si DV ≠ le plus faible → revisiter hypothèse "vision = low-rank"

#### **Structurelles (B) — la signature compression**

| Propriété | OR | LL | SC | DV | CL | ES |
|-----------|:--:|:--:|:--:|:--:|:--:|:--:|
| **B1 Toeplitz-ness** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| B2 Rang Hankel | 🟡 | 🟡 | 🟡 | 🟢 | 🟡 | 🟡 |
| B4 Sparsité | 🟡 | 🟡 | 🟢 | ❌ | ❌ | ❌ |
| B5 Block-diag | 🔴 | 🔴 | 🟡 | 🟢 | 🟡 | 🔴 |
| B6 Bandedness | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟢 |

**Pattern attendu** :
- **DINOv2** : block-diagonality forte (patches voisins)
- **StarCoder** : sparsité + block (fonctions = blocs séparés)
- **ESM-2** : bandedness (interactions locales protéines)
- Pas de Toeplitz pur attendu (data-dépendance exclut ça)

#### **Rang de déplacement (O) + Ho-Kalman (P)**

| Propriété | OR | LL | SC | DV | CL | ES |
|-----------|:--:|:--:|:--:|:--:|:--:|:--:|
| O1 Toeplitz-like (r ≤ 2) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| O2 Cauchy-like (r ≤ 1) | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ |
| **P1 Rang Hankel fini** | ❌ | ❌ | 🟡 | 🟡 | ❓ | ❓ |
| P3 Décroissance HSV expo | 🟡 | 🟡 | 🟡 | 🟢 | 🟡 | 🟡 |
| Q2 Rang blocs faible | ❓ | 🟡 | 🟡 | 🟢 | 🟡 | 🟡 |
| Q4 Nestedness H²/HSS | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ |

**Pattern attendu** : DINOv2 candidat le plus fort pour H-matrices. Cauchy-like (O2) = **opportunité de découverte si trouvé**. Nestedness (Q4) = vraie inconnue partout.

#### **Mercer / RKHS (R) — discriminant null pour Oracles denses**

| Propriété | OR | LL | SC | DV | CL | ES |
|-----------|:--:|:--:|:--:|:--:|:--:|:--:|
| **R1 Positive-définie Mercer** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| R3 Stationnarité K(x-y) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

→ Pas attendu de kernel Mercer sur Oracles denses. Test de validation de la batterie.

### 4.3 Confrontation paris vs mesures

Format post-exécution :

| Pari (00c) | Mesure | Validé ? | Note |
|-----------|--------|----------|------|
| DV a r_eff le plus faible | val_DV vs autres | ✅/❌ | |
| SC a r_eff le plus haut | val_SC vs autres | ✅/❌ | |
| DV compatible H-matrix | rang blocs + nestedness | ✅/❌ | |
| ... | | | |

**Score global** : N/35 paris discriminants validés.

---

## § 5. Format du rapport batterie (post-exécution)

### 5.1 Tableau global propriété × Oracle

| Propriété | OR | LL | SC | DV | CL | ES | Variance |
|-----------|:--:|:--:|:--:|:--:|:--:|:--:|----------|
| A1 r_eff (médiane) | val | val | val | val | val | val | range |
| A4 entropie spectrale | val | val | val | val | val | val | range |
| B2 rang Hankel | val | val | val | val | val | val | range |
| ... | | | | | | | |

### 5.2 Signatures par Oracle

Pour chaque Oracle, résumé des classes mathématiques identifiées :

- "Llama 3.2 1B est compatible avec : low-rank effectif (A1), pas Mercer (R1 ❌), pas Toeplitz (B1 ❌)"
- "DINOv2 est compatible avec : H-matrix (Q2 ✅), spectre Laplacien (K1 ✅), low-rank fort (A1 ✅), block-diag (B5 ✅)"
- etc.

### 5.3 Découvertes inattendues

Section pour documenter les surprises qui contredisent les prédictions. Valeur scientifique maximale.

---

## § 6. Critères de qualité

- **Reproductibilité** : seeds fixés, datasets versionnés, code commit-référencé
- **Précision numérique** : invariants en FP32, contrôle erreurs quantification
- **Statistiques significatives** : N_sample ≥ 1000 pour Spearman/bootstrap (500 pour multimodal/code)
- **Validation invariance** : pour ≥ 1 Oracle, appliquer transformation (rotation random) et vérifier stabilité

---

## § 7. Extension Palier 2 (optionnel, ~$30-50 compute)

| Code | Oracle | Domaine |
|------|--------|---------|
| DC | DeepSeek-Coder 1.3B | Comparaison code |
| WP | Whisper-small encoder | Audio |
| DM | DeepSeek-Math 7B | Math/raisonnement |
| L8 | Llama 3.1 8B | Scaling texte |

**Recommandation** : commencer par Palier 1 (6 Oracles), étendre seulement si nécessaire.

---

## § 8. Scope explicite

### ✅ Inclus
- Architectures denses O(N²)
- Propriétés mathématiques mesurables via "boîte noire"
- Diversité domaines via Oracles indépendants
- Tests comparatifs cross-domaine

### ❌ Explicitement exclu
- Architectures sub-quadratiques (Mamba, Performer, Hyena, etc.)
- APIs propriétaires sans accès aux attentions
- Étude FFN, résidus, normalisation, embeddings

---

**Version** : 2026-05-12 | **Fusion** : 00b + 00c + 00d
