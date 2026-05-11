# 00c — Prédictions a priori des signatures mathématiques

**But.** Pré-enregistrer des **paris intuitifs** sur les résultats attendus de la batterie Partie 1 (DOC/00b) appliquée à différents Oracles d'attention. Document **falsifiable** : la confrontation de ces prédictions aux mesures réelles produit de la connaissance dans tous les cas.

> **Méthodologie** : ce document est **pré-enregistré avant exécution de la batterie**. Il NE doit PAS être modifié post-hoc en fonction des résultats. Toute modification doit être horodatée et justifiée. La valeur scientifique réside précisément dans la confrontation paris vs mesures.
>
> **Cadrage** : Partie 1 (science fondamentale, cf. DOC/00b). Pas spécifique au projet ASP — c'est un test de l'état de l'art théorique sur un panel d'opérateurs d'attention.

Date d'enregistrement : **2026-05-11 ~11:00 UTC**.

---

## 1. Oracles considérés (8 candidats représentatifs)

| Code | Oracle | Famille | Référence |
|---|---|---|---|
| **DT** | Transformer dense (softmax attention classique) | référence O(N²) | Vaswani et al. 2017 |
| **LA** | Linear Attention (kernel feature map) | kernel φ(Q)φ(K)^T | Katharopoulos et al. 2020 |
| **PF** | Performer (Random Fourier Features) | kernel approx aléatoire | Choromanski et al. 2021 |
| **LF** | Linformer (low-rank projection) | low-rank global k | Wang et al. 2020 |
| **MB** | Mamba / S4 (State-Space Model) | state-space causal | Gu & Dao 2023 |
| **HY** | Hyena (long convolution) | implicit conv FFT | Poli et al. 2023 |
| **BB** | BigBird (sparse global+local+random) | sparse hybride | Zaheer et al. 2020 |
| **RF** | Reformer (LSH attention) | hash-based sparse | Kitaev et al. 2020 |

## 2. Légende cellules

- ✅ oui (forte conviction théorique)
- 🟢 oui (probable, indices empiriques)
- 🟡 partiel ou conditionnel
- 🔴 non (probable)
- ❌ non (forte conviction théorique)
- ❓ indécis / pas d'intuition forte → **valeur scientifique de la mesure maximale**

---

## 3. Tableaux de prédictions

### Tableau 1 — Propriétés algébriques (G) + invariants linéaires fondamentaux

| Propriété | DT | LA | PF | LF | MB | HY | BB | RF |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **G5** Linéarité (en x) | ❌ | ❌ | ❌ | ❌ | 🟢 | 🟢 | ❌ | ❌ |
| G2 Symétrie A=A^T | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| G3 Idempotence A²≈A | 🔴 | 🔴 | 🔴 | 🔴 | 🔴 | 🔴 | 🔴 | 🔴 |
| Stochasticité ligne (sum=1) | ✅ | ✅ | ✅ | ✅ | ❓ | 🔴 | ✅ | ✅ |
| Causalité (triangulaire inf.) | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ | 🟡 |

**Patterns** : SSM/conv (MB, HY) sont les seuls **vraiment linéaires en x** (pas de softmax data-dependent). Le reste est non-linéaire à cause du softmax/kernel data-dep. La **symétrie est nulle pour tous** (Q/K asymétriques). **Linéarité G5 = test discriminant entre "data-dependent" vs "fixed-kernel"**.

### Tableau 2 — Spectrales (A) — où les sub-quadratiques laissent une signature

| Propriété | DT | LA | PF | LF | MB | HY | BB | RF |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **A1** r_eff faible par fenêtre K | 🟢 (concentré) | 🟢 | 🟢 | ❌ (rang≤k) | 🟢 | 🟡 | 🟡 | 🟢 |
| A3 Conditionnement κ borné | 🟡 | 🟢 | 🟢 | ❌ (mal cond.) | 🟢 | 🟢 | 🟡 | 🟡 |
| A4 Entropie spectrale faible | 🟢 | 🟢 | 🟢 | ✅ | 🟢 | 🟢 | 🟡 | 🟢 |
| A5 Décroissance σ_i exponentielle | 🟡 | 🟢 | 🟢 | ✅ (à r) | 🟢 | 🟢 | 🟡 | 🟡 |
| A6 Participation ratio bornée | 🟢 | 🟢 | 🟢 | ✅ | 🟢 | 🟢 | 🟡 | 🟢 |

**Patterns** : tout le monde est **low-rank effectif** (résultat empirique connu). Linformer est le SEUL avec **rang strictement borné** par construction (=k). Conditionnement de Linformer **mauvais** (rank-deficient → σ_min très petit).

### Tableau 3 — Structurelles (B) + Rang de déplacement (O) — la signature compression

| Propriété | DT | LA | PF | LF | MB | HY | BB | RF |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **B1** Toeplitz-ness (data-indep.) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| B2 Rang Hankel local faible | 🟡 | 🟢 | 🟢 | ✅ | ✅ | ✅ | 🟡 | 🟡 |
| B4 Sparsité effective | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| B5 Block-diagonality | 🔴 | ❌ | ❌ | ❌ | 🔴 | ❌ | ✅ (local) | 🟡 (LSH bucket) |
| B6 Bandedness (largeur faible) | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | 🟡 | ❌ |
| **O1** Toeplitz-like (rang dépl. ≤ 2) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| O2 Cauchy-like (rang dépl. ≤ 1) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Low-rank global | ❌ | ❌ | ❌ | ✅ | 🟡 | ❌ | ❌ | ❌ |

**Patterns clés** :
- **Hyena** est le SEUL Toeplitz vrai (long conv = convolution = Toeplitz par construction). Test de commutation Z·A = A·Z **doit passer** pour Hyena, échouer pour les autres.
- **Linformer** est explicitement low-rank global (= construction).
- **BigBird/Reformer** sont sparse mais pas low-rank.
- **Cauchy-like** : aucun Oracle classique ne tombe ici → opportunité de classe non-explorée.

### Tableau 4 — Réalisation Ho-Kalman (P) + Hiérarchique (Q)

| Propriété | DT | LA | PF | LF | MB | HY | BB | RF |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **P1** Rang Hankel global fini | ❌ | 🟢 | 🟢 | ✅ | ✅ (=d state) | ✅ | ❓ | 🟡 |
| P3 Décroissance HSV exponentielle | ❌ | 🟢 | 🟢 | ✅ | ✅ | 🟢 | ❓ | ❓ |
| **P4** Ordre minimal n borné | ❌ | 🟢 | 🟢 | ✅ (=k) | ✅ (=d) | 🟢 | ❓ | ❓ |
| Q2 Rang local blocs admissibles k faible | ❓ | 🟢 | 🟢 | ✅ | 🟡 | 🟢 | ❌ | ❓ |
| Q4 Nestedness (H²/HSS) | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❌ | ❓ |

**Patterns** : Ho-Kalman favorise les SSM (Mamba) et conv (Hyena) qui ont une **réalisation d'état explicite et bornée**. Pour DT (dense), HSV ne décroit pas → projet ASP repose sur l'**hypothèse** que sur des données structurées HSV décroit empiriquement. **Nestedness reste un point d'interrogation pour TOUS — peu d'études existent**.

### Tableau 5 — Mercer / RKHS (R)

| Propriété | DT | LA | PF | LF | MB | HY | BB | RF |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **R1** Positive-définitude Mercer | ❌ (softmax non-PD) | ✅ (kernel φφ^T) | ✅ | 🔴 | 🔴 | 🔴 | ❌ | 🔴 |
| R3 Stationnarité K(x-y) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| R4 Bochner (FT μ ≥ 0) | ❌ | 🟡 | ✅ | ❌ | ❌ | 🟡 | ❌ | ❌ |
| R6 Approximable par RFF | 🔴 | ✅ | ✅ (par construction) | ❌ | ❌ | 🟡 | ❌ | ❌ |

**Patterns** : Linear Attention et Performer SONT des kernels Mercer par construction (φ(x)φ(y)^T). Les autres ne respectent PAS Mercer (softmax dense fait des produits négatifs après normalisation). **R1 est LE test discriminant kernel-based vs autre**.

### Tableau 6 — Cross-layer / cross-head (H, I) + dynamiques (F)

| Propriété | DT | LA | PF | LF | MB | HY | BB | RF |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| H3 Évolution rang croissante avec ℓ | 🟢 | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ |
| I1 Diversité inter-heads forte | 🟢 | 🟡 | 🟡 | 🟡 | 🟡 (un état/head) | 🟡 | 🟢 | 🟢 |
| I2 Spécialisation des heads | 🟢 | ❓ | ❓ | ❓ | 🟢 | ❓ | ❓ | ❓ |
| **F1** Lipschitzness vs entrée | 🟡 (softmax) | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | 🟡 (LSH break) |

**Patterns** : ces propriétés sont **mal documentées** dans la littérature, beaucoup d'incertitudes. C'est précisément où la batterie Partie 1 apporterait du nouveau. Les SSM/conv ont une Lipschitzness explicite (linéaires), DT est non-Lipschitz à cause de softmax data-dep.

### Tableau 7 — Markov (J), équivariances (T), butterfly/sparse (U)

| Propriété | DT | LA | PF | LF | MB | HY | BB | RF |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| J1 Markov-ness propre (A^k cv) | 🟡 | 🟡 | 🟡 | ❓ | 🟢 | 🟡 | 🟡 | 🟡 |
| **T1** G-équivariance (translation) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (conv) | ❌ | ❌ |
| T3 Circulante (G = C_n) | ❌ | ❌ | ❌ | ❌ | ❌ | 🟡 (cyclic conv) | ❌ | ❌ |
| **U1** Butterfly (FFT-like) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (conv = FFT) | ❌ | ❌ |
| U2 Monarch | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

**Patterns** : Hyena domine sur équivariance translation + butterfly (parce que c'est une convolution). Les autres ne sont pas équivariants par construction.

---

## 4. Synthèse — Signatures attendues par classe d'Oracle

| Oracle | Signature mathématique attendue | Classe(s) qui le caractérisent |
|---|---|---|
| **DT** (dense) | Non-linéaire, non-PD, rang effectif faible empirique mais pas borné théorique, pas d'invariant structurel exploitable a priori | "Aucune classe pure" → besoin d'arguments empiriques (faible rang sur distribution réaliste) |
| **LA** (linear) | Mercer + Hankel-finite + Lipschitz | **R1, P1, F1** — kernel + state-space hybride |
| **PF** (Performer) | Mercer + RFF + Hankel-finite | **R1, R6, P1** — kernel approximation pure |
| **LF** (Linformer) | Low-rank global strict | **B(rang≤k)** — projection construction |
| **MB** (Mamba) | Linéaire en x + Ho-Kalman + Lipschitz + ordre minimal d | **G5, P1-P4, F1** — state-space |
| **HY** (Hyena) | Toeplitz + Butterfly/FFT + Equivariant | **B1, O1, T1, U1** — convolution longue |
| **BB** (BigBird) | Sparse-block | **B4, B5** — sparse hybride |
| **RF** (Reformer) | Sparse + bucket clustering | **B4, B5** (local par bucket) |

---

## 5. Incertitudes principales (zones de découverte potentielle maximale)

1. **Nestedness (Q4)** — peu de littérature mesurant cette propriété sur des opérateurs d'attention réels. **Vraie inconnue.** Aucune intuition pour aucun Oracle.
2. **Évolution rang cross-layer (H3)** — peu d'études systématiques. Hypothèse courante : "rang croit avec ℓ" mais pas mesuré rigoureusement.
3. **Markov-ness fine (J1-J4)** — interpréter A comme matrice de transition est mathématiquement valide mais peu exploré pour les attentions.
4. **Toutes les catégories K (TDA), L (wavelets), V (pseudo-différentiel)** — zéro intuition empirique. Tests vraiment expérimentaux.
5. **Cauchy-like (O2)** — aucun Oracle classique ne tombe ici a priori. Si on en trouve, c'est une découverte de classe non-explorée.

---

## 6. Falsifiabilité de mes paris

Si la batterie Partie 1 contredit mes paris **discriminants** :

| Pari pré-enregistré | Si invalidé → conclusion |
|---|---|
| Hyena est Toeplitz/Butterfly | Architecture Hyena mal comprise théoriquement OU implementations divergent de la spec |
| Performer est Mercer-PD empiriquement | Bug d'implem ou approximation pire que théorique |
| DT a rang effectif borné indépendamment de N | Résultat MAJEUR — justifie ASP dans cette direction |
| MB/HY sont vraiment linéaires en x | Si non → softmax-like component caché ou data-dep non documenté |
| Tous les Oracles ont r_eff faible empiriquement | Si non → surprise majeure, remet en cause toute la motivation des sub-quad |

**Valeur scientifique** : tester la batterie produit de la connaissance dans tous les cas — soit on confirme l'état de l'art (utile comme benchmark formel), soit on découvre un écart théorie/pratique (publiable).

---

## 7. Patterns transversaux remarqués

### Test discriminants identifiés (un seul OUI/NON suffit à classer)

- **G5 Linéarité** : sépare {MB, HY} des autres
- **R1 Mercer** : sépare {LA, PF} des autres
- **B1/O1 Toeplitz** : isole {HY}
- **Rang global borné** : isole {LF}
- **Sparsité** : sépare {BB, RF} des denses

### Tests redondants (corrélés à un autre)

- B2 (rang Hankel local faible) corrélé à A1 (r_eff par fenêtre)
- A3, A4, A5, A6 (spectraux) corrélés entre eux
- T1 et U1 (équivariance translation et butterfly) corrélés via convolution

### Tests à fort gain d'information (incertitude haute)

- Q4 (nestedness) — vraiment inconnu
- H3 (évolution rang cross-layer) — peu mesuré
- K (TDA), L (wavelets) — toutes propriétés
- N (comparatives Oracle/student) — non encore évaluable

---

## 8. Confrontation aux résultats (à compléter post-batterie)

Cette section reste vide jusqu'à ce que la batterie soit exécutée. Format proposé :

| Pari | Mesure réelle | Verdict | Surprise ? |
|---|---|---|---|
| (à remplir) | | | |

Score global sur 70 paris discriminants : **/70 paris validés** — à mesurer.

---

## Liens

- **Catalogue Partie 1** : [00b_classification_proprietes.md](00b_classification_proprietes.md)
- **Cadrage projet** : [00_vision.md](00_vision.md)
- **Roadmap batterie** : [../ROADMAP.md](../ROADMAP.md) section "Stage 1.5+ — Classification mathématique étendue"
- **Carnet** : [carnet_de_bord.md](carnet_de_bord.md) (entrée 2026-05-11 ~11:00)
