# 00b — Classification mathématique des propriétés d'attention

**But.** Catalogue exhaustif des propriétés mathématiques candidates à mesurer sur les opérateurs d'attention (Oracle dense + attentions synthétiques), avec protocoles de test "boîte noire" et indicateurs d'invariance. Constitue le **livrable Partie 1 — Science fondamentale** du projet ASP.

> **Cadrage projet.** Le projet ASP a deux livrables distincts (cf. carnet 2026-05-11 décision 10:15) :
> - **Partie 1 — Science fondamentale** : classification mathématique + batterie de tests, réutilisable par toute la communauté.
> - **Partie 2 — Validation hypothèse polymorphique** : application à l'attention sub-quadratique via allocation dynamique guidée par signal observable.
>
> Ce document structure la Partie 1. La Partie 2 (cf. DOC/01b) utilise un sous-ensemble (S_KL, S_Grad, S_Spectral) comme signaux candidats.

---

## I. Cadre théorique : la signature mathématique de la linéarité

### I.1 Pourquoi une classification mathématique

La réduction de la complexité algorithmique d'un opérateur de boîte noire de O(N²) à O(N) ou O(N log N) **n'est pas une simple optimisation** : c'est le résultat de la découverte et de l'exploitation de **structures algébriques et analytiques sous-jacentes**.

Un opérateur peut être "fondamentalement linéaire" sans qu'on s'en rende compte si on n'identifie pas la structure qui le rend tel. Inversement, on peut chercher en vain à compresser un opérateur qui n'a aucune structure exploitable.

Le but de la **Partie 1** est de fournir une **batterie exhaustive de tests** pour **identifier la classe d'indistinguabilité minimale** d'un opérateur d'attention donné — c'est-à-dire la signature mathématique qui détermine si (et comment) il peut être implémenté en complexité réduite.

### I.2 Concepts unificateurs

Quatre cadres théoriques traversent l'ensemble du catalogue :

#### (1) Rang de déplacement (Kailath, Pan)
Pour un couple d'opérateurs (A, B), le **déplacement** de T est :
∇_{A,B}(T) = AT - TB    (Sylvester)
Δ_{A,B}(T) = T - ATB^T  (Stein)

Le **rang de déplacement** rang(∇_{A,B}(T)) = r mesure l'écart de T à une structure idéale. Si r ≪ N, T est compressible en O(rN) paramètres. Spécialise :
- Toeplitz : A = B = Z (décalage), r ≤ 2
- Cauchy : A = diag(y)⁻¹, B = diag(x), r ≤ 1
- Vandermonde : A = D(x), B = Z

**Invariance majeure** : le rang de déplacement est invariant par similarité (V T W^T avec opérateurs transformés A' = V A V⁻¹, B' = W B W⁻¹).

#### (2) Réalisation d'état (Ho-Kalman)
Si T est un système linéaire invariant dans le temps (LTI) avec paramètres de Markov G_k = C A^{k-1} B, on construit la matrice de Hankel des réponses impulsionnelles :

H = [G_1 G_2 G_3 ...; G_2 G_3 G_4 ...; ...]

**Théorème de Ho-Kalman** : T a une réalisation d'état de dimension finie ssi rang(H) = n est fini. n est l'**ordre minimal** du système. Si n ≪ N, on peut récursifier T en O(nN).

**Invariance** : l'ordre minimal et les valeurs singulières de Hankel (HSV) sont indépendants de la base de réalisation choisie.

#### (3) Matrices hiérarchiques (Hackbusch, Tyrtyshnikov)
Si la structure globale n'est pas apparente, T peut avoir un **rang faible par blocs** (block-wise low-rank). Pour une partition récursive des indices, les blocs hors-diagonaux admissibles (clusters distants) sont approximables par rang k ≪ N.

| Format | Admissibilité | Bases | Stockage | Produit MV |
|---|---|---|---|---|
| **H-matrix** | standard | indépendantes par bloc | O(Nk log N) | O(Nk log N) |
| **H²-matrix / HSS** | faible (tous blocs) | **emboîtées** (parent-fils) | O(Nk) | **O(Nk)** |

La propriété de **nestedness** (emboîtement des bases) est la clé du O(N) strict.

#### (4) Théorie des noyaux (Mercer, RKHS, RFF)
Si T représente un noyau K(x, y) symétrique défini positif (Mercer), il admet une décomposition en série :
K(x, y) = Σ λ_i φ_i(x) φ_i(y)

Si tronquable à D ≪ N termes : feature map Φ(x) ∈ ℝ^D telle que K(x,y) ≈ Φ(x)^T Φ(y) → opérateur O(N²) → O(ND).

Pour noyau **stationnaire** (K(x-y)) : théorème de **Bochner** → K PD ssi sa transformée de Fourier est mesure positive → approximation par **Random Fourier Features** : ω_j ∼ μ, Φ(x) = e^{iω_j^T x}, K̃(x,y) = (1/D) Σ cos(ω_j^T(x-y)).

### I.3 Critère de "fundamental linearity"

Un opérateur est **fondamentalement linéaire** si l'un de ses invariants structurels est borné **indépendamment de N** :
- rang de déplacement r borné → compression O(rN)
- ordre minimal n du Hankel borné → récursification O(nN)
- nestedness borné → H²/HSS en O(N)
- D termes Mercer suffisants → kernel approx en O(ND)

Si **aucun** de ces invariants n'est borné quand N croît, l'opérateur est intrinsèquement O(N²) et **aucune approximation linéaire fidèle n'existe**. C'est un résultat scientifique fort.

---

## II. Catalogue exhaustif des propriétés

23 catégories ouvertes, ~130+ items. Statut : ◆ implémenté, ◇ partiel, ○ à coder.

### Vue d'ensemble

| Catégorie | Items | Cadre théorique principal |
|---|---|---|
| **A** Spectrales | rang, conditionnement, entropie, décroissance, participation, discriminant, résultant | spectral analysis + elimination theory |
| **B** Structurelles | Toeplitz, Hankel, Cauchy, Vandermonde, sparsité, blocs, bandes, tropical, Sylvester | rang de déplacement + tropical geometry |
| **C** Statistiques par token | KL, entropies, variance, gradient, Fisher, Wasserstein | information theory + differential geometry |
| **D** Géométriques | cosine sim, Frobenius, angles subspaces, Fisher metric, wave fronts, characteristic variety | linear algebra + microlocal geometry |
| **E** Information-théoriques | mutual info, compressibilité | information theory |
| **F** Dynamiques | Lipschitzness, stabilité temporelle | sensitivity |
| **G** Algébriques | trace/det, symétries, idempotence, polynômes, Bernstein-Sato, D-modules, syzygies | linear + algebraic geometry |
| **H** Cross-layer | résidus, compositions, évolution rang, convergence | dynamics |
| **I** Cross-head | diversité, spécialisation, clustering | statistics |
| **J** Stochastiques (Markov) | Markov-ness, stationnaire, mixing time | stochastic processes |
| **K** Topologiques / graphes | Laplacien, persistent homology, centralité | spectral graph / TDA |
| **L** Fréquentielles | FFT 2D, wavelets, quasi-périodicité, symbol calculus, CZ regularity | harmonic analysis + microlocal |
| **M** Conditionnelles à l'entrée | sensitivity par type token, variation vs (ω,Δ,ℋ) | functional analysis |
| **N** Comparatives Oracle/student | F-divergence, préservation distill, Lipschitz différentielle | comparative |
| **O** Rang de déplacement | Toeplitz/Cauchy/Vandermonde-like, generators | Kailath/Pan |
| **P** Réalisation d'état | rang Hankel block, HSV, ordre minimal | Ho-Kalman |
| **Q** Hiérarchiques | rang faible blocs, H-matrix, HSS, nestedness | Hackbusch/Tyrtyshnikov |
| **R** Noyau Mercer/RKHS | Mercer PD, RFF, Bochner, énergie tronquée, Clifford, Arf, Hasse-Witt | Mercer/Bochner + algebraic invariants |
| **S** Tenseurs | Tucker, Tensor Train, Hierarchical, MERA/PEPS, PARAFAC | tensor decomposition |
| **T** Équivariances | G-équivariance, permutation-invariance, circulantes, block-circulantes, invariants rotation | representation theory |
| **U** Sparse-structurées | Butterfly, Monarch, Pixelfly, block-sparse, random sparse+low-rank | modern fast transforms |
| **V** Opérateurs analytiques | pseudo-différentiels, CZ, compacts, Wishart, analytique, microlocal ellipticity, prismathic, Donaldson-Thomas, categorical | functional analysis + frontier |
| **W** Complexité logique | NIP, NTP₂, Littlestone, Shelah stability, dependence measure, definability | model theory + learning theory |

### A — Propriétés spectrales

| ID | Nom | Définition | Statut | Code |
|---|---|---|---|---|
| A1 | Rang effectif r_eff par fenêtre K | nb σ_i > τ·σ_max sur sous-matrice K×K | ◆ | `phase1b/signals/s_spectral.py` |
| A2 | Spectre singulier complet | distribution {σ_i} normalisée | ○ | — |
| A3 | Conditionnement | κ = σ_max / σ_min | ○ | — |
| A4 | Entropie spectrale | H = -Σ p_i log p_i, p_i = σ_i²/Σσ_j² | ◆ | `phase1/metrics/spectral.py` |
| A5 | Décroissance spectrale | régression log(σ_i) ~ -α·i | ○ | — |
| A6 | Participation ratio | PR = (Σσ_i²)² / Σσ_i⁴ | ○ | — |
| A7 | Discriminant du polynôme caractéristique | Disc(det(λI - A)) mesure séparation des valeurs propres | ○ | — |
| A8 | Résultant spectral | Resultant(P(λ), P'(λ)) capture multiplicités algébriques | ○ | — |
| A9 | Weyl équidistribution | test statistique des espaces inter-valeurs propres vs GUE | ○ | — |

### B — Propriétés structurelles (entrée vers catégories O et Q)

| ID | Nom | Définition | Statut | Code |
|---|---|---|---|---|
| B1 | Toeplitz-ness | distance Frobenius à Toeplitz le plus proche | ○ | — |
| B2 | Rang de Hankel | rang numérique de H(A) (sub-diagonale ou ligne) | ◆ | `phase1/metrics/hankel.py` |
| B3 | Cauchy/Vandermonde-ness | distance aux familles paramétriques | ○ | — |
| B4 | Sparsité effective | ratio coefficients > seuil | ○ | — |
| B5 | Block-diagonality | score structure block (Schmidt decomp) | ○ | — |
| B6 | Bandedness | largeur de bande effective | ○ | — |
| B7 | Tropical degeneracy | hiérarchie magnitude log-échelle : ordonnancement des termes dominants | ○ | — |
| B8 | Sylvester matrix rank | rang de la matrice de Sylvester pour GCD polynomial implicite | ○ | — |
| B9 | Quasiseparable structure | rangs de déplacement généralisés (générateurs courts) | ○ | — |

### C — Propriétés statistiques par token

| ID | Nom | Définition | Statut | Code |
|---|---|---|---|---|
| C1 | S_KL vs baseline empirique | KL(A[t,:] ‖ baseline) | ◇ V1.5 amendée | `phase1b/signals/s_kl.py` |
| C2 | S_KL vs uniform causale | KL(A[t,:] ‖ uniform_t) | ○ | — |
| C3 | Entropie Shannon par token | H(A[t,:]) | ○ | — |
| C4 | Entropie Rényi | H_α = (1-α)⁻¹ log Σ p_i^α (α=2 typique) | ○ | — |
| C5 | Variance par tête | var(A[t,h,:]) | ○ | — |
| C6 | S_Grad | ‖∇_x_t L_task‖ | ○ jamais codé | `phase1b/signals/s_grad.py` (squelette) |
| C7 | Fisher information matrix | Hessian de KL divergence sur famille softmax ; métrique Riemannienne | ○ | — |
| C8 | Nombre de condition Fisher | κ_F(I_fisher) ; curvature of softmax manifold | ○ | — |
| C9 | Wasserstein distance | distance optimal transport entre distribution attentions | ○ | — |

### D — Propriétés géométriques inter-heads

| ID | Nom | Définition | Statut |
|---|---|---|---|
| D1 | Cosine similarity entre lignes/têtes | sim(A[h_1,t,:], A[h_2,t,:]) | ○ |
| D2 | Distance Frobenius vs uniform/identité | ‖A - U‖_F, ‖A - I‖_F | ○ |
| D3 | Angle entre subspaces | principal angles entre row-spaces de heads | ○ |
| D4 | Variété statistique (Fisher metric) | Ricci scalar de la métrique Fisher sur espace paramètres softmax | ○ |
| D5 | Wave front set (microlocal) | WF(A) = singularités en (position, fréquence) ; support micro-local | ○ |
| D6 | Characteristic variety | variété algébrique en fibré cotangent ; analyse opérateur différentiel | ○ |
| D7 | Curvature concentrations | loci de haute courbure → identifie concentrations d'information | ○ |

### E — Propriétés information-théoriques

| ID | Nom | Définition | Statut |
|---|---|---|---|
| E1 | Mutual information entre tokens | I(t_1; t_2) via attention | ○ |
| E2 | Compressibilité | proxy LZ ou rang spectral | ○ |

### F — Propriétés dynamiques

| ID | Nom | Définition | Statut |
|---|---|---|---|
| F1 | Lipschitzness vs entrée | ‖A(x+δ) - A(x)‖ / ‖δ‖ | ○ |
| F2 | Stabilité temporelle | ‖A_t - A_{t+1}‖ | ○ |

### G — Propriétés algébriques

| ID | Nom | Définition | Statut |
|---|---|---|---|
| G1 | Trace, déterminant | tr(A), det(A) | ○ |
| G2 | Symétrie / antisymétrie | ‖A - A^T‖ / ‖A‖ | ○ |
| G3 | Idempotence | ‖A² - A‖ / ‖A‖ | ○ |
| G4 | Polynôme caractéristique / minimal | coefficients, racines | ○ |
| G5 | **Additivité / Linéarité** | ‖A(x+y) - A(x) - A(y)‖ / max(‖A(x)‖,‖A(y)‖) | ○ |
| G6 | Bernstein-Sato polynomial | b(s) : équation fonctionnelle det(A)^{s+1} = b(s)·det(A)^s | ○ | — |
| G7 | D-module structure | annihilateurs polynomiaux ; holonomicité | ○ | — |
| G8 | Syzygies | relations entre colonnes/rangées (module des syzygies) | ○ | — |

### H — Propriétés cross-layer

| ID | Nom | Définition | Statut |
|---|---|---|---|
| H1 | Résidu inter-layers | ‖A_ℓ - A_{ℓ+1}‖_F | ○ |
| H2 | Composition | A_ℓ × A_{ℓ+1} (propagation effective) | ○ |
| H3 | Évolution du rang à travers profondeur | r_eff(ℓ) trajectoire | ○ |
| H4 | Convergence vers stationnaire layer→∞ | distance A_ℓ vs A_∞ extrapolé | ○ |

### I — Propriétés cross-head (intra-couche)

| ID | Nom | Définition | Statut |
|---|---|---|---|
| I1 | Diversité inter-heads | var_h(A[ℓ,h,:,:]) | ○ |
| I2 | Spécialisation | entropie inter-heads, cluster-ability | ○ |
| I3 | Corrélation/cluster heads | hierarchical clustering sur pairs (h_1, h_2) | ○ |

### J — Propriétés stochastiques (Markov)

| ID | Nom | Définition | Statut |
|---|---|---|---|
| J1 | Markov-ness | A^k convergence (Perron-Frobenius) | ○ |
| J2 | Distribution stationnaire | eigenvector dominant π de A^T | ○ |
| J3 | Mixing time | t_mix(ε) = min t : ‖A^t - π‖_TV < ε | ○ |
| J4 | Réversibilité (detailed balance) | π_i A_{ij} ≈ π_j A_{ji} | ○ |

### K — Propriétés topologiques / théorie des graphes

| ID | Nom | Définition | Statut |
|---|---|---|---|
| K1 | Spectral graph theory | Laplacien L = D - A, spectre L | ○ |
| K2 | Persistent homology | TDA sur A vue comme distance/poids | ○ |
| K3 | Centralité (PageRank-like) | eigencentrality | ○ |
| K4 | Communautés / clustering | Louvain, modularité | ○ |

### L — Propriétés fréquentielles

| ID | Nom | Définition | Statut |
|---|---|---|---|
| L1 | FFT 2D | spectre Fourier 2D de A | ○ |
| L2 | Wavelets | analyse multi-échelle (Daubechies, Haar) | ○ |
| L3 | Quasi-périodicité | autocorrélation, peaks de Fourier | ○ |
| L4 | Symbol calculus (microlocal) | symboles polaires a(x,ξ) ; singularités fréquentielles par classe S^m_{ρ,δ} | ○ | — |
| L5 | Calderon-Zygmund regularity | caractérisation par noyau CZ(α) ; Hölder exponent | ○ | — |
| L6 | Spectral clustering microlocal | clusters fréquentiels ; persistance vs (ω, Δ, ℋ) | ○ | — |

### M — Propriétés conditionnelles à l'entrée

| ID | Nom | Définition | Statut |
|---|---|---|---|
| M1 | Sensitivity par type de token | E[A[t,:] \| type(t) ∈ {digit, op, noise, pad}] | ○ |
| M2 | Variation A vs (ω, Δ, ℋ) | corrélations Spearman globales | ◇ partiel | `phase1b/bench/spearman.py` |

### N — Propriétés comparatives Oracle vs student

| ID | Nom | Définition | Statut |
|---|---|---|---|
| N1 | F-divergence Oracle/student | KL, JS, χ² entre A_oracle et A_student | ○ |
| N2 | Préservation des propriétés sous distillation | écart sur chaque propriété A-M | ○ |
| N3 | Lipschitzness différentielle | différence de sensitivity Oracle vs student | ○ |

### O — Rang de déplacement (Kailath / Pan)

| ID | Nom | Définition | Test invariant |
|---|---|---|---|
| O1 | **Toeplitz-like** : rang(T - ZTZ^T) | ≤ 2 pour toute Toeplitz, généralisable | invariant similarité |
| O2 | **Cauchy-like** : rang(T - D_y⁻¹ T D_x) | ≤ 1 pour toute Cauchy | invariant similarité |
| O3 | **Vandermonde-like** : rang(T - D_x T Z) | ≤ 1 pour toute Vandermonde | invariant similarité |
| O4 | **Generators (G, B)** | T = somme produits factorisée par déplacement | longueur des générateurs |
| O5 | **Largeur de récurrence** (Pan) | structure plus fine que rang de déplacement | invariant fin |

Lien complexité : rang(L_{A,B}(T)) = r → produit MV en **O(rN log N)** ou **O(rN)** selon classe. Conditionnement de l'opérateur L à monitorer (stabilité numérique).

### P — Réalisation d'état (Ho-Kalman)

| ID | Nom | Définition | Test invariant |
|---|---|---|---|
| P1 | **Rang du Hankel block** des réponses impulsionnelles | rang(H) où H_{ij} = G_{i+j} = CA^{i+j-2}B | invariant base |
| P2 | **Hankel Singular Values (HSV)** | σ_i(H), invariants globaux énergie par état | invariant base |
| P3 | **Décroissance des HSV** | σ_i ≤ C·exp(-α·i) → réalisation d'ordre faible | quantifie compressibilité |
| P4 | **Ordre minimal n** | n = rang(H), dimension espace d'états minimal | invariant fondamental |
| P5 | **Stabilité algorithme Ho-Kalman** | convergence SVD H tronqué | sensibilité numérique |
| P6 | **Realisation LPV** (paramètres variants) | observabilité + span-reachability conjointes | extension non-LTI |

Lien complexité : rang(H) = n borné → récursification O(nN).

### Q — Matrices hiérarchiques (Hackbusch / Tyrtyshnikov)

| ID | Nom | Définition | Test invariant |
|---|---|---|---|
| Q1 | **Admissibilité η** des blocs (t,s) | min(diam(V_t), diam(V_s)) ≤ η·dist(V_t, V_s) | partition-dépendant |
| Q2 | **Rang local k** des blocs admissibles | rang(A\|_{t×s}) ≈ k via SVD/ACA | invariant base interne |
| Q3 | **Profondeur hiérarchique** | nombre de niveaux récursifs | dépend de la partition |
| Q4 | **Nestedness** (H²-matrix) | U_parent = bloc-diag(U_{c1}, U_{c2}) X (transfert père-fils) | invariant à l'arborescence canonique |
| Q5 | **Croissance norme générateurs** | stabilité numérique sous partition profonde | indicateur de stabilité |
| Q6 | **Énergie résiduelle hors-cluster admissible** | ratio Frobenius des blocs non-admissibles | qualité de la partition |

Lien complexité : H-matrix O(Nk log N), H²/HSS **O(Nk)**.

### R — Noyau Mercer / RKHS

| ID | Nom | Définition | Test invariant |
|---|---|---|---|
| R1 | **Positive-définitude (Mercer)** | tous c_i c_j K(x_i, x_j) ≥ 0 ; Gram K_{ij} = K(x_i, x_j) PSD | invariant base unitaire |
| R2 | **Décroissance valeurs propres λ_i** | λ_i normalisées → tronquable à D ≪ N | invariant troncation |
| R3 | **Stationnarité K(x-y)** | invariance par translation entrée | invariant translation |
| R4 | **Bochner** : densité spectrale μ ≥ 0 | F.T. de K est mesure positive | équivalent stationnarité PD |
| R5 | **Energy Ratio Criterion (ERC)** | (Σ_{i≤D} λ_i) / (Σ_i λ_i) → seuil pour D | indicateur de troncation |
| R6 | **Random Fourier Features** | erreur sup_x \|K̃ - K\| ∼ O(1/√D) (Bernstein) | bound concentration |
| R7 | **Operator-Valued Kernels** | extension vectorielle, opérateur intégral compact | extension |
| R8 | **Clifford algebra signature** | signature quadratique induite (p+, q−, r dégénérés) de A^T A | invariant algébrique |
| R9 | **Arf invariant** | invariant mod 8 pour formes quadratiques ; détecte orientabilité algébrique | invariant fin |
| R10 | **Hasse-Witt symbol** | produit fini des symboles de Hilbert ; invariant local-global | invariant archiméd |

Lien complexité : K tronquable à D ≪ N → O(ND) au lieu de O(N²).

### S — Décompositions tensorielles structurées

Pour A vu comme tenseur d'ordre supérieur (par ex. (L, H, N, N)) ou via reshapes de matrices N×N en tenseurs.

| ID | Nom | Définition | Test invariant |
|---|---|---|---|
| S1 | **Tucker decomposition** | rangs multilinéaires (r_1, ..., r_d) faibles | invariants de Tucker |
| S2 | **Tensor Train (TT)** | rangs TT bornés (Oseledets 2011) | TT-rank invariant base |
| S3 | **Hierarchical Tucker (HT)** | profondeur d'arborescence + rangs feuilles | invariant arbre canonique |
| S4 | **Tensor Network (MERA, PEPS)** | réseau de tenseurs avec rangs locaux faibles | invariants topologiques |
| S5 | **CANDECOMP/PARAFAC (CP)** | rang CP minimal (Kruskal) | rang CP NP-hard exact |

Lien complexité : TT-rank r borné → produit MV en O(r² N log N). HT en O(rN). Très efficace pour tenseurs d'ordre élevé.

### T — Équivariances par groupe / symétries

Une matrice T équivariante sous un groupe G satisfait T·ρ(g) = ρ(g)·T pour tout g ∈ G (où ρ est une représentation).

| ID | Nom | Définition | Test invariant |
|---|---|---|---|
| T1 | **G-équivariance** générale | commutation avec ρ(g) pour groupe G donné (cyclique, dihédral, symétrique S_n, ...) | invariant action de G |
| T2 | **Permutation-invariance** | T·P = P·T pour permutations P (cas G = S_n) | matrices doublement stochastiques + structure |
| T3 | **Matrices circulantes** (cas G = C_n) | équivariance translation cyclique → diagonalisable par DFT | spectres = DFT |
| T4 | **Block-circulantes** (cas G produit) | circularité par blocs → BlockDFT | rangs équivariants |
| T5 | **Algèbre de groupe** | T appartient à C[G] / fonction sur G | dimension ≤ |G| |
| T6 | **Matrices invariantes par rotation** (SO(d)) | applications via tenseurs sphériques | rangs harmonique |

Lien complexité : G-équivariance avec |G| = m → O(m²) paramètres → O(m·N log N) via diagonalisation par caractères. Cas circulant : O(N log N) via FFT.

### U — Sparse-structurées (butterfly, Monarch)

Matrices issues du croisement structure + sparsité. Fondement des transformées rapides modernes (FFT, Hadamard, etc.) et des architectures Dao et al. (Pixelfly, Monarch).

| ID | Nom | Définition | Test invariant |
|---|---|---|---|
| U1 | **Butterfly** (Cooley-Tukey) | factorisation en log N matrices sparse permutées (FFT) | structure de récurrence |
| U2 | **Monarch** (Dao 2022) | produit de deux block-diagonales avec permutation | rangs blocs faibles |
| U3 | **Pixelfly / BFLY** (Chen 2021) | sparsité fixe en motif butterfly | motif fixe paramétré |
| U4 | **Block-sparse** | sparsité par blocs (motifs locaux + globaux) | longueur des connections |
| U5 | **Random sparse + low-rank** (Reformer-like) | LSH + sparse attention | qualité approximation LSH |

Lien complexité : Butterfly → O(N log N), Monarch → O(N√N) ou O(N log N) selon profondeur. Toutes admettent un produit MV linéaire en N pour une qualité fixée.

### V — Opérateurs analytiques / pseudo-différentiels

Cadre fonctionnel-analyste : T comme opérateur intégral avec noyau analytique.

| ID | Nom | Définition | Test invariant |
|---|---|---|---|
| V1 | **Opérateurs pseudo-différentiels** (Hörmander, Weyl) | symbol calculus, classes S^m_{ρ,δ} | classe symbol |
| V2 | **Calderon-Zygmund operators** | intégrales singulières, kernel ∈ CZ(α) | régularité Hölder noyau |
| V3 | **Opérateurs intégraux compacts** | Hilbert-Schmidt (Σλ_i² < ∞) ou trace-class (Σ\|λ_i\| < ∞) | classes Schatten |
| V4 | **Matrices résultantes / Sylvester / Bezout** | provenant de polynômes algébriques | rangs algébriques |
| V5 | **Matrices de Wishart** (statistiques) | M = X X^T / n, X ∼ Gaussian | spectre Marchenko-Pastur |
| V6 | **Matrices à noyau analytique** | T_{ij} = K(x_i, x_j) avec K analytique → décroissance exponentielle σ_i | rate exponentielle |
| V7 | **Microlocal ellipticity** | ellipticité partielle en (x, ξ) ; principal symbol invertible hors locus singulier | invariant microlocal |
| V8 | **Prismathic cohomology** (frontier 2023+) | cohomologie de complexes prismatiques ; détecte structure p-adic → perfectoid | cohomologie algébrique |
| V9 | **Donaldson-Thomas invariants** (frontier 2021+) | déformations algébriques ; invariants énumératifs sous perturbation stress | invariant énumérat |
| V10 | **Categorical weight filtration** | graduation par poids ; perverse sheaf decomposition | invariant catégorique |

Lien complexité : opérateurs CZ → précondition pour FMM (Fast Multipole Method) en O(N log N) ou O(N). Compacts → décroissance λ_i contrôle compression.

### W — Complexité logique / Stabilité modèle-théorique

Perspective logique : l'opérateur A défini implicitement par ses relations algébriques (équations, dépendances) encodes une "complexité de définissabilité" mesurable via théorie des modèles.

| ID | Nom | Définition | Test invariant |
|---|---|---|---|
| W1 | **NIP (No Independent Pairs)** | absence de paires indépendantes dans formule de définissabilité ; learnability upper bound | invariant logique |
| W2 | **NTP₂ (No Tree Property of order 2)** | absence de branchement arborescent dans formules ; stabilité supérieure | invariant logique fin |
| W3 | **Littlestone dimension** | dimension VC généralisée pour adversarial online learning ; board size | invariant complexité |
| W4 | **Shelah stability rank** | rang de stabilité pour théorie complète de A ; classification selon S(n) | invariant théorique |
| W5 | **Dependence measure (mutual dimension)** | quantification des dépendances multiples ; information-theoretic stabilization | invariant dépendance |
| W6 | **Definability complexity** | nombre de quantificateurs minimaux pour définir A en logique du premier ordre | invariant syntaxique |

Lien complexité : Stabilité NIP ↔ algorithmes d'apprentissage efficaces. NTP₂ ↔ bornes PAC complexité. Littlestone ↔ lower bounds online.

---

**Récap classes A-W (couverture exhaustive de l'état de l'art mathématique 2024 sur opérateurs structurés)** :

**Groupes stratégiques** :
1. **Linéaires-dépendantes-d'input** : A, C, D, F, M — mesures par évaluation sur données stressées
2. **Structurelles "matrix algebra"** : B, G, O, P, Q, R, S, T, U, V — théorie matrices/tenseurs/groupes + elimination theory + frontière
3. **Cross-axes** (layers/heads) : H, I — propagation et diversité
4. **Statistico-stochastiques** : E, J, K, L — caractères probabilistes + analyse microlocale
5. **Comparatives + Logique** : N, W — relativité Oracle vs student + expressibilité algébrique
6. **Frontier (2023+)** : V8-V10, W — cohomologie prismatique, Donaldson-Thomas, instabilité modèle-théorique

**Propriété clé** : Si **aucun** invariant de A-W n'est satisfait par un Oracle, l'opérateur n'appartient à aucune classe théorisée connue → soit O(N²) intrinsèquement, soit classe mathématique nouvelle à théoriser (publication math + révision théorie opérateurs).

### II.7 Enrichissements intégrés depuis Atlas (mai 2026)

**Intégration Tier 1 (ASP-critical)** — 16 propriétés nouvelles :
- **Elimination theory** (A7, A8, B8) : discriminant/résultant polynôme caractéristique, rang Sylvester
- **Fisher information geometry** (C7, C8, D4) : métrique Riemannienne sur espace softmax, courbure variétés statistiques
- **Microlocal analysis** (D5, D6, L4, L5, L6) : wave front set, characteristic variety, symbol calculus, CZ regularity, clustering microlocal
- **Tropical geometry** (B7) : hiérarchie magnitude log-échelle (dominance structure)
- **Holonomic systems** (G6, G7, G8) : Bernstein-Sato polynomial, D-module structure, syzygies

**Intégration Tier 2 (scientific completeness)** — 19 propriétés nouvelles :
- **Model-theoretic stability** (W1-W6) : NIP, NTP₂, Littlestone dimension, Shelah rank, dependence measure, definability complexity
- **Clifford/algebraic invariants** (R8, R9, R10) : quadratic signature, Arf invariant, Hasse-Witt symbol
- **Categorical framework** (V10) : weight filtration, perverse sheaf decomposition
- **Frontier concepts** (V8, V9) : prismathic cohomology, Donaldson-Thomas invariants

**Impact** : Catalogue étendu de ~70 à ~130+ propriétés. Couverture mathématique 2024 : élimination poly, géométrie différentielle, analyse microlocale, géométrie tropicale, holonomie algébrique, stabilité logique, invariants algébriques, catégoriques, et frontière (prismathique, énumératif).

**Universalité attendue** : Les Tier 1 propriétés (surtout elimination theory + Fisher + microlocal) doivent distinguer les grandes classes attention. Les Tier 2 enrichissent pour publication Partie 1. Les frontier (Tier 3, V8-V9) restent exploratoires — testables à stress extrême ℋ.

### II.8 Infrastructure d'extraction (refactor 2026-05-11)

L'infrastructure d'extraction des matrices d'attention (`CODE/phase1_metrologie/oracle/extract.py`) a été refactorée pour supporter le catalogue étendu A-W et les besoins mémoire des phases 2+.

**Trois APIs disponibles** :

| API | Cas d'usage | Pic mémoire FP64 |
|---|---|---|
| `extract(...)` | Batch processing avec besoin de toutes les couches (S_KL, S_Spectral cross-layer max-pool) — backward compat phase 1.5 | L × (B,H,N,N) |
| `extract_per_layer(...)` | Audit per-layer indépendant (phase 2 SVD, r_eff par couche) | 1 × (B,H,N,N) |
| `extract_streamed(..., callback)` | Streaming compute (signaux par couche, sauvegarde disque) | 1 × (B,H,N,N) |
| `extract_windowed_per_layer(..., K=64)` | Fenêtres K×K diagonales (phase 2, r_eff windowed, A1) | 1 × (B,H,K,K) |

**ExtractorConfig** :
- `fp64: bool = True` — cast FP64 (par défaut) ou conserve natif
- `validate_numerics: bool = False` — check NaN/Inf/range[0,1] post-softmax
- `empty_cache_per_layer: bool = True` — `torch.cuda.empty_cache()` entre couches
- `max_layers: int | None` — limit nb couches (debug, dev)
- `stream_to_disk: Path | None` — sauvegarde `layer_NNN.pt` par couche (très grandes seq_len)

**Limites** :
- Le forward Oracle reste monolithique : toutes attentions matérialisées dans buffers `.last_attn` pendant le forward. Le streaming réduit la mémoire FP64 + downstream, PAS le pic du forward (~B·L·H·N²·dtype_size).
- Pour seq_len > ~8192 : refactor transformer.py avec hook per-layer (TODO phase 2, cf. DOC/02 §extraction).

**Lien avec catalogue A-W** :
- A1, A4 (r_eff, entropie spectrale) → `extract_windowed_per_layer(K)` puis SVD locale
- A7-A9 (discriminant, résultant) → `extract_per_layer` puis `torch.linalg.eigvals` + polynômes
- B1-B9 (structurelles) → `extract_per_layer` puis distances Frobenius / projections
- C7-C9 (Fisher) → `extract_per_layer` puis dérivées numériques
- D5-D7, L4-L6 (microlocal) → `extract_per_layer` + FFT 2D + symbol estimation
- P1-P4 (Hankel) → `extract_per_layer` + construction Hankel + SVD
- Q1-Q6 (hiérarchique) → `extract_per_layer` + admissibilité + ACA

---

## III. Protocoles de test "boîte noire"

Tests applicables à un opérateur A vu comme fonction Input → Output, sans accès au code source.

### III.1 Test de linéarité / additivité (G5)
**Input** : x, y vecteurs.
**Procédure** : comparer A(x + y) à A(x) + A(y) ; A(c·x) à c·A(x).
**Output** : ε = ‖A(x+y) - A(x) - A(y)‖ / max(‖A(x)‖, ‖A(y)‖).
**Verdict** : ε ≪ 1 → linéaire ; sinon mesure de la non-linéarité.

### III.2 Test de shift / Toeplitz (O1)
**Input** : impulsions de Dirac δ_i, δ_{i+k}.
**Procédure** : sorties h_i = A(δ_i), h_{i+k} = A(δ_{i+k}). Si stationnaire : h_{i+k}(n) = h_i(n - k).
**Output** : corrélation entre h_{i+k} et shift(h_i, k).
**Test commutation** : Z·A vs A·Z (Z = matrice de décalage).
**Verdict** : commutation → opérateur Toeplitz.

### III.3 Test de rang de Hankel (P1, P2, P4)
**Input** : impulsions δ_0, δ_1, ..., δ_{m-1}.
**Procédure** :
1. Collecter G_k = A(δ_k) (paramètres de Markov).
2. Construire H bloc-Hankel : H_{ij} = G_{i+j}.
3. SVD(H) → spectre σ_i.
**Output** : décroissance des σ_i, gap spectral.
**Verdict** : gap net à n → ordre minimal n. Décroissance rapide → réalisation d'ordre faible.

### III.4 Test de rang de déplacement (O1, O2, O3, O4)
**Input** : choix d'opérateurs (A, B) candidats (Z, D_x, D_y, etc.).
**Procédure** : estimer L_{A,B}(T) via probe vectors aléatoires (T n'est pas connu explicitement).
1. Générer x random, calculer T·x et A·T·B^T·x.
2. Estimer dimension du sous-espace spanné par (T(x), A T(B^T x)) via SVD partielle.
**Output** : rang projeté.
**Verdict** : rang faible → structure correspondante (Toeplitz, Cauchy, etc.).

### III.5 Test H-matrice / HODLR / HSS (Q1, Q2, Q4)
**Input** : partition récursive des indices (binarisation).
**Procédure** : pour chaque bloc admissible (t, s), estimer rang via SVD aléatoire ou ACA. Vérifier que rang ≤ k pour petit k.
**Test nestedness** : vérifier que U_parent = bloc-diag(U_{c1}, U_{c2}) X (transfert exprimable).
**Output** : profil de rangs par niveau, présence/absence du transfert père-fils.
**Verdict** : rangs bornés → H-matrix ; nestedness vérifiée → H²/HSS → O(N).

### III.6 Test de positivité Mercer (R1)
**Input** : ensemble de points (x_i)_{i≤m}.
**Procédure** : construire Gram K_{ij} = K(x_i, x_j) (ou ⟨A(e_{x_i}), A(e_{x_j})⟩ via outputs).
**Output** : valeurs propres de K.
**Verdict** : toutes ≥ 0 → noyau PD (Mercer-compatible) ; négatives → non-Mercer.

### III.7 Test Bochner / RFF (R3, R4, R6)
**Input** : noyau supposé stationnaire K(x-y).
**Procédure** :
1. Échantillonner ω_j selon densité spectrale μ.
2. Construire φ_j(x) = e^{iω_j^T x}.
3. Vérifier (1/m) Σ_j φ_j(x) φ_j(y)* ≈ K(x-y).
**Output** : erreur sup et taux de décroissance avec D.
**Verdict** : décroissance ∼ O(1/√D) (Bernstein) → RFF valide.

### III.8 Test d'audit hiérarchique (Q1-Q6)
**Input** : masquage permettant d'isoler les blocs (t, s) avec t ∩ s = ∅.
**Procédure** : évaluer ∂y_t / ∂x_s (sensibilité matricielle).
**Output** : rang numérique de la matrice de sensibilité.
**Verdict** : stable avec N croissant → structure hiérarchique confirmée.

---

## IV. Indicateurs de stabilité et invariance

Une signature mathématique robuste **doit résister à un changement de base** pour être intrinsèque à l'opérateur, pas à sa représentation.

### IV.1 Invariance par similarité (changement de base)
- **Rang de déplacement** : invariant sous T → V T W^T, A → V A V⁻¹, B → W B W⁻¹.
- **Rang de Hankel** (ordre minimal LTI) : invariant pour toute réalisation.
- **HSV** : invariants globaux, indépendants de la forme (compagne, diagonale, modale).
- **Décomposition Mercer** : positivité indépendante de la base de l'espace des fonctions.

**Test pratique** : appliquer une transformation aléatoire (Fourier, wavelet, rotation) sur les entrées et vérifier que les invariants ne bougent pas.

### IV.2 Conditionnement et stabilité numérique
- **Nombre de condition de l'opérateur de déplacement** : κ(L_{A,B}) faible → récupération stable des générateurs.
- **Norme des matrices de transfert** (HSS) : croissance bornée sous partition profonde → stabilité.
- **Énergie résiduelle Mercer (ERC)** : ratio (Σ_{i≤D} λ_i) / (Σ_i λ_i) doit être stable sous bruit.

### IV.3 Stabilité sous perturbation
- **Bornes Bernstein** pour RFF : concentration de mesure quantifiable.
- **Décroissance des HSV** : décroissance rapide → robustesse au bruit en compression.

---

## V. Roadmap d'implémentation

### V.0 Priorisation pragmatique (décision 2026-05-11 10:40)

Le catalogue est exhaustif (Partie 1 complète) mais l'implémentation se fait en **deux temps** pour optimiser le coût/valeur :

#### V.0.a — **Phase prioritaire ASP** (~25-35h dev, ~$10-20 compute)
Items directement utiles à valider l'**hypothèse polymorphique** (Partie 2) ET à fournir une étude structurelle minimale (Partie 1 partielle).

**Signaux candidats supplémentaires** (compléter S_KL/S_Spectral en cours) :
- A4 (entropie spectrale par fenêtre)
- B2 (rang Hankel local — branchement signal)
- C6 (S_Grad — Sprint dédié)

**Étude structurelle minimale** :
- A2-A6 (spectre complet, conditionnement, décroissance, participation)
- G1-G5 (trace, déterminant, **G5 linéarité critique**, symétrie, idempotence)
- H1-H4 (évolution rang cross-layer — essentiel Spectromètre)
- I1-I3 (diversité cross-head — justifie max-pool)
- B1, B5, B6 (Toeplitz, block, band)
- O1-O2 (rang de déplacement Toeplitz/Cauchy-like)
- P1-P4 (Hankel global + HSV + ordre minimal)
- R1 (positivité Mercer — test discriminant pour kernel-based vs autre)

**Sprints (priorisés)** :
- Sprint 1a — Tests boîte noire fondamentaux (G5, O1, P1-P4, R1) : 6-10h
- Sprint 2 — Spectral + statistique (A2-A6, C2-C5) : 3-5h
- Sprint 3a — Algébrique + structurel partiel (G1-G5, B1, B5, B6) : 4-6h
- Sprint 5a — Cross-layer/head (H1-H4, I1-I3) : 5-8h
- Sprint S_Grad dédié (C6) : 3-5h dev (signal manquant H2)

#### V.0.b — **Phase Partie 1 enrichie Tier 1 + 2** (~60-80h dev, ~$20-40 compute, après V.0.a ou parallèle si budget)

**Tier 1 items** (ASP-adjacent, science-critical) :
- A7, A8, A9 (discriminant, résultant, Weyl equidistribution) — polynôme caractéristique
- B7, B8, B9 (tropical, Sylvester, quasiseparable) — structures magnitude + rank generalized
- C7, C8, C9 (Fisher, Wasserstein, diffusion geometry) — geometric statistics
- D4, D5, D6, D7 (Fisher curvature, wave front, characteristic variety, microlocal) — geometric-microlocal
- G6, G7, G8 (Bernstein-Sato, D-module, syzygies) — algebraic holonomic systems
- L4, L5, L6 (symbol calculus, CZ regularity, microlocal clustering) — microlocal analysis + harmonic
- R8, R9, R10 (Clifford, Arf, Hasse-Witt) — algebraic quadratic invariants
- V8, V9, V10 (prismathic, Donaldson-Thomas, categorical weight) — frontier algebraic geometry

**Tier 2 items** (science completeness) :
- W1-W6 (NIP, NTP₂, Littlestone, Shelah, dependence, definability) — model-theoretic stability
- Q1-Q6 (matrices hiérarchiques H/HSS) — compression hiérarchique
- R2-R7 (Mercer décomposition, Bochner, RFF, ERC) — kernel approximation
- J1-J4 (Markov / state-space) — stochastic processes
- K1-K4 (topologie / graphes / TDA) — topological data analysis
- E1-E2 (information-théoriques) — information geometry
- F1-F2 (dynamiques sensitivity) — functional sensitivity
- N1-N3 (comparatives Oracle/student) — distillation analysis

**Sprints (priorisés Tier 1 > Tier 2)** :
- **Sprint Tier1a — Elimination + Polynomial invariants** (A7-A9, B8) : 4-6h dev
- **Sprint Tier1b — Fisher + Differential Geometry** (C7-C9, D4) : 5-7h dev
- **Sprint Tier1c — Microlocal analysis** (D5-D7, L4-L6) : 6-10h dev (frontend, image processing)
- **Sprint Tier1d — Tropical + Holonomic systems** (B7, B9, G6-G8) : 5-7h dev
- **Sprint Tier1e — Clifford + Frontier (V)** (R8-R10, V8-V10) : 6-8h dev (algebraic geometry)
- **Sprint Tier1f — Logical Complexity (W)** (W1-W6) : 4-6h dev (model theory, PAC bounds)
- **Sprint Tier2a — Hiérarchique + Mercer** (Q, R2-R7) : 10-15h dev
- **Sprint Tier2b — Markov + Topo + Freq** (J, K, remaining L) : 12-18h dev
- **Sprint Tier2c — Info-th + Dyn + Comparatives** (E, F, N) : 6-10h dev

### V.1 Détail des Sprints

Ordonnée par **priorité scientifique** (combien d'items distinguent les grandes classes structurelles) et par **coût dev**.

### Sprint Classification 1 — Tests boîte noire fondamentaux (priorité critique)
- **G5** Test de linéarité (probe x+y vs x, y)
- **O1** Test Toeplitz/shift commutation
- **P1, P2** Test rang Hankel + HSV (Ho-Kalman)
- **R1** Test positive-définitude Mercer
- ~6-10h dev. Pose les fondations pour distinguer "linéaire / Toeplitz / Hankel-finite / Mercer".

### Sprint Classification 2 — Compléter spectral + statistique
- A2-A6 (spectre complet, conditionnement, décroissance, participation)
- C2-C5 (KL uniform, entropies Shannon/Rényi, variance)
- ~3-5h dev. Réutilise infra phase 1.5 existante.

### Sprint Classification 3 — Structurel + algébrique
- B1, B3-B6 (Toeplitz/Cauchy distance, sparsité, block, band)
- G1-G4 (trace/det, symétrie, idempotence, polynômes)
- O2-O5 (Cauchy-like, Vandermonde-like, generators, recurrence width)
- ~5-8h dev.

### Sprint Classification 4 — Hiérarchique (H-matrix, HSS)
- Q1-Q6 (admissibilité, rang local, nestedness, transfert, stabilité)
- ~8-12h dev (libraries spécialisées : éventuellement `scipy` + ACA custom).

### Sprint Classification 5 — Cross-layer / cross-head + dynamiques
- H1-H4, I1-I3, F1-F2
- ~5-8h dev.

### Sprint Classification 6 — Markov + topologique + fréquentiel + RFF
- J, K, L (Markov, TDA, FFT, wavelets)
- R2-R7 (Mercer décomposition, Bochner, RFF, ERC)
- ~12-18h dev (libraries : networkx, gudhi, pywavelets, numpy.fft).

### Sprint Classification 7 — Information-théoriques + comparatives
- E1-E2, N1-N3
- ~6-10h dev. Conditionnel à existence d'un student validé pour N.

### Sprint Classification 8 — Multi-Oracle (Sprint 4)
- Réplication de tous les tests précédents sur autres Oracles (CIFAR, texte, etc.)
- Mesure de l'**universalité** : quelles propriétés sont stables cross-domain ?
- Coût : multiplie le temps de calcul, peu de dev nouveau.

---

## VI. Convention d'évaluation et de rapport

Pour chaque propriété, la batterie fournit :

1. **Mesure scalaire ou vectorielle** par (couche, tête, token) selon item
2. **Agrégation cross-layer/head** (max-pool head + concat layer comme phase 1.5 §3, ou autre selon item)
3. **Statistiques descriptives** : médiane, IQR, distribution
4. **Corrélation Spearman** vs paramètres de stress (ω, Δ, ℋ) avec IC95 bootstrap
5. **Sensitivity analysis** : variation selon hyperparamètres (K, seuils)
6. **Test d'invariance** : application transformation similarité → mesure de stabilité

Le **rapport Partie 1** présente :
- **Matrice complète** (propriété × dimension de stress × Oracle)
- **Verdict structurel** : à quelle(s) classe(s) (Toeplitz, Hankel-finite, H², Mercer, ...) chaque Oracle appartient
- **Comparaison cross-Oracle** : signature de la "physique de l'attention" universelle vs domain-specific
- **Recommandations** : pour chaque Oracle, complexité minimale atteignable et mécanisme suggéré

---

## VI.1 Priorité implémentation post-enrichissement (mai 2026)

**Pour validation H2 (ASP)** — continuer avec phases 2-5 :
- Tier 1c (microlocal : D5-D7) peut servir pour **phase 2 spectral audit** (signature locale fréquentielle)
- Tout autre Tier 1 enrichit Partie 1 science mais ne bloque pas ASP

**Pour publication Partie 1 complète** — après verdict ASP :
- Implémenter Tier 1a-f (priorité : Tier1c > Tier1a > Tier1d) pour couverture mathématique robuste
- Ajouter Tier 2 (Logical Complexity W prioritaire pour structure dépendance)
- Tier 2 (Hierarchical Q, Mercer R2-R7) utiles pour algorithmic complexity reporting

**Budget temps estimé** :
- Tier 1 complet : 30-40h dev (peut paralleliser avec phase 2 pilot)
- Tier 2 complet : 30-40h dev
- Total catalogue vers exhaustivité : 60-80h après V.0.a (ASP sprint)

---

## VII. Références bibliographiques

### Matrices structurées et rang de déplacement
- Kailath, T., Sayed, A. H. (1995) — *Displacement structure: theory and applications*. SIAM Review.
- Pan, V. Y. (2001) — *Structured Matrices and Polynomials*. Birkhäuser.
- Chandrasekaran, S. et al. (ISSAC 2018) — Rang de déplacement généralisé, FFT-réduction Toeplitz→Cauchy.
- Olshevsky, V., Tyrtyshnikov, E. — Travaux sur low-rank structure.

### Réalisation de systèmes
- Ho, B. L., Kalman, R. E. (1966) — *Effective construction of linear state-variable models from input/output functions*. Regelungstechnik.
- Moore, B. C. (1981) — *Principal component analysis in linear systems: Controllability, observability, and model reduction*. IEEE TAC.
- Juang, J.-N., Pappa, R. S. (1985) — Eigensystem Realization Algorithm (ERA).

### Matrices hiérarchiques
- Hackbusch, W. (1999) — *A sparse matrix arithmetic based on H-matrices*. Computing.
- Hackbusch, W. (2015) — *Hierarchical Matrices: Algorithms and Analysis*. Springer.
- Tyrtyshnikov, E. (1996) — Bases du rang faible par blocs.
- Lin, L. et al. (JCP 2011) — HSS pour basses rangs de Hankel.
- Xia, J. (2016) — Stabilité des factorisations hiérarchiques.

### Théorie des noyaux
- Mercer, J. (1909) — *Functions of positive and negative type*. Phil. Trans. Royal Soc.
- Aronszajn, N. (1950) — *Theory of reproducing kernels*. Trans. AMS.
- Rahimi, A., Recht, B. (NIPS 2007) — *Random Features for Large-Scale Kernel Machines*.
- Bochner, S. — Harmonic Analysis and the Theory of Probability.

### Stabilité numérique
- Gu, M. (1995) — Stabilisation algorithme de Schur.
- Higham, N. J. (2002) — *Accuracy and Stability of Numerical Algorithms*. SIAM.

### Théorie de l'élimination & Résultants polynomiaux
- Macaulay, F. S. (1902) — *The Algebraic Theory of Modular Systems*. Cambridge UP.
- van der Waerden, B. L. (1950) — *Modern Algebra* (élimination theory, Sylvester matrix). Dover.
- Gröbner, W. (1939) — *Über ein neues Ideal-Theoretisches Rechtsverfahren*. Monatsh. Math.
- Basu, S., Pollack, R., Roy, M.-F. (2003) — *Algorithms in Real Algebraic Geometry*. Springer (resultants, discriminants).
- Sturmfels, B. (2002) — *Solving Systems of Polynomial Equations*. AMS (elimination via resultants).

### Géométrie tropicale
- Mikhalkin, G. (2000, 2005) — Tropical geometry, tropical varieties.
- Maclagan, D., Sturmfels, B. (2015) — *Introduction to Tropical Geometry*. AMS.
- Speyer, D., Sturmfels, B. (2009) — Tropical mathematics. arXiv:0901.2008.
- Baker, M., Payne, S. (2016) — Tropical geometry survey. Handbook of moduli.

### Analyse microlocale & Opérateurs pseudo-différentiels
- Hörmander, L. (1985) — *The Analysis of Linear Partial Differential Operators*. Springer (symbol calculus, microlocal analysis).
- Duistermaat, J. J., Guillemin, V. (1975) — *The Geometry of the Moment Map*. (wave front sets, microlocality).
- Gelfand, I. M., Shilov, G. E. (1964) — *Generalized Functions*. Vol 1. (distributions, singularities).
- Shubin, M. A. (2001) — *Pseudodifferential Operators and Spectral Theory*. Springer (opérateurs pseudo-diff).

### Géométrie différentielle & Géométrie de l'information
- Amari, S. (1985) — *Differential-Geometric Methods in Statistics*. Springer (Fisher metric, information geometry).
- Amari, S., Nagaoka, H. (2000) — *Methods of Information Geometry*. AMS (variétés statistiques, géométrie Riemannienne).
- Efron, B. (1975) — *Defining the Curvature of a Statistical Problem* (avec comment). AAS (Fisher curvature).
- Otto, F., Villani, C. (2000) — *Generalization of an Inequality by Talagrand and Application to Second-Order Markov Chains*. JFA (Wasserstein, transport optimal).

### Systèmes holonomes & D-modules
- Kashiwara, M. (1970) — *b-functions and holonomic systems*. Inventiones Math.
- Bjork, J.-E. (1979) — *Rings of Differential Operators*. North-Holland (D-modules, holonomicity).
- Bernstein, I. N., Sato, F. (1972) — *b-functions and holonomic systems*. Functional Anal. Appl.
- Grayson, D. R., Stillman, M. E. — *Macaulay2* (syzygies, free resolutions, D-modules computation).

### Théorie des modèles & Complexité logique (NIP, NTP₂)
- Shelah, S. (1990) — *Classification Theory: The Number of Non-Isomorphic Models*. North-Holland (NIP, stability rank).
- Simon, P. (2015) — *A Guide to NIP Theories*. Lecture Notes in Logic, CUP (No Independent Pairs).
- Chernikov, A., Simon, P. (2015) — *Definably connected NIP fields*. arXiv (NTP₂ complexity).
- Ben-Yaacov, I., Itaï, T., Usvyatsov, A. (2009) — *Continuous model theory*. Logic Colloq. Lecture Notes.

### Apprentissage & Dimension de Littlestone
- Littlestone, N. (1987) — *Learning Quickly when Irrelevant Attributes Abound*. FOCS (Littlestone dimension).
- Shalev-Shwartz, S., Shai Ben-David (2014) — *Understanding Machine Learning: Theory to Algorithms*. Cambridge UP (VC/Littlestone dimensions, PAC bounds).
- Alon, N., Ben-David, S., Cesa-Bianchi, N., Haussler, D. (1997) — *Scale-sensitive Dimensions, Uniform Convergence, and Learnability*. JCSS.

### Invariants catégoriques & Géométrie algébrique énumératives
- Leinster, T. (2008) — *The Euler Characteristic of a Category* (magnitude). Documenta Mathematica.
- Kontsevich, M., Soibelman, Y. (2000) — *Cohomological Hall algebra, exponential Hodge structures and motivic Donaldson-Thomas invariants* (DT invariants). arXiv:math/0006102.
- Joyce, D. (2007) — *Configurations in Abelian Categories and Moduli Problems for Bridging Categories*. arXiv (t-structures, categorical invariants).
- Bhatt, B., Scholze, P. (2013) — *The Prism* (prismathic cohomology). arXiv:1305.5908.

---

## Liens internes

- **Cadrage projet (vision)** : [DOC/00_vision.md](00_vision.md)
- **Phase 1 metrologie** (Oracle, SSG, hankel/spectral metrics) : [DOC/01_phase_metrologie.md](01_phase_metrologie.md)
- **Phase 1.5 spec** (signaux phase 1.5 utilisés en pratique pour Partie 2) : [DOC/01b_phase_calibration_signal.md](01b_phase_calibration_signal.md)
- **Phase 2 audit spectral** (SCH, structures) : [DOC/02_phase_audit_spectral.md](02_phase_audit_spectral.md)
- **Carnet — décisions cadrage** : [DOC/carnet_de_bord.md](carnet_de_bord.md) (entrées 2026-05-11 10:10, 10:15, 10:25, 10:35, refactor extract)

## Liens code (infrastructure)

- **Extraction matrices** (per-layer / streamed / windowed) : `CODE/phase1_metrologie/oracle/extract.py`
- **Tests extraction streaming** : `CODE/phase1_metrologie/tests/test_extract_streaming.py`
- **Signaux phase 1.5** : `CODE/phase1b_calibration_signal/signals/` (s_kl, s_spectral, s_grad)
- **Audit spectral phase 2** : `CODE/phase2_audit_spectral/` (svd_pipeline, stress_rank_map, transfer_law)
