# 00c — Prédictions a priori des signatures mathématiques

**But.** Pré-enregistrer des **paris intuitifs** sur les résultats attendus de la batterie Partie 1 (DOC/00b) appliquée à un panel d'**Oracles denses** diversifiés par domaine d'entraînement. Document **falsifiable** : la confrontation de ces prédictions aux mesures réelles produit de la connaissance dans tous les cas.

> **Méthodologie** : ce document est **pré-enregistré avant exécution de la batterie**. Il NE doit PAS être modifié post-hoc en fonction des résultats. Toute modification doit être horodatée et justifiée. La valeur scientifique réside précisément dans la confrontation paris vs mesures.
>
> **Cadrage** : Partie 1 (science fondamentale, cf. DOC/00b). Le but du projet ASP est d'étudier les **Oracles denses** (attentions O(N²)) pour comprendre comment synthétiser leurs propriétés dans une couche sub-quadratique (Partie 2). La diversité du panel vient donc du **domaine d'entraînement**, pas de l'architecture.

Date d'enregistrement initial : **2026-05-11 ~11:00 UTC** (8 Oracles dense+sub-quad).
Date de révision **(pivot méthodologique)** : **2026-05-11 ~11:30 UTC** (6 Oracles denses diversifiés). Justification de la révision : recentrage explicite sur l'objectif "comprendre les Oracles denses" — les architectures sub-quadratiques (Mamba, Performer, etc.) ne sont plus dans le scope de la batterie.

---

## 1. Oracles considérés (6 Oracles denses, diversité par domaine)

| Code | Oracle | Domaine | Taille | Attention |
|---|---|---|---|---|
| **OR** | Notre Oracle SMNIST V1 | contrôlé (structure (ω,Δ,ℋ) connue) | ~5M | dense softmax causale |
| **LL** | Llama 3.2 1B | texte général | 1B | dense softmax causale |
| **SC** | StarCoder 2 3B | code | 3B | dense softmax causale |
| **DV** | DINOv2 ViT-B/14 | vision pure (sans texte) | 86M | dense softmax bidirectionnel (ViT) |
| **CL** | CLIP ViT-L/14 | multimodal vision↔texte | 304M | dense softmax bidirectionnel (ViT) |
| **ES** | ESM-2 (35M-150M) | biologique (séquences protéines) | 35-150M | dense softmax bidirectionnel |

**Toutes denses, toutes O(N²)** par construction. La diversité est dans **le domaine d'entraînement** (SMNIST jouet / langage / code / vision pure / multimodal / biologique).

## 2. Légende cellules

- ✅ oui (forte conviction théorique)
- 🟢 oui (probable, indices empiriques)
- 🟡 partiel ou conditionnel
- 🔴 non (probable)
- ❌ non (forte conviction théorique)
- ❓ indécis / pas d'intuition forte → **valeur scientifique de la mesure maximale**

---

## 3. Tableaux de prédictions

### Tableau 1 — Algébriques (G) — toutes les attentions denses sont quasi-identiques ici

| Propriété | OR | LL | SC | DV | CL | ES |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **G5** Linéarité (en x) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| G2 Symétrie A=A^T | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| G3 Idempotence A²≈A | 🔴 | 🔴 | 🔴 | 🔴 | 🔴 | 🔴 |
| Stochasticité ligne (sum=1) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Causalité (triangulaire inf.) | ✅ | ✅ | ✅ | ❌ (ViT bidir) | ❌ (ViT bidir) | ❌ (ESM bidir) |

> **Pattern** : ces propriétés ne discriminent PAS entre Oracles denses (toutes pareilles). Sauf la **causalité** qui sépare les autorégressifs (OR, LL, SC) des bidirectionnels (DV, CL, ES). Pour Partie 1, ces tests servent surtout à **valider la batterie** (tous les Oracles doivent passer "stochasticité ligne", sinon bug).

### Tableau 2 — Spectrales (A) — où la diversité des domaines apparaît

| Propriété | OR | LL | SC | DV | CL | ES |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **A1** r_eff faible par fenêtre K | 🟢 | 🟢 | 🟡 (code = précis) | ✅ (vision spat.) | 🟢 | 🟢 |
| A3 Conditionnement κ borné | 🟡 | 🟡 | 🟡 | 🟢 | 🟡 | 🟡 |
| A4 Entropie spectrale faible | 🟢 | 🟢 | 🟡 | ✅ | 🟢 | 🟢 |
| A5 Décroissance σ_i exponentielle | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | 🟡 |
| A6 Participation ratio bornée | 🟢 | 🟢 | 🟡 | ✅ | 🟢 | 🟢 |

> **Pattern attendu** : DINOv2 (vision pure spatiale) doit avoir le **rang effectif le plus faible** (pixels voisins corrélés → low-rank). StarCoder (code) doit avoir le **rang effectif le plus haut** parmi les denses (logique précise = chaque token compte → moins de redondance). Si pas observé → revisiter l'hypothèse "vision = low-rank".

### Tableau 3 — Structurelles (B) — la signature compression

| Propriété | OR | LL | SC | DV | CL | ES |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **B1** Toeplitz-ness (data-indep.) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| B2 Rang Hankel local faible | 🟡 | 🟡 | 🟡 | 🟢 | 🟡 | 🟡 |
| B4 Sparsité effective | 🟡 | 🟡 | 🟢 (matching local) | ❌ | ❌ | ❌ |
| B5 Block-diagonality | 🔴 | 🔴 | 🟡 (fonctions) | 🟢 (patches voisins) | 🟡 | 🔴 |
| B6 Bandedness (largeur faible) | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟢 (séq. courtes) |
| Low-rank global | ❌ | ❌ | ❌ | 🟡 | 🔴 | 🔴 |

> **Pattern attendu** :
> - **DINOv2** : block-diagonality forte attendue (patches voisins = bloc d'attention concentrée)
> - **StarCoder** : sparsité effective + block (fonctions = blocs séparés)
> - **ESM-2** : bandedness pour séquences courtes (interactions locales protéines)
> - Pas de Toeplitz pur attendu (data-dep. exclut ça pour toute attention dense)

### Tableau 4 — Rang de déplacement (O) + Ho-Kalman (P)

| Propriété | OR | LL | SC | DV | CL | ES |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| O1 Toeplitz-like (rang dépl. ≤ 2) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| O2 Cauchy-like (rang dépl. ≤ 1) | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ |
| **P1** Rang Hankel global fini | ❌ | ❌ | 🟡 | 🟡 | ❓ | ❓ |
| P3 Décroissance HSV exponentielle | 🟡 | 🟡 | 🟡 | 🟢 | 🟡 | 🟡 |
| **P4** Ordre minimal n borné par domaine | ❓ | ❓ | ❓ | 🟡 | ❓ | ❓ |
| Q2 Rang local blocs admissibles k faible | ❓ | 🟡 | 🟡 | 🟢 | 🟡 | 🟡 |
| Q4 Nestedness (H²/HSS) | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ |

> **Pattern attendu** : DINOv2 candidat le plus fort pour H-matrices (vision spatiale → blocs admissibles). Cauchy-like (O2) reste un point d'interrogation universel — **opportunité de découverte si trouvé**. Nestedness (Q4) = vraie inconnue partout — **information maximale si mesuré**.

### Tableau 5 — Mercer / RKHS (R) — discriminant null pour Oracles denses

| Propriété | OR | LL | SC | DV | CL | ES |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **R1** Positive-définitude Mercer | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| R3 Stationnarité K(x-y) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| R4 Bochner (FT μ ≥ 0) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

> **Pattern** : aucune attention softmax dense n'est Mercer-PD (softmax fait des produits qui ne sont pas garantis ≥ 0 après normalisation). Ces tests sont des **invalidations attendues uniformes** — utiles comme contrôle (toute mesure ≠ ❌ = bug ou découverte majeure).

### Tableau 6 — Cross-layer (H) / Cross-head (I) / Dynamiques (F) — incertitudes maximales

| Propriété | OR | LL | SC | DV | CL | ES |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| H1 Résidu inter-layers décroissant | ❓ | 🟢 | ❓ | ❓ | ❓ | ❓ |
| **H3** Évolution rang croissante avec ℓ | 🟢 | 🟢 | ❓ | ❓ | ❓ | 🟡 |
| H4 Convergence vers stationnaire | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ |
| **I1** Diversité inter-heads forte | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 |
| I2 Spécialisation heads | 🟢 | 🟢 | 🟢 (syntaxe vs sémantique) | 🟢 (positions vs contenu) | 🟢 | 🟢 |
| I3 Cluster heads visibles | 🟡 | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 |
| **F1** Lipschitzness vs entrée | 🟡 | 🟡 | 🟡 | 🟢 | 🟡 | 🟡 |
| F2 Stabilité temporelle (A_t ≈ A_{t+1}) | 🟡 | 🟡 | 🔴 | n/a (pas séq.) | n/a | 🟡 |

> **Pattern attendu** : la spécialisation des heads (I2) doit être **forte partout** mais sur des dimensions différentes (syntaxe/sémantique pour LLM, position/contenu pour ViT, structure secondaire pour ESM). Les évolutions cross-layer restent **majoritairement inconnues** — gain d'info maximal.

### Tableau 7 — Topologiques (K) / Fréquentielles (L) / Markov (J) / Comparatives (N)

| Propriété | OR | LL | SC | DV | CL | ES |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| J1 Markov-ness (A^k convergence) | 🟡 | 🟡 | 🟡 | 🟢 | 🟡 | 🟡 |
| J2 Distribution stationnaire informative | ❓ | ❓ | ❓ | 🟢 | ❓ | ❓ |
| K1 Spectre Laplacien graphe interp. | ❓ | ❓ | ❓ | 🟢 | ❓ | ❓ |
| K2 Persistent homology (TDA) signature | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ |
| K4 Communautés / clustering tokens | 🟢 (digit/op/noise) | 🟢 (parts of speech) | 🟢 (syntaxe) | 🟢 (objets) | 🟢 | 🟢 (résidus) |
| L1 FFT 2D structure | ❓ | ❓ | ❓ | 🟢 (vision spatiale) | ❓ | ❓ |
| L3 Quasi-périodicité | ❓ | 🟡 | 🟡 (boucles) | 🟢 (textures) | ❓ | 🟢 (motifs) |
| **N1** F-divergence Oracle/student | n/a | n/a | n/a | n/a | n/a | n/a |

> **Patterns attendus** :
> - **Communautés (K4)** : tous les Oracles doivent montrer des clusters de tokens correspondant à leur structure linguistique/spatiale.
> - **Wavelets/FFT (L)** : DINOv2 candidat fort pour structure périodique (textures, motifs visuels).
> - **N (comparatives)** : non applicable tant qu'on n'a pas de student — vivra plus tard quand l'ASP existera.

### Tableau 8 — Conditionnelles à l'entrée (M) — la dimension naturelle de notre projet

| Propriété | OR | LL | SC | DV | CL | ES |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **M1** Sensitivity par type de token | ✅ (digit/op/noise) | 🟢 (mots-clés) | 🟢 (keywords/operators) | 🟢 (CLS vs patches) | 🟢 | 🟢 (acides aminés) |
| **M2** Variation A vs paramètres de stress | ✅ (mesuré phase 1.5) | ❓ (à mesurer) | ❓ | ❓ | ❓ | ❓ |

> **Pattern** : c'est la dimension **la plus directement utile pour ASP** — comprendre comment l'attention varie selon le type d'input permet de prédire le rang requis. **À tester systématiquement sur tous les Oracles**.

---

## 4. Synthèse — Signatures attendues par domaine

| Oracle | Signature mathématique attendue | Hypothèses de classes |
|---|---|---|
| **OR** SMNIST V1 | r_eff faible structuré, M1/M2 forts (par construction du SSG), pas d'invariant structurel pur | Référence empirique connue |
| **LL** Llama (texte) | r_eff modéré, I2 forte (heads spécialisées syntaxe/sémantique), évolution rang cross-layer croissante, sensibilité aux mots-clés | "Mixed" — pas de classe pure |
| **SC** StarCoder (code) | r_eff plus haut (logique précise), sparsité + block (fonctions), F2 faible (transitions abruptes) | Sparse + précis |
| **DV** DINOv2 (vision pure) | r_eff faible, block-diag fort (patches voisins), Lipschitz forte, candidat **H-matrix/HSS** | Hiérarchique-spatial |
| **CL** CLIP (multimodal) | Hybride DV + texte, attention cross-modal asymétrique | Hybride |
| **ES** ESM-2 (protéines) | Bandedness forte (interactions locales aa), motifs périodiques (structures secondaires), I2 forte (acides hydrophobes/polaires) | Bandedness + motifs |

## 5. Tests les plus discriminants pour ce panel

| Test | Discrimination attendue |
|---|---|
| **A1** r_eff par fenêtre | DV (très bas) ↔ SC (modéré) ↔ LL/CL/ES (intermédiaire) |
| **B5** Block-diagonality | DV/SC (forte) ↔ LL/CL/ES (faible) |
| **B6** Bandedness | ES (forte) ↔ autres (modérée) |
| **F1** Lipschitzness | DV (forte) ↔ SC (faible) ↔ autres (modérée) |
| **K4** Communautés/clustering | tous forts mais sur dimensions différentes |
| **L1/L3** Périodicité fréquentielle | DV/ES (fort) ↔ LL/SC (faible) |
| **M1/M2** Conditionnelles aux types d'entrée | tous forts — **dimension la plus utile pour ASP** |

## 6. Tests à faible discrimination (uniformes pour tous denses)

- **Tous G** (algébriques) — denses sont toutes pareilles ici
- **Tous R** (Mercer) — denses échouent toutes uniformément
- **Stochasticité ligne** — toutes la respectent (sauf bug)

→ Ces tests servent de **contrôle de validation de la batterie** (résultat attendu uniforme = check de bon fonctionnement).

## 7. Incertitudes principales (zones de découverte potentielle maximale)

1. **Q4 Nestedness** — aucune intuition pour aucun Oracle dense. Vraie inconnue.
2. **H3, H4** Évolution cross-layer — peu mesuré au-delà de Llama-like.
3. **K (TDA)** — TDA sur attentions est peu exploré en littérature.
4. **L (wavelets)** — analyse multi-échelle de A pas dans la littérature standard.
5. **O2 Cauchy-like** — si trouvé sur un Oracle dense, c'est une découverte de classe non explorée.
6. **P (Ho-Kalman) sur DT** — théoriquement DT est non-LTI, mais HSV peuvent être bornés empiriquement sur distribution réaliste.

## 8. Falsifiabilité de mes paris

| Pari pré-enregistré | Si invalidé → conclusion |
|---|---|
| DV a r_eff le plus faible des 6 | Hypothèse "vision = low-rank empirique" remise en cause |
| SC a r_eff le plus haut des 6 | Hypothèse "code = précis donc plein-rang" remise en cause |
| Tous échouent R1 Mercer | Si un passe → bug ou découverte majeure |
| ES a bandedness forte | Si non → interactions long-distance dominent dans ESM-2 |
| K4 communautés visibles partout | Si non → pas de structure latente cohérente dans ces Oracles |

---

## 9. Confrontation aux résultats (à compléter post-batterie)

| Pari | Mesure réelle | Verdict | Surprise ? |
|---|---|---|---|
| (à remplir) | | | |

Score global discriminant : **/35 paris testables** (cellules 🟢/🔴/✅/❌ ; on exclut 🟡/❓ qui sont par nature non-falsifiés strictement).

---

## Liens

- **Catalogue Partie 1** : [00b_classification_proprietes.md](00b_classification_proprietes.md)
- **Sélection Oracles + protocole d'entrées** : [00d_oracles_battery.md](00d_oracles_battery.md)
- **Cadrage projet** : [00_vision.md](00_vision.md)
- **Roadmap batterie** : [../ROADMAP.md](../ROADMAP.md) section "Stage 1.5+"
- **Carnet** : [carnet_de_bord.md](carnet_de_bord.md) (entrée 2026-05-11 11:30)
