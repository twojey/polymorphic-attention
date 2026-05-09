# 01 — Phase 1 : Métrologie (RCP)

**Reverse Capacity Profiling.** On mesure la capacité que l'attention pleine *consomme effectivement* sur des tâches dont on contrôle la difficulté structurelle. C'est la borne supérieure que l'ASP devra atteindre, et c'est la **baseline de difficulté mathématique indiscutable** sur laquelle s'appuieront toutes les phases suivantes.

## 1. Objectif

Produire une cartographie quantitative :

```
{ (ω, Δ, ℋ) → (rang_Hankel, entropie_spectrale, qualité) }
```

À la sortie de la phase, on doit pouvoir répondre : *« pour quelle classe de paramètres l'attention utilise-t-elle réellement plus qu'un opérateur de rang faible ? »*

## 2. Banc de stress : SSG (Structural Stress Generator)

Le **SSG** produit des variantes synthétiques contrôlées de MNIST (Structure-MNIST) où la difficulté est paramétrée selon trois axes orthogonaux :

- **ω — profondeur de récursion.** Nombre d'étapes de composition fonctionnelle requises avant la sortie. Exemple : appliquer une transformation récurrente ω fois sur l'image, demander la classe finale.
- **Δ — distance de dépendance.** Écart maximal (en tokens / patches) entre l'information pertinente et le point de décision.
- **ℋ — entropie de la tâche.** Mesure de l'imprévisibilité injectée : ratio signal/distracteurs, bruit, multiplicité des chemins de résolution. ℋ haute → tâche peu structurée, le modèle ne peut pas s'appuyer sur des régularités.

**Principe d'isolation.** Chaque axe est varié **séparément** (les autres restant à une valeur de référence basse) pour isoler son effet. Les balayages croisés (ω × Δ, etc.) viennent dans un second temps, et seulement après que les courbes monovariées ont été comprises.

Le choix de MNIST n'est pas anodin : la base est connue, peu coûteuse, et permet de balayer (ω, Δ, ℋ) sans contamination par les biais d'un dataset naturel.

### 2b. Limite reconnue — les axes (ω, Δ, ℋ) sont des proxies humains de la difficulté

Les trois axes du SSG sont des **paramètres humainement interprétables** de ce qu'on imagine être la difficulté structurelle. Mais la "vraie" dimension de difficulté pour l'attention pourrait être ailleurs :

- profondeur de composition fonctionnelle (pas réductible à ω seul),
- ambiguïté de référence ou polysémie (orthogonal à Δ),
- densité d'information conditionnelle (proche mais distinct de ℋ),
- distance dans un espace latent appris (non observable a priori),
- ou une dimension latente dont le concept n'a pas encore été formalisé.

En choisissant ω/Δ/ℋ, le protocole **pré-détermine quelles découvertes sont possibles**. C'est un fish-in-water problem : un poisson pose la question "qu'est-ce qui rend l'eau dure à nager ?" en termes de température et viscosité, en manquant peut-être la salinité ou les courants — dimensions qu'il n'a pas pensé à mesurer.

**Mitigations partielles :**

1. **Multi-Oracle / multi-domaine** (section 7b) — si les domaines exhibent des structures spectrales radicalement différentes, c'est un signe que la SCH a une dimension cachée non capturée par (ω, Δ, ℋ).
2. **PCA cross-régimes des r_eff** (livrable phase 2, DOC/02) — projeter les régimes du SSG dans l'espace des r_eff observés. Si la structure des clusters s'aligne sur les axes (ω, Δ, ℋ), les axes capturent bien la difficulté. Si elle traverse les axes, une dimension latente existe.
3. **Extension future (V4)** — explorer des paramétrisations alternatives du stress : complexité MDL, profondeur de composition logique, entropie conditionnelle, longueur de description algorithmique. Documenté ici comme piste d'extension, pas implémenté en V1.

## 3. Métriques

### 3.1 Rang de Hankel de la matrice d'attention

Pour A ∈ ℝ^{N×N} (post-softmax), construire la Hankelisation H(A) et calculer son rang numérique (nombre de valeurs singulières au-dessus d'un seuil τ relatif à σ_max). Interprétation : nombre de modes nécessaires pour reconstruire A par un système à mémoire finie.

### 3.2 Entropie spectrale

Soit (σ_k) les valeurs singulières de A. Définir p_k = σ_k² / Σ_j σ_j² et :

```
H_spectrale(A) = − Σ_k p_k log p_k
```

H ≈ 0 → A est essentiellement de rang 1. H ≈ log N → A est uniformément distribuée sur ses modes (cas le pire pour la compression).

### 3.3 Qualité

Loss / accuracy de référence sur la tâche, à Oracle fixé. Sert d'ancrage pour les phases ultérieures.

## 4. Protocole

1. Entraîner l'**Oracle** — un Transformer à attention dense, dimensionné largement au-dessus du besoin — sur Structure-MNIST avec balayages monovariés (ω puis Δ puis ℋ). C'est ce même Oracle qui sera consommé en phase 2 pour l'audit spectral.
2. À l'inférence sur le set de test, extraire les matrices d'attention de l'Oracle par couche et par tête.
3. Pour chaque (couche, tête, exemple) : mesurer (rang_Hankel, H_spectrale).
4. Agréger par valeur du paramètre varié. Produire **trois courbes** : rang_Hankel(ω), rang_Hankel(Δ), rang_Hankel(ℋ), idem pour H_spectrale.
5. Une fois les effets monovariés caractérisés, lancer les balayages croisés et produire les **cartes 2D** pour les paires d'axes pertinentes.
6. Identifier les régions de l'espace où ces métriques sont basses (compressibles) et hautes (non compressibles).

## 5. Sortie attendue

Un rapport contenant :

- Les courbes monovariées et les cartes 2D des métriques.
- L'estimation de la **demande de capacité moyenne** : E[rang_Hankel] / N et E[H_spectrale] / log N par régime.
- Une recommandation préliminaire : *ordre de grandeur de R_max pour la phase 3* (raffinée en phase 2 sur la base du r_eff mesuré).
- La référence formelle qui sera reprise comme baseline de difficulté en phases 2 et 5.

## 6. Critère go/no-go

**Go** : il existe une portion non négligeable de l'espace (ω, Δ, ℋ) où rang_Hankel ≪ N **ou** H_spectrale ≪ log N tout en conservant la qualité. Le protocole continue.

**No-go** : sur tout l'espace exploré, l'attention sature à rang ≈ N et entropie ≈ log N. L'hypothèse 1 (hétérogénéité) est rejetée. Le protocole s'arrête ici — l'ASP est sans marge.

## 7. Choix de la stack — *uniquement*

À la fin de cette phase, **une seule décision** est arrêtée : la stack d'implémentation (PyTorch / JAX / autre). Critère : capacité à instrumenter proprement les matrices d'attention intermédiaires sans pénalité de performance prohibitive, et disponibilité de primitives d'algèbre structurée (FFT, scan parallèle générique, SVD batchée).

**Aucune famille de Backbone n'est pré-sélectionnée à ce stade**. Le Backbone de l'ASPLayer (phase 3) sera *dérivé* du dictionnaire SCH produit par la phase 2 — pas importé d'une famille existante (Mamba, S4, S5, etc.). Pré-sélectionner reviendrait à importer les biais de design de ces architectures avant d'avoir mesuré ce que l'attention exige réellement (cf. principe Discovery > Reproduction, DOC/00 section 4b).

Conséquence pour la phase 1.5 : le signal `S_Surprise` (qui requérait un Backbone proxy) est **retiré** du quatuor de candidats. Phase 1.5 teste 3 signaux : S_KL, S_Grad, S_Spectral.

## 7b. Protocole multi-Oracle

Le protocole utilise **plusieurs Oracles indépendants**, un par domaine de données, pour découvrir si la SCH est universelle ou domain-spécifique. Chaque Oracle est entraîné *from scratch*, sans transfert ni initialisation depuis un modèle pré-existant.

### 7b.1 Domaines V1 (proof of concept)

Au moins deux domaines pour le proof of concept, choisis pour leur diversité structurelle attendue :

| Domaine | Hypothèse de signature structurelle | Données |
|---------|-------------------------------------|---------|
| **Structure-MNIST** | Contrôle (la structure est connue par construction du SSG) | Généré par le SSG — phase 1 cœur |
| **Code synthétique** | Hiérarchisation forte (binding, composition fonctionnelle), récursion explicite via parenthèses imbriquées | Générateur de code synthétique paramétré (Dyck-k étendu, mini-DSL) |
| **Vision (optionnel V1)** | Commutativité forte, équivariance par translation héritée des conv | MNIST/CIFAR adapté pour attention par patches |

Structure-MNIST reste le **cœur méthodologique** : c'est la seule où on connaît la "vraie" difficulté par construction (paramètres ω, Δ, ℋ du SSG). Les autres domaines servent à tester la généralité de la SCH.

### 7b.2 Domaines V2 (extension future)

À documenter mais pas à implémenter en V1 :
- Langue naturelle (sous-échantillon LM)
- Math symbolique (équations, démonstrations)
- Audio (séquence temporelle dense)
- Graphes (structure non-séquentielle)

### 7b.3 Indépendance stricte des Oracles

- **Pas d'initialisation partagée** entre Oracles. Chaque Oracle part de zéro.
- **Pas de pretraining** sur un autre domaine.
- **Architecture identique** entre Oracles (Transformer dense pur, mêmes hyperparamètres) — sauf adaptations strictement nécessaires au domaine (taille d'embedding pour le vocabulaire/patches).
- L'**isomorphisme architectural** est crucial : sans lui, les différences de signatures spectrales pourraient être attribuées à l'architecture plutôt qu'au domaine.

### 7b.4 Tri-partition par Oracle

La règle des trois sets disjoints (section 8.6) s'applique **par domaine** : chaque domaine a son propre découpage `train_oracle` / `audit_svd` / `init_phase3`.

## 8. Intégrité de l'Oracle — quality gates obligatoires

L'Oracle est l'**instrument de mesure** des phases 2 et 4. Sa validité conditionne tout le protocole en aval. Les contraintes suivantes sont non-négociables et pré-enregistrées dans `OPS/configs/phase1/oracle.yaml` :

### 8.1 Architecture pure
- **Attention dense pleine** : pas de GQA (Grouped Query Attention), pas de MQA, pas de Sliding Window Attention. Chaque tête a ses propres K et V, sur tout le contexte.
- **Pas d'optimisations modifiant les patterns d'attention** : pas de routing top-k, pas de sparsité injectée. Flash Attention est autorisé (mathématiquement équivalent au niveau du résultat et des gradients).
- **Pas d'astuces "production"** importées sans nécessité (RoPE acceptable, mais documenter ; positional bias additionnel comme ALiBi à éviter sauf justification).

### 8.2 Capacité dimensionnée pour la mesure
- **Sur-paramétrage modéré** : suffisant pour saturer toutes les tâches du SSG sans bottleneck ; pas tellement large que l'Oracle mémorise du bruit.
- **Context length ≥ max(Δ) du SSG** : si l'Oracle est aveugle aux dépendances longues, le r_eff mesuré en phase 2 sera faussement bas. Critère vérifié avant entraînement.

### 8.3 Convergence par plateau
- **Critère d'arrêt = plateau de la loss validation** (variation < seuil sur N époques), **pas un nombre fixe d'époques**.
- Vérification par régime du SSG : la loss validation doit avoir plateauté **sur chaque axe** (ω, Δ, ℋ), pas seulement en moyenne.
- Un Oracle sous-entraîné produit des patterns d'attention bruités qui rendent la phase 2 illisible.

### 8.4 Précision : entraînement vs mesure
- **Entraînement en BF16 mixed-precision** (ou FP16 si nécessaire) — performance standard SOTA, suffisant pour la qualité d'entraînement.
- **Cast explicite en FP64** des matrices A au moment de l'extraction pour la SVD de phase 2 — précision chirurgicale sur l'instrument de mesure, pas sur le calcul d'entraînement.

### 8.5 Zéro modification post-entraînement
- **Interdiction stricte** de modifier l'Oracle entre la fin de l'entraînement et l'extraction des matrices A : pas de pruning, pas de distillation, pas de quantization, pas de re-fine-tuning.
- Un Oracle modifié n'est plus l'instrument qu'on a entraîné — toute mesure faite dessus est invalide.

### 8.6 Trois sets disjoints (règle des sets)
Dès la phase 1, les données du SSG sont partitionnées en **trois sets strictement disjoints**, pré-enregistrés dans la config :

| Set | Usage | Phase consommatrice |
|-----|-------|---------------------|
| `train_oracle` | Entraînement de l'Oracle | 1, 1.5 |
| `audit_svd` | Extraction des matrices A pour la SVD et le calcul de r_eff | 2 |
| `init_phase3` | Extraction de structure pour Smart Init Matriochka (si utilisé) | 3 (ablation) |

Cette tri-partition est **constitutive** du protocole. Sans elle, on mesure la mémorisation de l'Oracle au lieu de sa généralisation, ou on initialise les bases U/V de phase 3 avec de l'information déjà vue. La proportion (par défaut : 70/20/10) est arrêtée en début de phase 1.

### 8.7 Pas d'agrégation pré-SVD
- L'extraction des matrices A se fait **par tête, par couche, par exemple** — sans moyenne ou réduction préalable.
- L'agrégation (par tête, par couche) est faite *après* la SVD, comme analyse de structure interne.
- Moyenner avant SVD masque la spécialisation des têtes et fausse r_eff agrégé.
