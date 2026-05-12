# 00 — Fondations : Thèse ASP + Vocabulaire

Document fusionné contenant la thèse scientifique du projet ASP et les vocabulaires architectural/mathématique associés.

---

## § 1. Thèse ASP

### 1.1 Scope — ce que le projet étudie, et ce qu'il ne touche pas

Le projet étudie l'**attention** comme sous-module du Transformer. Toutes les analyses (SVD, r_eff, SCH, dictionnaire structurel) portent sur la matrice d'attention A et l'opérateur qui la calcule.

Le **MLP/FFN** (qui contient ~2/3 des paramètres d'un Transformer moderne), les **résiduels**, les **LayerNorms**, les **embeddings** et les **têtes de sortie** sont conservés tels quels. Toute amélioration ou dégradation venant de ces composants est **hors-scope**.

Conséquences pratiques :
- L'ASPLayer est un *drop-in replacement* d'une couche d'attention dense, pas d'un Transformer entier.
- Les comparateurs phase 5 sont évalués à *architecture FFN/résiduel/normalisation strictement identique* à l'ASP. Si un Transformer avec MLP plus large obtient mieux, ce n'est pas un échec d'ASP — c'est hors périmètre.
- La prémisse implicite — que la structure intéressante vit dans A — est une **hypothèse de cadrage non interrogée**. Si le travail computationnel important se fait majoritairement dans le MLP, optimiser l'attention seule a un plafond intrinsèque que l'ASP ne pourra pas dépasser.

Ce cadrage est revendiqué, pas masqué. Il limite l'ambition mais rend le projet exécutable.

### 1.2 Le problème

L'attention pleine est en O(N²) en temps et mémoire. Les alternatives linéaires (SSM, attention linéaire, RNN modernes) règlent le coût mais perdent la qualité sur les tâches qui demandent de la **récursion** ou des **dépendances longues**. Le débat actuel oppose deux positions :

- **Attention pleine** — qualité maximale, coût rédhibitoire à grand N.
- **SSM / linéaire** — coût plat, plafond de qualité observable empiriquement sur tâches structurées.

Le protocole ASP rejette l'opposition : il fait l'hypothèse que la qualité de l'attention ne se distribue pas uniformément dans la séquence. Elle vit dans des **moments critiques** rares ; le reste du temps, une structure linéaire suffit.

### 1.3 La thèse

> La quantité de calcul utile par token est variable et mesurable. Un modèle qui alloue dynamiquement son rang d'interaction peut atteindre la qualité d'une attention pleine en payant le coût d'un SSM en moyenne.

Trois sous-hypothèses, falsifiables individuellement :

1. **Hétérogénéité mesurable** — les tâches admettent un profil (rang Hankel, entropie spectrale) variable selon paramètres structurels (ω récursion, Δ distance, ℋ entropie). *Testée en phase 1 via le SSG.*
2. **SCH — correspondance structurelle** — il existe une correspondance reproductible (ω, Δ, ℋ) → r_eff sur le rang structurel effectif (mesuré par SVD de la matrice d'attention des Oracles), résumée par une loi de transfert ajustable. *Question ouverte : la loi est-elle universelle entre domaines (vision, code, langue) ou domain-spécifique ? Testée en phase 2 sur plusieurs Oracles ; statut épistémique d'hypothèse, pas de théorème.*
3. **Allocation contrôlable** — un mini-réseau (le Spectromètre) peut, sous pression d'une loss de budget λ_budget, prédire et allouer le rang token par token de manière causalement reliée à la difficulté locale. *Construite en phases 3 et 4, validée par les cinq tests de phase 5 (Identifiabilité, Élasticité, SE/HT, Pareto, OOD).*

### 1.4 Le livrable

Une frontière de Pareto **strictement dominée** sur l'axe (qualité, complexité), validée en phase 5. Les conditions précises :

- À iso-qualité du Transformer plein : FLOPs/token et mémoire restent **plats** quand N croît.
- À iso-complexité d'un SSM pur : gain mesurable de qualité sur les tâches récursives et longue distance.

Si aucun de ces deux régimes n'est atteint, le protocole a échoué.

#### L'Oracle est une *borne supérieure*, pas une cible à reproduire

Précision méthodologique critique. L'Oracle est un Transformer dense entraîné — **il n'est pas le comportement optimal**, c'est le comportement *appris sous contraintes de softmax non-régularisée*. Si l'Oracle alloue r_eff = 30 sur un régime, c'est peut-être parce que :

- l'attention dense **sur-utilise sa capacité par défaut** (toujours allouer le maximum disponible est le minimum local naturel d'une softmax),
- les têtes redondantes ne peuvent pas se "désactiver" individuellement même si elles n'apportent rien,
- l'optimisation Adam/AdamW pousse vers des solutions denses sans pression budgétaire explicite.

Une architecture vraiment optimale pourrait faire la même tâche à r_eff = 5. L'objectif d'ASP n'est pas de **matcher la consommation de rang de l'Oracle** ; c'est d'**atteindre la qualité de l'Oracle à strictement moins de rang**.

Conséquences concrètes :
- La loi de transfert phase 2 décrit ce que l'Oracle utilise, pas ce qui est nécessaire.
- La distillation phase 4a est un *prior souple*, pas un plafond.
- Phase 5 inclut un test "R_max réduit" (DOC/05 section 6c) où l'ASP est entraîné avec `R_max = médiane(r_eff_oracle) / 2`. Si la qualité tient, l'ASP a *dépassé* l'Oracle structurellement.

### 1.5 Pourquoi ce n'est pas une optimisation

L'argument n'est pas « on peut compresser un Transformer en gardant 99 % de la qualité ». L'argument est que l'attention **dilue** sa capacité sur des tokens qui n'en avaient pas besoin, et qu'une mesure préalable révèle où elle aurait pu être économisée. La compression n'est pas un effet recherché ; c'est la conséquence d'un meilleur modèle de la demande.

### 1.6 Principe fondateur — Discovery, pas Reproduction

Le protocole ne vise pas à *reproduire* une architecture existante (Mamba, RetNet, Hyena, MoD, etc.) en y ajoutant une touche de polymorphisme. Il vise à **découvrir** la structure que l'attention exhibe sur des données réelles, et à dériver un opérateur qui *implémente* cette structure mesurée.

Conséquences strictes :

1. **Aucune architecture n'est pré-sélectionnée**. Le Backbone de l'ASPLayer est construit *à partir* du dictionnaire SCH issu de la phase 2 — pas importé d'un projet académique antérieur. Si la phase 2 révèle que les régimes faciles sont Toeplitz de rang 2, le Backbone est un opérateur Toeplitz paramétré. Pas un Mamba2 entraîné à se comporter comme.
2. **Aucun signal n'est emprunté à une architecture pré-existante**. Le Spectromètre se calibre sur des signaux mesurés directement (KL local, gradient, rang spectral), pas sur des résidus calculés via un proxy importé.
3. **Aucun benchmark n'est sacralisé sans audit**. LRA, induction, copy, etc. sont des comparateurs, pas des cibles. La validation se fait d'abord sur le SSG (où la structure est connue par construction), ensuite sur des domaines extérieurs.
4. **L'Oracle n'est pas unique**. Différents domaines (vision, code, langue, math) exhibent des signatures structurelles différentes — la vision a tendance à la commutativité (héritée des conv), le code à la hiérarchisation (binding lourd), la langue à un mélange. Le protocole utilise **plusieurs Oracles**, un par domaine, pour découvrir si la SCH est universelle ou domain-spécifique.

C'est cette discipline du *fresh start* qui distingue le projet d'une optimisation marginale parmi d'autres. Le succès n'est pas "ASP bat MoD de 5 %" — c'est "voici la structure mathématique que l'attention exhibe sur ces domaines, et voici l'opérateur dérivé qui l'implémente avec un coût plat".

### 1.7 Carte du protocole

```
Phase 1   (RCP / SSG + multi-Oracle) →  Oracles entraînés (1 par domaine), baselines de difficulté
       ↓
Phase 1.5 (Calibration Signal)       →  IDENTIFICATION GATE : 3 signaux candidats (S_KL, S_Grad, S_Spectral), ≥ 1 validé ou arrêt
       ↓
Phase 2   (Audit Spectral)           →  SCH corroborée par domaine : r_eff, Stress-Rank Map, loi de transfert + comparaison cross-domain
       ↓
Phase 3   (ASPLayer)                 →  Backbone *dérivé* du dictionnaire SCH (pas importé) + correction Matriochka, Soft-Mask, Loss Consistency
       ↓
Phase 4   (Spectromètre)             →  Politique d'allocation R_target apprise sous λ_budget : 4a (warm-up + distillation) → 4b (autonome)
       ↓
Phase 5   (Validation)               →  5 tests : Identifiabilité, Élasticité, SE/HT, Pareto (comparateurs domain-aware), OOD croisé
```

Chaque flèche est conditionnelle au passage du critère go/no-go de la phase amont (cf. [`FALSIFIABILITE.md`](FALSIFIABILITE.md)).

---

## § 2. Vocabulaire architectural

Définitions opérationnelles pour les **modules architecturaux** du codebase. Vocabulaire utilisé dans le code, les rapports et les commits.

Distinct de § 3 (vocabulaire mathématique) qui couvre les termes théoriques.

### Property

**Définition** — un module qui calcule UNE mesure mathématique sur une (ou plusieurs) matrice(s) d'attention.

Une Property est :
- **Identifiée** : `name` (ex `A1_r_eff_theta099`), `family` (ex `A`)
- **Datée en coût** : `cost_class` ∈ {1, 2, 3, 4, 5} (1 = rapide < 1s/régime, 5 = lent > 1min)
- **Typée en pré-requis** : `requires_fp64`, `requires_symmetric`, `scope` (per_regime | cross_regime)
- **Single-responsibility** : retourne `dict[str, float | str]`, NE logge PAS à MLflow elle-même

Une Property = ~30-80 lignes de Python, un fichier dédié.

Le catalogue [`CATALOGUE.md`](CATALOGUE.md) spécifie 131 Properties (A1-W6). Le code projet vise à en implémenter ~60 (priorité haute + médium) en Sprint A.

### Family

**Définition** — regroupement sémantique de Properties qui partagent un cadre théorique (rang de déplacement, Ho-Kalman, équivariances, etc.).

Les Families sont les sections de [`CATALOGUE.md`](CATALOGUE.md). Couvertes par les sous-packages `catalog/properties/family_*/`.

### Projector

**Définition** — primitive mathématique réutilisable qui implémente UNE structure (Toeplitz, Hankel, Cauchy, Vandermonde, Butterfly, Monarch, etc.).

Un Projector expose deux opérations :
- `project(A: Tensor) → Tensor` : projection orthogonale (Frobenius) de A sur l'espace de la classe (utilisé par Properties pour calculer ε_C)
- `operator(params) → callable` : opérateur paramétré apprenable (utilisé par Backbone phase 3 le cas échéant)

Un Projector vit dans `catalog/projectors/<nom>.py` et est consommé par les Properties de la family correspondante.

### PropertyContext

**Définition** — contexte d'exécution partagé entre Properties d'un même run, jouant le rôle de **cache lazy** pour les pre-computations coûteuses.

Quand `B1_toeplitz_distance` et `B5_block_diagonal` veulent toutes deux la même SVD batchée, la première remplit le cache via `ctx.get_or_compute("svd", ...)` et la seconde réutilise. Évite la N²+ duplication entre Properties.

Le PropertyContext porte aussi :
- `device`, `dtype` : politique machine (résolue par `infra/machine.py`)
- `regime` : dict des paramètres de stress (ω, Δ, ℋ) en cours pour cette computation per_regime
- `attentions` : référence à la collection d'attentions (per_regime ou cross_regime selon scope)

### Battery

**Définition** — composition de Properties selon un **niveau** (minimal | principal | extended | full | research). Orchestre l'exécution sur un Oracle, l'agrégation distributionnelle (V3.5 RegimeStats), et le logging MLflow.

Niveaux :
- `level_minimal` : ~5 Properties, smoke check (~1 min wall-clock)
- `level_principal` : ~30 Properties, priorité haute (~10 min)
- `level_extended` : ~80 Properties, priorité med + haute (~1h)
- `level_full` : 131 Properties [`CATALOGUE.md`](CATALOGUE.md) (~plusieurs heures)
- `level_research` : familles V (analytique frontière) + W (model theory)

Une Battery = un fichier `catalog/batteries/level_<nom>.py` qui énumère ses Properties.

### Oracle (Adapter)

**Définition** — Adapter qui fournit des matrices d'attention dense à la Battery, pour un **domaine** donné (SMNIST, LL, vision, code…).

Un Oracle expose :
- `extract_regime(regime: RegimeSpec, n_examples: int) → AttentionDump` : récupère un dump pour un point du sweep de stress
- `oracle_id`, `domain`, `n_layers`, `n_heads`, `vocab_size` : métadonnées

Le seam Oracle permet la **comparaison cross-domain** (SMNIST × LL × ...) qui est le cœur de la Partie 1 (livrable scientifique "Mathematical Signatures of Attention"). Implémentations : `SMNISTOracle` (wrappe phase 1 existant), `LanguageOracle` (à coder).

### AttentionDump

**Définition** — structure de données fixée du contrat Oracle ↔ Battery.

```
AttentionDump :
  attn:       list[L tensors (B, H, N, N) FP64]   # par couche
  omegas:     tensor (B,) int                      # stress récursion
  deltas:     tensor (B,) int                      # stress binding distant
  entropies:  tensor (B,) float                    # stress ℋ
  tokens:     tensor (B, N) long                   # input tokens (pour replay)
  query_pos:  tensor (B,) long                     # position du QUERY token
  metadata:   dict                                 # oracle_id, seed, etc.
```

### MachineProfile

**Définition** — Module qui répond à "comment optimiser sur cette machine ?" en un point d'autorité unique.

Auto-détecte :
- `device` : cuda / cpu
- `gpu_arch` : Blackwell sm_120 / Hopper / Ampere / ... (impacte choix FP64 vs FP32)
- `precision` : fp64 / fp32 (FP64 = 1/64 du FP32 sur consumer Blackwell, donc CPU FP64 ou GPU FP32)
- `batch_cap` : selon VRAM / RAM
- `n_threads_blas` : 1 forcé si fork multiprocessing actif, sinon all-cores

Consommé par tout code qui faisait `if cuda.is_available()` ou choisissait device/precision localement. Source unique de vérité.

---

## § 3. Vocabulaire mathématique

### Algèbre linéaire structurée

**Matrice de Hankel** — matrice H telle que H_{ij} = h_{i+j}. Constante sur les anti-diagonales. Le rang d'une matrice de Hankel est lié à l'ordre minimal d'un système linéaire qui la génère.

**Rang de Hankel d'un signal** — rang de la matrice de Hankel construite à partir du signal. Estime le nombre de modes internes minimum pour reconstruire ce signal par récurrence linéaire.

**Matrice de Toeplitz** — T_{ij} = t_{i−j}. Constante sur les diagonales. Encode une convolution / une invariance par translation. Multiplication matrice-vecteur en O(N log N) via FFT.

**Matrice de Cauchy** — C_{ij} = 1/(x_i − y_j) pour deux ensembles de points distincts. Famille des matrices à *structure de déplacement*.

**Matrice de Vandermonde** — V_{ij} = x_i^j. Encode des évaluations polynomiales.

**Rang de déplacement** — pour une matrice M et deux opérateurs de décalage Z₁, Z₂, le rang de Z₁M − MZ₂. Hankel, Toeplitz, Cauchy, Vandermonde sont à rang de déplacement faible (≤ 2). Notion utilisée pour caractériser la *classe* d'opérateur structuré le mieux adapté.

**Matrice low-rank** — M = U V^T avec U, V ∈ ℝ^{N×r}, r ≪ N. Multiplication M·x en O(N r).

**SVD (Décomposition en Valeurs Singulières)** — A = U Σ Vᵀ avec Σ = diag(σ₁, …, σ_N), σ_k ≥ 0 décroissantes. Outil central de la phase 2 pour extraire le rang effectif de la matrice d'attention.

**Rang effectif r_eff(θ)** — plus petit k tel que (Σ_{i=1..k} σ_i) / (Σ_{j=1..N} σ_j) ≥ θ. Définit la dimension structurelle d'une matrice à un seuil d'énergie θ donné. Valeurs canoniques : θ = 0,95 (par défaut), θ = 0,99 (strict).

**Entropie spectrale** — pour (σ_k), p_k = σ_k² / Σ_j σ_j², H = − Σ_k p_k log p_k. Mesure la dispersion de l'énergie sur les modes.

**Théorème de Ho-Kalman** — un système LTI a une réalisation d'état de dimension finie ssi le rang de sa matrice de Hankel des paramètres de Markov est fini. Ce rang = ordre minimal du système. Fournit les bornes inférieures de complexité pour récursification.

**HSV (Hankel Singular Values)** — valeurs singulières de la matrice de Hankel d'un système LTI. Invariants globaux indépendants de la base de réalisation. Leur décroissance contrôle la compressibilité.

**H-matrix / H²-matrix / HSS** (Hackbusch) — formats de matrices hiérarchiques avec rang faible par blocs. H-matrix : O(Nk log N). H²-matrix / HSS : O(Nk) grâce à la **nestedness** (emboîtement des bases parent-fils). Voir [`CATALOGUE.md`](CATALOGUE.md).

**Nestedness** — propriété d'une matrice hiérarchique où la base d'un cluster parent peut être exprimée à partir des bases de ses enfants via une matrice de transfert de petite taille. Clé du O(N) strict pour H²/HSS.

**Théorème de Mercer** — un noyau continu, symétrique, défini positif K(x,y) admet une décomposition K(x,y) = Σ λ_i φ_i(x) φ_i(y) en série convergente. Si tronquable à D ≪ N termes, opérateur en O(ND).

**Théorème de Bochner** — un noyau stationnaire K(x-y) est défini positif ssi sa transformée de Fourier est une mesure positive. Justifie l'approximation par Random Fourier Features.

**Random Fourier Features (RFF)** — Rahimi-Recht 2007. Pour un noyau stationnaire K(x-y), échantillonner ω_j selon la densité spectrale μ et construire φ_j(x) = e^{iω_j^T x}. Donne K̃(x,y) = (1/D) Σ φ_j(x) φ_j(y)* avec erreur sup ∼ O(1/√D). Permet kernel approx en O(ND).

**Tensor Train (TT)** (Oseledets 2011) — décomposition d'un tenseur d'ordre élevé en chaîne de tenseurs d'ordre 3 avec rangs TT bornés. Stockage en O(d N r²), produit MV en O(r² N log N).

**Butterfly / Monarch** — matrices sparse-structurées admettant un produit MV en O(N log N) ou O(N√N). Fondement des transformées rapides (FFT, Hadamard) et des architectures Dao et al. 2022.

### Architectures séquentielles

**SSM (State Space Model)** — modèle à récurrence linéaire h_t = A h_{t-1} + B x_t, y_t = C h_t + D x_t. Coût O(N) en inférence, capacité bornée par la dimension d'état.

**S4, S5, Mamba, Mamba2** — variantes de SSM avec parameterizations spécifiques (HiPPO, sélectivité, scan parallèle). Choix précis différé.

**Attention pleine** — softmax(QKᵀ/√d) V, coût O(N²).

**Attention linéaire** — réécriture par feature map φ pour passer en O(N), au prix d'une expressivité réduite.

**Hyena, RWKV** — autres familles d'opérateurs séquentiels sub-quadratiques, comparateurs naturels en phase 5.

**MoD (Mixture of Depths)** — modèle d'allocation dynamique de calcul par token, Raposo et al. (DeepMind 2024). Routage **binaire** : chaque token passe ou saute la couche. Baseline principal de phase 5 pour évaluer la valeur ajoutée de l'allocation **spectrale** (continue) de l'ASP par rapport à l'allocation binaire de MoD.

**Oracle** — Transformer à attention dense, sur-dimensionné par rapport au besoin, entraîné en phase 1 *from scratch* sur un domaine donné. Sa matrice d'attention A est l'observable consommé en phase 2 (SVD, r_eff). Le protocole utilise **plusieurs Oracles** indépendants (multi-Oracle), un par domaine (Structure-MNIST cœur, code synthétique, vision optionnelle, etc.), pour découvrir si la SCH est universelle ou domain-spécifique. Soumis à des **quality gates** stricts : pas de GQA/sliding window, dense pure, context length ≥ max(Δ), convergence par plateau validation, BF16 entraînement + FP64 SVD, zéro modification post-entraînement, pas d'agrégation pré-SVD.

**Trois sets disjoints (règle des sets)** — partition stricte des données par domaine en `train_oracle` (entraînement Oracle), `audit_svd` (extraction matrices A pour la SVD phase 2), `init_phase3` (extraction structure pour Smart Init si utilisé). Pré-enregistrée en début de phase 1, jamais modifiée. Sans cette tri-partition, le protocole mesure de la mémorisation au lieu de généralisation.

### Concepts ASP

**ASP (Attention Superlinéaire Polymorphe)** — projet et architecture cibles. Opérateur séquentiel à coût O(N) en moyenne, capable d'allouer dynamiquement un rang de correction low-rank dans les moments critiques.

**ASPLayer** — unité de calcul de référence implémentée en phase 3. Contient un Backbone (SSM ou linéaire), une correction low-rank Matriochka, un Spectromètre. Aussi appelée *Adaptive Structural Processor* (alias).

**Polymorphisme** — capacité d'un modèle, à poids fixés, à changer de régime computationnel (récurrent ↔ attentionnel) sous l'effet d'un coefficient externe (typiquement λ_budget).

**Hiérarchie Matriochka** — organisation des bases U_max, V_max telle que pour tout r ≤ R_max, les r premières colonnes forment elles-mêmes une approximation valide de rang r. Permet une augmentation de rang **continue et non disruptive**. Inspiré de Kusupati et al., *Matryoshka Representation Learning*.

**Loss Matriochka** — terme d'entraînement qui force chaque rang de troncature à produire une **sortie** valide pour la tâche : `Σ_r w_r · L_task(ASPLayer(x; mask=[1×r, 0×(R_max−r)]), y_target)`. Formulation output-based (V3).

**Loss Consistency** — terme V2 ajouté à la loss de phase 3 pour stabiliser la pente du Soft-Mask : `E_{r, δ}[‖y(x;r) − y(x;r+δ)‖²]` avec δ ∈ {−1, +1}. Garantit que la qualité varie *lissement* avec le rang.

**Backbone** — branche stationnaire de l'ASPLayer. **Opérateur structuré paramétré, dérivé du dictionnaire SCH** produit en phase 2 (Toeplitz, Hankel, Cauchy, Vandermonde, ou combinaison). Sa structure est imposée par la mesure, **pas importée** d'une famille d'architecture pré-existante.

**Spectromètre** — mini-réseau qui transforme le signal latent x_t en α_t ∈ [0, 1] (score de rang) ou directement en m_t ∈ [0, 1]^{R_max} (vecteur de gates monotone décroissant). Volontairement petit pour rester interprétable.

**Soft-Masking** — fonction de masquage douce m_{t,i} = σ(β · (α_t · R_max − i + ½)) qui permet la continuité du gradient à travers le choix de rang. Respecte automatiquement la monotonie Matriochka.

**Curriculum de Stress** — protocole d'entraînement de phase 4 en trois étages : warm-up sur bruit (apprendre le rang plancher), injection structurelle légère, stress maximal.

**Phase 4a (Warm-up avec distillation)** — première sous-phase de la calibration : la loss inclut un terme `λ_distil · ‖α_t · R_max − r_target(ω, Δ, ℋ)‖²` qui ancre le Spectromètre sur la loi de transfert phase 2.

**Phase 4b (Apprentissage autonome)** — seconde sous-phase : retrait *total* de la distillation (λ_distil = 0). Le Spectromètre s'auto-calibre sur L_task + λ_budget · L_sparsity uniquement.

**SCH (Structural Correspondence Hypothesis)** — hypothèse centrale de la phase 2. **Reformulation V3.5** : il existe une **distribution conditionnelle** `P(r_eff | ω, Δ, ℋ, domaine)` reproductible et structurée, dont les statistiques (médiane, IQR, percentiles) sont prédictibles à partir des paramètres de stress. Pas une fonction déterministe — une distribution. Statut épistémique d'**hypothèse**, pas de théorème. *Corroborée* (fortement ou faiblement) ou *rejetée*, pas postulée.

**Loi de transfert / Loi de Puissance** — fonction r_target = f(ω, Δ, ℋ) ajustée sur la Stress-Rank Map de phase 2. Forme typique multiplicative : r_target ≈ a · ω^α · Δ^β · g(ℋ). Saint-Graal du protocole : c'est elle que le Spectromètre tente d'imiter dynamiquement.

**Surprise / Signaux** — signaux locaux mesurant la complexité/difficulté structurelle :
- **S_KL** : KL local entre p(x_t | x_<t) et baseline global
- **S_Grad** : norme du gradient local (train uniquement)
- **S_Spectral** : rang effectif r_eff de la sub-matrice d'attention

Validés ou rejetés en phase 1.5.

---

**Version** : 2026-05-12 | **Fusion** : 00_vision + CONTEXT + glossaire
