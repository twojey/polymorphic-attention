# Glossaire

Termes algébriques, architecturaux et protocolaires utilisés dans le projet ASP. Définitions opérationnelles, pas exhaustives.

## Algèbre linéaire structurée

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

**H-matrix / H²-matrix / HSS** (Hackbusch) — formats de matrices hiérarchiques avec rang faible par blocs. H-matrix : O(Nk log N). H²-matrix / HSS : O(Nk) grâce à la **nestedness** (emboîtement des bases parent-fils). Voir [00b §I.2 (3)](00b_classification_proprietes.md).

**Nestedness** — propriété d'une matrice hiérarchique où la base d'un cluster parent peut être exprimée à partir des bases de ses enfants via une matrice de transfert de petite taille. Clé du O(N) strict pour H²/HSS.

**Théorème de Mercer** — un noyau continu, symétrique, défini positif K(x,y) admet une décomposition K(x,y) = Σ λ_i φ_i(x) φ_i(y) en série convergente. Si tronquable à D ≪ N termes, opérateur en O(ND).

**Théorème de Bochner** — un noyau stationnaire K(x-y) est défini positif ssi sa transformée de Fourier est une mesure positive. Justifie l'approximation par Random Fourier Features.

**Random Fourier Features (RFF)** — Rahimi-Recht 2007. Pour un noyau stationnaire K(x-y), échantillonner ω_j selon la densité spectrale μ et construire φ_j(x) = e^{iω_j^T x}. Donne K̃(x,y) = (1/D) Σ φ_j(x) φ_j(y)* avec erreur sup ∼ O(1/√D). Permet kernel approx en O(ND).

**Tensor Train (TT)** (Oseledets 2011) — décomposition d'un tenseur d'ordre élevé en chaîne de tenseurs d'ordre 3 avec rangs TT bornés. Stockage en O(d N r²), produit MV en O(r² N log N).

**Butterfly / Monarch** — matrices sparse-structurées admettant un produit MV en O(N log N) ou O(N√N). Fondement des transformées rapides (FFT, Hadamard) et des architectures Dao et al. 2022.

## Architectures séquentielles

**SSM (State Space Model)** — modèle à récurrence linéaire h_t = A h_{t-1} + B x_t, y_t = C h_t + D x_t. Coût O(N) en inférence, capacité bornée par la dimension d'état.

**S4, S5, Mamba, Mamba2** — variantes de SSM avec parameterizations spécifiques (HiPPO, sélectivité, scan parallèle). Choix précis différé.

**Attention pleine** — softmax(QKᵀ/√d) V, coût O(N²).

**Attention linéaire** — réécriture par feature map φ pour passer en O(N), au prix d'une expressivité réduite.

**Hyena, RWKV** — autres familles d'opérateurs séquentiels sub-quadratiques, comparateurs naturels en phase 5.

**MoD (Mixture of Depths)** — modèle d'allocation dynamique de calcul par token, Raposo et al. (DeepMind 2024). Routage **binaire** : chaque token passe ou saute la couche. Baseline principal de phase 5 pour évaluer la valeur ajoutée de l'allocation **spectrale** (continue) de l'ASP par rapport à l'allocation binaire de MoD.

**Oracle** — Transformer à attention dense, sur-dimensionné par rapport au besoin, entraîné en phase 1 *from scratch* sur un domaine donné. Sa matrice d'attention A est l'observable consommé en phase 2 (SVD, r_eff). Le protocole utilise **plusieurs Oracles** indépendants (multi-Oracle), un par domaine (Structure-MNIST cœur, code synthétique, vision optionnelle, etc.), pour découvrir si la SCH est universelle ou domain-spécifique (DOC/01 section 7b). Soumis à des **quality gates** stricts : pas de GQA/sliding window, dense pure, context length ≥ max(Δ), convergence par plateau validation, BF16 entraînement + FP64 SVD, zéro modification post-entraînement, pas d'agrégation pré-SVD (cf. DOC/01 section 8).

**Trois sets disjoints (règle des sets)** — partition stricte des données par domaine en `train_oracle` (entraînement Oracle), `audit_svd` (extraction matrices A pour la SVD phase 2), `init_phase3` (extraction structure pour Smart Init si utilisé). Pré-enregistrée en début de phase 1, jamais modifiée. Sans cette tri-partition, le protocole mesure de la mémorisation au lieu de généralisation, ou injecte de l'information du test set vers l'init de l'ASPLayer.

**Diagnostic de Spécialisation des têtes** — sortie complémentaire de phase 2 (section 5b). Pour chaque tête h : `spec_h = var(r_eff_h)` à travers les régimes (ω, Δ, ℋ). Têtes spécialisées (haute spec_h) vs endormies (basse spec_h). Obligatoire si la phase 3 utilise Smart Init Matriochka, optionnel sinon. Ne modifie pas r_eff agrégé.

**Smart Init Matriochka** — stratégie d'initialisation de phase 3 (DOC/03 section 5.2) consommant le Diagnostic de Spécialisation : les premières colonnes de U_max, V_max sont initialisées à partir des vecteurs singuliers des têtes spécialisées de l'Oracle (extraits sur le set `init_phase3`). Reste : init aléatoire. Status : ablation, jamais défaut. Le défaut reste **random init**.

## Concepts ASP

**ASP (Attention Superlinéaire Polymorphe)** — projet et architecture cibles. Opérateur séquentiel à coût O(N) en moyenne, capable d'allouer dynamiquement un rang de correction low-rank dans les moments critiques.

**ASPLayer** — unité de calcul de référence implémentée en phase 3. Contient un Backbone (SSM ou linéaire), une correction low-rank Matriochka, un Spectromètre. Aussi appelée *Adaptive Structural Processor* (alias).

**Adaptive Structural Processor** — synonyme de l'ASPLayer, soulignant le caractère adaptatif (rang variable) et structurel (basé sur la mesure de la SCH).

**Polymorphisme** — capacité d'un modèle, à poids fixés, à changer de régime computationnel (récurrent ↔ attentionnel) sous l'effet d'un coefficient externe (typiquement λ_budget).

**Hiérarchie Matriochka** — organisation des bases U_max, V_max telle que pour tout r ≤ R_max, les r premières colonnes forment elles-mêmes une approximation valide de rang r. Permet une augmentation de rang **continue et non disruptive**. Inspiré de Kusupati et al., *Matryoshka Representation Learning*.

**Loss Matriochka** — terme d'entraînement qui force chaque rang de troncature à produire une **sortie** valide pour la tâche : `Σ_r w_r · L_task(ASPLayer(x; mask=[1×r, 0×(R_max−r)]), y_target)`. Formulation output-based (V3) — la version matrice-based (v1/v2) était mal posée car U/V sont statiques et ne peuvent pas approximer simultanément des matrices A* dépendantes de l'entrée. Couplée systématiquement à L_consistency.

**Loss Consistency** — terme V2 ajouté à la loss de phase 3 pour stabiliser la pente du Soft-Mask : `E_{r, δ}[‖y(x;r) − y(x;r+δ)‖²]` avec δ ∈ {−1, +1}. Garantit que la qualité varie *lissement* avec le rang. Sans ce terme, des sauts de qualité entre rangs voisins rendent la phase 4 instable. Coefficient λ_C calibré pour passer le sanity check de lissité.

**Backbone** — branche stationnaire de l'ASPLayer. **Opérateur structuré paramétré, dérivé du dictionnaire SCH** produit en phase 2 (Toeplitz, Hankel, Cauchy, Vandermonde, ou combinaison). Sa structure est imposée par la mesure, **pas importée** d'une famille d'architecture pré-existante (Mamba2, S4, etc.). Coût O(N).

**Extension low-rank** — branche dynamique de l'ASPLayer. Décomposition U·Vᵀ avec rang r_t ∈ [r_min, R_max] modulé en temps réel par le Spectromètre.

**Rang adaptatif r_t** — rang de la correction low-rank au pas t. Sortie effective du Spectromètre.

**R_target** — rang entier visé par le Spectromètre, projection de α_t · R_max via Soft-Masking ou STE.

**Spectromètre** — mini-réseau qui transforme le signal latent x_t en α_t ∈ [0, 1] (score de rang) ou directement en m_t ∈ [0, 1]^{R_max} (vecteur de gates monotone décroissant). Volontairement petit pour rester interprétable et résistant à la fraude.

**Soft-Masking** — fonction de masquage douce m_{t,i} = σ(β · (α_t · R_max − i + ½)) qui permet la continuité du gradient à travers le choix de rang. Respecte automatiquement la monotonie Matriochka.

**STE (Straight-Through Estimator)** — astuce de continuité : forward avec valeur arrondie, backward avec gradient continu. Implémentation : `r_continuous + (r_discrete − r_continuous).detach()`.

**Gumbel-Softmax** — échantillonnage différentiable d'une distribution catégorielle, paramétré par une température. Variante stochastique du masquage, à utiliser en deuxième intention.

**Curriculum de Stress** — protocole d'entraînement de phase 4 en trois étages : warm-up sur bruit (apprendre le rang plancher), injection structurelle légère, stress maximal. Évite le gel précoce du Spectromètre. Activé pendant la sous-phase 4a.

**Phase 4a (Warm-up avec distillation)** — première sous-phase de la calibration : la loss inclut un terme `λ_distil · ‖α_t · R_max − r_target(ω, Δ, ℋ)‖²` qui ancre le Spectromètre sur la loi de transfert phase 2. Curriculum de Stress activé. Valide uniquement sur tâches synthétiques où (ω, Δ, ℋ) sont des labels connus.

**Phase 4b (Apprentissage autonome)** — seconde sous-phase : retrait *total* de la distillation (λ_distil = 0). Le Spectromètre s'auto-calibre sur L_task + λ_budget · L_sparsity uniquement. C'est la configuration évaluée en phase 5.

**Critère de transition 4a → 4b** — trois conditions pré-enregistrées : convergence de L_distillation, corrélation Spearman R_target ↔ r_target théorique > 0,80, absence de plafond artificiel. Mesure de la dérive aux N premiers steps de 4b.

**Diagramme de Phase** — livrable visuel de la phase 4. Trace R_target choisi par le Spectromètre vs. complexité réelle de la tâche. Doit être une fonction croissante.

**Surprise** — signal local mesurant à quel point l'état courant n'est pas prévisible / structuré. Proxys candidats : KL local (S_KL), norme du gradient (S_Grad), rang spectral local (S_Spectral). Validés ou rejetés en phase 1.5.

**S_KL** — KL local entre p(x_t \| x_<t) et un baseline empirique global. Insensible au bruit blanc si le baseline est correctement calibré. Coût d'inférence O(V).

**S_Grad** — norme du gradient local par rapport à L_task. Train uniquement (pas d'inférence). Asymétrie noise/structure : fire aussi sur le bruit. Probable rejet en phase 1.5.

**S_Spectral** — rang effectif r_eff de la sub-matrice d'attention sur fenêtre glissante K. Mesure directe de la complexité. Insensible au bruit par construction. Coût d'inférence O(K² log K) via SVD partielle randomisée.

**S_Surprise (retiré V3)** — anciennement quatrième candidat (résidu prédictif d'un Backbone proxy). Retiré car (1) il aurait nécessité d'instancier un Backbone pré-existant, en violation du principe Discovery > Reproduction (DOC/00 section 4b), et (2) il aurait probablement échoué le test d'immunité au bruit de toute manière.

**Distillabilité** — hypothèse de phase 1.5b : un MLP léger peut prédire S_Spectral à partir de x_t avec ρ > 0.85. Si validée, on déploie le MLP comme student au lieu de la SVD directe. Sinon, fallback sur S_Spectral en direct ou simplification du Backbone.

**Identification Gate** — nom alternatif de la phase 1.5. Verrou intercalaire : aucun signal validé → arrêt du protocole. C'est le point de défaillance numéro 1 du projet.

**Multi-signal** — autorisation pour le Spectromètre de recevoir plusieurs signaux validés en entrée si chacun couvre un axe de stress différent (ex. S_KL pour ω, S_Spectral pour Δ). Décision multi-factorielle d'allocation de rang.

**Agrégation Max-Pool / Concat** — schéma standard de réduction des signaux de phase 1.5 : max-pool sur les têtes par couche, concaténation sur la deep-stack. Sortie : vecteur de dimension L (nombre de couches) par signal.

**λ_budget** — coefficient de la loss de budget. λ_budget grand → modèle paresseux ; λ_budget faible → modèle gourmand.

**Loss de budget (L_sparsity)** — pénalité sur l'usage cumulé du rang. Forme canonique (1/T) Σ_t α_t. Variantes : pondération Matriochka, L1, quadratique.

**Frontière de Pareto** — ensemble des points (qualité, coût) non dominés. Pour ASP, paramétrée par λ_budget.

## Principes méthodologiques

**Discovery > Reproduction** — principe fondateur (DOC/00 section 4b) : le protocole vise à *découvrir* la structure mathématique que l'attention exhibe sur des données réelles, pas à reproduire une architecture existante. Conséquences : pas de pré-sélection de famille de Backbone (Mamba, S4, etc.), pas de signal calibré sur un proxy importé (S_Surprise retiré), pas de comparateurs sacralisés sans audit.

**Multi-Oracle (Protocole)** — phase 1 entraîne plusieurs Oracles indépendants, un par domaine (Structure-MNIST cœur ; code synthétique ; vision optionnelle ; etc.). Chacun part de zéro, sans transfert ni pretraining. L'isomorphisme architectural entre Oracles permet d'attribuer les différences de signature spectrale au domaine plutôt qu'à l'architecture. Permet de tester si la SCH est universelle ou domain-spécifique.

**Universalité vs Domain-spécificité (SCH)** — question ouverte de phase 2 (DOC/02 section 4b). Trois scénarios : SCH universelle (mêmes lois pour tous les domaines), SCH domain-spécifique (chaque domaine a sa propre loi), SCH partiellement universelle (régimes faciles divergent, régimes difficiles convergent). Réponse oriente la conception du Backbone phase 3.

## Protocole et instrumentation

**SSG (Structural Stress Generator)** — banc de stress de phase 1. Produit Structure-MNIST avec balayages contrôlés sur ω, Δ, ℋ. Chaque axe est varié séparément avant les balayages croisés.

**Structure-MNIST** — dataset synthétique généré par le SSG, dérivé de MNIST, paramétré par (ω, Δ, ℋ).

**ω, Δ, ℋ** — axes de stress du SSG. ω = profondeur de récursion ; Δ = distance de dépendance ; ℋ = entropie de la tâche.

**RCP (Reverse Capacity Profiling)** — méthodologie de phase 1.

**SCH (Structural Correspondence Hypothesis)** — hypothèse centrale de la phase 2. **Reformulation V3.5** : il existe une **distribution conditionnelle** `P(r_eff | ω, Δ, ℋ, domaine)` reproductible et structurée, dont les statistiques (médiane, IQR, percentiles) sont prédictibles à partir des paramètres de stress. Pas une fonction déterministe — une distribution. Statut épistémique d'**hypothèse**, pas de théorème. *Corroborée* (fortement ou faiblement) ou *rejetée*, pas postulée. Renommée depuis la version v1/v2 où elle s'appelait SCT.

**Loi de transfert / Loi de Puissance** — fonction r_target = f(ω, Δ, ℋ) ajustée sur la Stress-Rank Map de phase 2. Forme typique multiplicative : r_target ≈ a · ω^α · Δ^β · g(ℋ). Saint-Graal du protocole : c'est elle que le Spectromètre tente d'imiter dynamiquement.

**Stress-Rank Map (Radar RCP)** — table et figures associées qui résument r_eff sur l'espace (ω, Δ, ℋ). Sortie centrale de la phase 2.

**Heatmap de Rang** — visualisation de R_target sur des séquences typées. Livrable de la phase 5, montre l'allumage sélectif.

**SE (Structural Efficiency)** — métrique mathématique d'efficience de phase 5 :
```
SE = Accuracy / Average_Rank
```
Hardware-indépendante. Mesure le rendement de chaque unité de rang allouée. Cible ASP : SE 2–3× supérieur au Transformer.

**HT (Hardware Throughput)** — métrique matérielle d'efficience de phase 5 :
```
HT = Accuracy / Inference_Time
```
Hardware-dépendante. Mesure l'exploitation du silicium. Cible ASP : au moins équivalent au SSM pur.

**Note historique : SRI**. La V1/V2 utilisait `SRI = Accuracy / (Average_Rank × Inference_Time)`. Cette métrique composite confondait deux grandeurs hétérogènes (math vs hardware) et pouvait classer dans le mauvais sens. La V3 la **remplace** par SE et HT rapportés séparément. Ne plus utiliser SRI.

**Lag de Réaction** — métrique du test d'Élasticité (phase 5). Distance en tokens entre l'apparition d'une structure et le moment où R_target atteint la moitié de sa montée. Bon ASP : Lag ≈ 0–2 tokens.

**Test d'Identifiabilité** — test de vérité de phase 5. Vérifie que R_target reste au plancher sur du bruit blanc. Échec → modèle frauduleux.

**Test d'Élasticité** — test de vérité de phase 5. Vérifie la réponse en escalier (montée et descente) de R_target sur des séquences sandwich [bruit][structure][bruit].

**Test OOD (Hors Distribution)** — test de robustesse de phase 5 (V3 : croisement d'axes). Entraînement sur un axe de stress dominant (récursion ω), évaluation sur un autre (binding Δ), et inversement. Vérifie que le Spectromètre a appris la *loi* (SCH), pas des *mots-clés* axe-spécifiques.

**Test différentiel d'activation (5a.ii, V3.5)** — test ajouté pour distinguer un Spectromètre *silencieux* (R_target = 0 par défaut, passif) d'un Spectromètre *intelligent* (R_target catégorise activement les inputs). 4 conditions : bruit blanc, null/empty, répétition triviale, structuré. Métrique = somme des KL divergences entre paires de distributions. Critère go : diff_score > seuil_actif ET R_target sur structuré > R_target sur les 3 autres.

**Test R_max réduit (6c, V3.5)** — test décisif où l'ASP est entraînée avec `R_max = médiane(r_eff_oracle) / 2`. Si la qualité tient, ASP a structurellement *dépassé* l'Oracle. Obligatoire pour toute claim "ASP plus efficace que Transformer" au-delà du polymorphisme dynamique.

**Batterie de tests structurels (DOC/02 section 5c, V3.5)** — ensemble ouvert de tests menés en phase 2 pour caractériser la structure d'A et détecter les classes manquantes du catalogue. 4 batteries initiales (A fitting, B résidu, C robustesse, D out-of-catalogue) + extensions. **Liste vivante** : nouveaux tests ajoutés au fil des expériences, documentés avec motivation et critère.

**Loss asymétrique (V3.5)** — la distillation phase 4a pénalise la sous-allocation (α·R_max < r_target_p75) plus fortement que la sur-allocation (γ·1 quadratique avec γ ∈ [0.1, 0.3]). Justification : sous-allouer dégrade la qualité (coût élevé), sur-allouer ne fait que consommer du compute (déjà contraint par λ_budget).

**Binding** — au sens cognitif : liaison entre features distinctes (variable binding, anaphora, coreference). Dans le SSG actuel, *opérationnalisé* comme régime Δ-dominé (tâches de référence, copie, association à distance, sans récursion profonde). Raccourci documenté ; un enrichissement futur du SSG avec de vraies tâches de liaison (variable assignment, résolution pronominale) est listé comme extension possible mais non requis pour le test 5e actuel.
