# 02 — Phase 2 : Audit Spectral et SCH

C'est la phase **scientifique** du protocole. On arrête de deviner, on mesure. On observe comment une architecture de référence (l'**Oracle**) résout les tâches du SSG, puis on extrait le rang structurel effectif que sa solution exige.

## 1. L'Oracle

L'Oracle est une version du modèle dotée d'une capacité de calcul **sur-dimensionnée** par rapport au besoin. En pratique : un **Transformer à attention dense** de taille moyenne, sans aucune contrainte structurelle.

Pourquoi un Oracle ? Parce qu'on ne veut pas mesurer ce qu'un modèle *peut atteindre sous contrainte*, on veut mesurer ce qu'il *consomme effectivement quand rien ne le retient*. Si l'Oracle utilise peu de capacité sur une tâche, c'est que la tâche ne nécessite pas plus.

L'observable principal est la **matrice d'attention A** ∈ ℝ^{N×N} de l'Oracle, extraite (à l'inférence) sur les exemples du SSG.

## 2. Décomposition spectrale

### 2.1 Source des matrices : set held-out obligatoire

L'extraction des matrices A se fait **exclusivement sur le set `audit_svd`** (cf. règle des trois sets disjoints, DOC/01 section 8.6). Ce set n'a pas servi à l'entraînement de l'Oracle.

**Raison** : si on extrait A sur les données vues à l'entraînement, on mesure la mémorisation de l'Oracle, pas sa généralisation. Le r_eff observé reflèterait alors un overfit, pas une demande structurelle réelle de la tâche. La SCH testerait une fausse hypothèse.

### 2.2 Granularité de l'extraction

Pour chaque séquence de stress (ω, Δ, ℋ) du set `audit_svd`, on enregistre A **par tête, par couche, par exemple**. Pas d'agrégation pré-SVD. La SVD est calculée :

```
A_h_ℓ_n = U_h_ℓ_n Σ_h_ℓ_n V_h_ℓ_nᵀ
```

avec Σ = diag(σ₁, σ₂, …, σ_N), σ₁ ≥ σ₂ ≥ … ≥ σ_N ≥ 0, pour chaque tuple (tête h, couche ℓ, exemple n).

Cast en **FP64** au moment de l'extraction (l'Oracle s'entraîne en BF16, mais la SVD se calcule en pleine précision pour éviter les artefacts de quantization).

Les valeurs singulières (σ_k) portent l'énergie spectrale de la matrice d'attention. Leur décroissance traduit la **compressibilité** de A : décroissance rapide → A est essentiellement low-rank ; décroissance lente → A est diffuse.

#### Infrastructure d'extraction (refactor 2026-05-11)

Pour les seq_len cibles de phase 2 (jusqu'à 8192+), l'API d'extraction historique (`extract()` qui matérialise toutes les couches simultanément en FP64) est inadaptée. Le refactor expose trois APIs streaming dans `CODE/phase1_metrologie/oracle/extract.py` :

- **`extract_per_layer(...)`** — générateur yield LayerDump couche par couche. Pic FP64 = 1 couche au lieu de L. À utiliser pour `svd_pipeline.py`, `head_specialization.py`.
- **`extract_windowed_per_layer(..., K, stride)`** — fenêtres K×K diagonales. Pic FP64 = (B,H,K,K). À utiliser pour r_eff local (A1), stress-rank windowed (cf. §3 r_eff par fenêtre).
- **`extract_streamed(..., callback)`** — variante callback pour compute streaming sans matérialiser la liste complète.

Limites :
- Le forward Oracle reste monolithique (toutes les attentions matérialisées dans buffers `.last_attn` pendant le forward). Pour seq_len > ~8192, refactor transformer.py avec hook per-layer requis (TODO post-phase 2).

Cf. DOC/00b §II.8 pour le mapping API ↔ propriété catalogue A-W, et carnet 2026-05-11 14:30 UTC pour le contexte refactor.

## 3. Le rang effectif r_eff

On définit le **rang effectif à seuil θ** comme le plus petit k qui capture une fraction θ de l'énergie spectrale (en énergie totale, pas seulement linéaire) :

```
r_eff(θ) = min { k : (Σ_{i=1..k} σ_i) / (Σ_{j=1..N} σ_j) ≥ θ }
```

Valeurs canoniques : θ = 0,95 (par défaut), θ = 0,99 (mode strict). r_eff doit être rapporté pour les deux seuils, pour vérifier la robustesse de la conclusion au choix du seuil.

> Remarque : r_eff n'est pas formellement le rang de déplacement (rang de Z₁M − MZ₂) — c'est un proxy spectral plus directement actionnable. Le rang de déplacement reste utile pour caractériser la *classe* d'opérateur structuré le mieux adapté (Hankel, Toeplitz, Cauchy, Vandermonde) ; r_eff donne directement le rang à pré-câbler dans l'ASPLayer. Les deux notions coexistent, et la SCH s'exprime sur r_eff.

## 4. Validation de la SCH

La **SCH (Structural Correspondence Hypothesis)** affirme :

> À un niveau de stress structurel correspond une **distribution conditionnelle** `P(r_eff | ω, Δ, ℋ, domaine)` reproductible et structurée, dont les statistiques (médiane, IQR, percentiles, queue) sont prédictibles à partir des paramètres de stress.

**Reformulation V3.5** : la SCH n'est pas l'existence d'une *fonction* r_eff = f(ω, Δ, ℋ), mais l'existence d'une *distribution* dont la forme se déforme régulièrement avec le stress. À (ω, Δ, ℋ) fixé, différents exemples du SSG produisent différents r_eff — la "loi" qu'on cherche à corroborer est sur la distribution complète, pas sur la moyenne.

C'est une hypothèse, pas un théorème : son contenu est empirique, son statut est falsifiable. Elle est *corroborée* si, sur l'espace exploré par le SSG **et sur tous les Oracles** :

1. **Monotonie en tendance centrale** — médiane(r_eff | ω, Δ, ℋ) croît avec ω (idem pour Δ et ℋ).
2. **Reproductibilité** — variance de la médiane entre seeds, couches, têtes inférieure à un seuil pré-enregistré.
3. **Utilité** — il existe une portion non triviale du domaine où la médiane(r_eff) ≪ N.
4. **Concentration prédictible** — IQR(r_eff | régime) est lui-même une fonction prédictible du stress, pas du bruit pur. Critère : `IQR / médiane < seuil` sur une portion majoritaire des régimes.

### 4b. Question ouverte — universalité vs domain-spécificité

Le protocole multi-Oracle (DOC/01 section 7b) ouvre une question fondamentale : la SCH est-elle universelle (mêmes lois pour tous les domaines) ou domain-spécifique (chaque domaine a sa propre loi de transfert) ?

Trois scénarios possibles, à départager empiriquement :

- **SCH universelle.** Les lois de transfert convergent (à un facteur d'échelle près) entre domaines. Une seule architecture ASPLayer suffit à toutes les modalités.
- **SCH domain-spécifique.** Chaque domaine a sa propre loi de transfert, ses primitives structurelles dominantes (Toeplitz pour vision, Hankel pour code, mixte pour langue). L'ASPLayer doit être configurable par domaine.
- **SCH partiellement universelle.** Les régimes de bas stress sont domain-spécifiques (les "easy primitives" varient), mais les régimes de haut stress convergent (la "hard structure" est universelle).

Le résultat oriente directement la conception du Backbone phase 3.

### 4c. Conséquence — corroboration *faible* possible

Une SCH peut être :

- **Fortement corroborée** : conditions 1 + 2 + 3 + 4 satisfaites. Le Spectromètre peut prédire un r_target stable.
- **Faiblement corroborée** : conditions 1 + 2 + 3 OK mais 4 non (IQR comparable à médiane). La loi existe mais bruyante. Le Spectromètre devra prédire un *percentile haut* (ex. p75) plutôt que la médiane, pour éviter la sous-allocation systématique sur la queue droite (cf. DOC/04).
- **Rejetée** : condition 1, 2, ou 3 non satisfaite. Arrêt protocole.

### 4b. Question ouverte — universalité vs domain-spécificité

Le protocole multi-Oracle (DOC/01 section 7b) ouvre une question fondamentale : la SCH est-elle universelle (mêmes lois pour tous les domaines) ou domain-spécifique (chaque domaine a sa propre loi de transfert) ?

Trois scénarios possibles, à départager empiriquement :

- **SCH universelle.** Les lois de transfert convergent (à un facteur d'échelle près) entre domaines. Une seule architecture ASPLayer suffit à toutes les modalités.
- **SCH domain-spécifique.** Chaque domaine a sa propre loi de transfert, ses primitives structurelles dominantes (Toeplitz pour vision, Hankel pour code, mixte pour langue). L'ASPLayer doit être configurable par domaine.
- **SCH partiellement universelle.** Les régimes de bas stress sont domain-spécifiques (les "easy primitives" varient), mais les régimes de haut stress convergent (la "hard structure" est universelle).

Le résultat oriente directement la conception du Backbone phase 3.

## 5. La Stress-Rank Map (Radar RCP)

Sortie centrale de la phase. Reportée non pas comme un nombre par régime mais comme une **distribution complète** :

| Stress (ω) | Stress (Δ) | médiane r_eff(0,95) | IQR | p10 | p90 | queue droite (max) | régime |
|-----------:|-----------:|--------------------:|----:|----:|----:|-------------------:|--------|
| 2 (faible) | 100 | ~4 | 2 | 3 | 6 | 9 | linéaire / constant |
| 6 (moyen) | 500 | ~12 | 6 | 8 | 18 | 25 | logarithmique |
| 12 (critique) | 2000 | ~48 | 20 | 35 | 70 | 95 | récursif profond |

(Chiffres illustratifs ; valeurs réelles produites par l'expérience.)

Variantes attendues :
- Stress-Rank Map monovariée : distribution complète à chaque valeur de ω, Δ, ℋ.
- Stress-Rank Map croisée : carte 2D *médiane(r_eff)* + carte 2D *IQR(r_eff)* — la seconde révèle où la SCH est "stable" vs "bruyante".
- Décomposition par couche : médiane et IQR par (couche, ω) — pour voir si certaines couches portent toute la complexité *et* si la concentration varie selon la couche.
- **Histogrammes par régime** : visualisation de la forme complète de la distribution. Une distribution bimodale ou à queue lourde change l'interprétation par rapport à une distribution gaussienne.

## 5b. Diagnostic de Spécialisation des têtes

Sortie complémentaire à la Stress-Rank Map. Identifie quelles têtes de l'Oracle portent la complexité structurelle, et lesquelles sont "endormies" (statiques quel que soit le stress).

### 5b.1 Calcul

Pour chaque tête h, sur l'ensemble des régimes (ω, Δ, ℋ) du SSG :

```
spec_h = var_{(ω,Δ,ℋ) ∈ régimes}( r_eff_h(ω, Δ, ℋ) )
```

- **spec_h élevée** → tête *spécialisée* : son r_eff varie fortement selon le stress, elle réagit à la difficulté.
- **spec_h faible** → tête *endormie* : son r_eff est quasi-constant, elle ne porte pas de signal de stress utile.

### 5b.2 Statut

Le diagnostic est **obligatoire en livrable de phase 2** dès lors que la phase 3 considère le **Smart Init Matriochka** (cf. DOC/03 section 5). Si la phase 3 utilise uniquement l'init aléatoire, le diagnostic reste informatif mais non load-bearing.

### 5b.3 Garde-fou méthodologique

Le diagnostic **ne modifie pas le calcul de r_eff agrégé** qui définit la SCH. Il s'agit d'une analyse complémentaire de la structure interne de l'Oracle, pas d'un filtre appliqué à la mesure. Pruner les têtes endormies *avant* la SVD est interdit (cf. DOC/01 section 8.5 : zéro modification post-entraînement).

### 5b.4 Sortie

- Distribution de spec_h par couche (figure).
- Liste ordonnée des K têtes les plus spécialisées (consommée par phase 3 en cas de Smart Init).
- Identification des têtes systématiquement endormies (utilisable comme insight architectural pour Phase 3, e.g. "le Backbone n'a pas besoin de tant de têtes").

## 5c. Batterie de tests structurels

Au-delà du calcul de r_eff, la phase 2 exécute une batterie de tests pour caractériser la structure de l'attention de l'Oracle, détecter les limites du catalogue d'opérateurs structurés, et identifier d'éventuelles classes émergentes non enumérées.

**Principe d'extension (à graver dans l'esprit du protocole)** : la batterie ci-dessous est **un point de départ, pas une liste close**. Chaque expérience peut suggérer de nouveaux tests. Toute découverte d'un nouveau test pertinent doit être documentée dans ce document avec sa motivation et son critère, et propagée dans les rapports de phase. **Cataloguer les tests est une tâche permanente, pas un livrable final.**

### Batterie A — Fitting et identification de classe

- **A.1 Erreur de projection par classe.** Pour chaque (régime, classe C ∈ {Toeplitz, Hankel, Cauchy, Vandermonde, Identity}) : `ε_C(régime) = ‖A − proj_C(A)‖_F / ‖A‖_F` où `proj_C` est la projection orthogonale sur la classe C.
- **A.2 Identification de classe dominante.** Pour chaque régime : `class_best = argmin_C ε_C`. Reporter la distribution des classes dominantes par régime.
- **A.3 Composition multi-classe (somme).** Pour chaque paire (C₁, C₂) : `ε_{C₁+C₂} = min ‖A − M₁ − M₂‖` avec M_i ∈ C_i. Si la composition réduit ε significativement vs le meilleur singleton, le Backbone phase 3 doit être *additif* (somme d'opérateurs).
- **A.4 Composition multi-classe (produit).** Idem pour `A ≈ M₁ · M₂`. Compose multiplicatif si phase 3 doit composer (chain) plusieurs opérateurs.

### Batterie B — Analyse du résidu

- **B.1 Norme du résidu.** Après fit du best-of (singleton ou composition), `R = A − M_fit`. Reporter `‖R‖_F / ‖A‖_F` par régime. **Critère go/no-go associé** : si le résidu relatif dépasse un seuil pré-enregistré (ex. 30%) sur une portion non-triviale des régimes, **flag explicit "catalogue insuffisant"** et option d'extension à instruire.
- **B.2 SVD du résidu.** Le résidu R lui-même peut avoir une structure low-rank cachée. Calculer r_eff(R) à θ = 0.95. Si bas, R est compressible — une classe additionnelle manque.
- **B.3 FFT du résidu.** Calculer la transformée de Fourier 2D de R. Pic de fréquence concentré → structure type Toeplitz ou périodique manquée. Spectre étalé → bruit véritable, pas de structure cachée.
- **B.4 PCA cross-régimes du résidu.** Empiler les R(régime) en une matrice et faire PCA. Si les premières composantes principales portent une structure cohérente entre régimes, c'est une *classe émergente* non enumérée — candidate pour extension du catalogue.

### Batterie C — Robustesse et cross-domain

- **C.1 Stabilité par tête / par couche.** La classe dominante varie-t-elle selon (couche ℓ, tête h) pour un même régime ? Si oui, l'architecture phase 3 doit autoriser une diversité par tête (multi-tête au sein de l'ASPLayer).
- **C.2 Cohérence cross-domain.** Pour multi-Oracle : la classe dominante d'un régime (ω, Δ, ℋ) est-elle la même entre Oracles (Structure-MNIST vs code vs vision) ? Convergence → SCH universelle ; divergence → Backbone configurable par domaine.
- **C.3 Invariance par permutation.** Pour A_permuted (entrée permutée), comparer r_eff et class_best à A original. Stabilité → structure indépendante de l'ordre (Cauchy-like). Variabilité → structure liée à l'ordre (Toeplitz/Hankel-like).
- **C.4 Décomposition en blocs.** Partitionner A en blocs B×B. SVD par bloc, comparer r_eff_local à r_eff_global. Si r_eff_local ≪ r_eff_global, structure locale dense + globale low-rank (sparse-block-like).

### Batterie C+ — Robustesse de la paramétrisation du stress (cf. DOC/01 section 2b)

- **C.5 PCA cross-régimes des r_eff.** Empiler les vecteurs (régime, r_eff par couche/tête) et faire une PCA. Projeter dans le plan des deux premières composantes principales et colorer par (ω, Δ, ℋ). Si les régimes forment des clusters propres alignés sur les axes du SSG, les axes capturent bien la difficulté. Si les clusters traversent les axes ou révèlent une dimension orthogonale, **une dimension latente non capturée existe** — flag explicit pour V4.
- **C.6 Test de paramétrisation alternative (V4 minimum viable).** Calculer un proxy de complexité MDL ou de longueur de description algorithmique pour chaque exemple du SSG. Corréler avec r_eff observé. Si la corrélation `ρ(MDL, r_eff)` dépasse celle de `ρ((ω,Δ,ℋ), r_eff)`, la paramétrisation du SSG est sous-optimale.

### Batterie D — Détection d'out-of-catalogue

- **D.1 Régimes orphelins.** Régimes où ε_min(toutes classes + compositions) > seuil élevé (ex. 50%). Marqueurs d'une structure que le catalogue ne capte pas. Lister explicitement.
- **D.2 Signature spectrale fréquentielle.** Pour chaque régime, FFT 1D des lignes de A. Concentration en basses fréquences → Toeplitz ; queues de hautes fréquences → quelque chose de non-classique.
- **D.3 Eigendecomposition vs SVD.** Sur les A approximativement symétriques, comparer rang spectral via valeurs propres vs valeurs singulières. Discrépance → A n'est pas Gram-like, structure asymétrique non capturée par le catalogue de matrices à structure de déplacement classique.
- **D.4 Test non-linéaire.** Le catalogue actuel est *linéaire* (combinaisons linéaires d'opérateurs structurés). Tester si A est mieux approximée par `f(M)` avec f non-linéaire simple (ex. ReLU, soft-thresholding) appliquée à un M structuré. Si oui, signal qu'une classe non-linéaire serait pertinente.

### Sortie de la batterie

- Table d'identification : régime → classe dominante (singleton ou composition) → ε résiduel.
- Liste des régimes orphelins (D.1) avec hypothèses sur la classe manquante.
- Carte de cohérence cross-domain (C.2).
- Recommandations pour le Backbone phase 3 : additif vs composé, configurable vs universel, multi-tête vs simple, linéaire vs non-linéaire.
- **Liste des tests ajoutés au-delà de la batterie initiale**, avec motivation et critère de chaque ajout.

### Protocole d'extension de la batterie

Quand un nouveau test pertinent émerge :

1. Documenter sa motivation (quelle question structurelle il répond).
2. Documenter son critère (qu'est-ce qui constitue un signal positif/négatif).
3. L'ajouter à la batterie (DOC/02 section 5c) avec un ID unique (ex. E.1 pour la cinquième batterie).
4. Le re-exécuter sur les régimes déjà analysés pour cohérence rétrospective.
5. Mentionner dans le rapport de phase 2 si des conclusions changent rétrospectivement.

Cette section est vivante — elle s'enrichit au fil des expériences.

## 6. Loi de transfert : la cible que l'ASP devra imiter

L'objectif final de la phase est de produire une **loi de puissance** (ou plus généralement une fonction de transfert) qui résume la Stress-Rank Map, **par Oracle / par domaine** :

```
r_target_d = f_d(ω, Δ, ℋ)        pour chaque domaine d
```

Forme attendue : ajustement par régression sur la table mesurée. Un point de départ raisonnable est une loi multiplicative

```
r_target_d ≈ a_d · ω^{α_d} · Δ^{β_d} · g_d(ℋ)
```

avec exposants (α_d, β_d) à estimer **séparément par domaine**. La forme exacte n'est pas postulée — c'est un *résultat* de la phase, pas une hypothèse.

Si le scénario "SCH universelle" est vérifié (cf. section 4b), les paramètres convergent : a_d ≈ a, α_d ≈ α, etc. Sinon, on conserve un f_d distinct par domaine.

Cette/ces fonction(s) est/sont une **référence**, pas une cible absolue. Le Spectromètre (phase 4) s'en sert comme prior souple, pas comme plafond — l'ASP est explicitement encouragée à descendre *sous* r_target si la qualité tient (cf. DOC/00 section 3b et test R_max/2 phase 5).

**Statut épistémique** : r_target est ce que l'**Oracle utilise effectivement**, pas ce qui est *suffisant*. Un opérateur structuré paramétré (ASPLayer phase 3) pourrait avoir besoin de moins de rang qu'un Transformer dense pour atteindre la même qualité, si sa structure est mieux adaptée à la tâche (c'est précisément l'hypothèse que tout le projet teste).

## 7. Pourquoi cette phase est indispensable

- **Évite le sur-paramétrage** — si r_eff_max observé sur le SSG est ~16, configurer l'ASPLayer avec R_max = 128 est du gaspillage. La phase 2 chiffre R_max.
- **Définit la cible du Spectromètre** — la loi de transfert sert d'oracle d'apprentissage (au sens : valeurs cibles que le Spectromètre doit reproduire en phase 4, optionnellement par distillation).
- **Rend la falsification possible** — si r_eff n'augmente pas avec ω, soit le SSG est mal conçu, soit la SCH est rejetée. Aucun de ces deux verdicts n'est confortable, mais tous deux sont un résultat.

## 8. Critère go/no-go

**Go** : SCH corroborée selon les trois conditions de la section 4. Loi de transfert ajustée avec un R² acceptable (seuil pré-enregistré) sur les données du SSG. R_max recommandé chiffré.

**No-go** :
- r_eff sature à N partout → l'attention vraie n'est pas approximable par une structure low-rank. Hypothèse 2 (compressibilité) rejetée.
- r_eff ne croît pas avec le stress → la mesure n'est pas causalement reliée à la difficulté. La métrologie de phase 1 est à reprendre.
- Variance entre seeds/couches/têtes au-delà du seuil pré-enregistré → r_eff mesure du bruit, pas une propriété de la tâche.

## 9. Pièges méthodologiques

- **Mesurer r_eff sur A vs. mesurer la dégradation aval.** r_eff (énergie spectrale) est un proxy de la capacité utilisée ; ce qui compte *in fine* est la dégradation aval quand on remplace A par sa projection de rang r_eff. Vérifier la cohérence : à r_eff(0,95), la dégradation aval doit rester sous un seuil ε_aval pré-enregistré.
- **Sur-spécialisation au SSG.** Avant de fermer la phase, vérifier sur au moins une tâche tierce (sous-ensemble LRA) que la loi de transfert n'est pas un artefact de Structure-MNIST.
- **Choix du seuil θ.** Toujours rapporter au moins deux valeurs (0,95 et 0,99). Ne jamais ajuster θ après coup pour faire passer un test.
- **Couches et têtes.** Agréger trop tôt (moyenne sur toutes les têtes) peut masquer une concentration de capacité sur quelques têtes. Faire l'agrégation après analyse.
