# Projet Store Adaptor : Prévision d'Assortiment (mon trigramme LAZ)

[![Python Version](https://img.shields.io/badge/python-3.12.10-blue.svg)](https://www.python.org/downloads/release/python-31210/)
[![Framework](https://img.shields.io/badge/Model-LightGBM-orange.svg)](https://lightgbm.readthedocs.io/)
[![CI/CD](https://img.shields.io/badge/CI-GitHub%20Actions-green.svg)](https://github.com/features/actions)

Ce projet implémente un moteur de prévision de ventes, dont l'objectif est d'optimiser l'assortiment magasin en anticipant les volumes de ventes à 4 mois.

---

## 1. Contexte et objectifs

Dans le secteur de la grande distribution, l'optimisation de l'assortiment est un levier majeur de performance. Une mauvaise estimation du volume de ventes entraîne deux risques majeurs :
* **La Rupture de Stock** : perte directe de chiffre d'affaires et dégradation de l'expérience client.
* **Le Sur-stockage** : immobilisation de capital et risque de gaspillage (démarque), particulièrement sur les produits frais.

Ce projet propose un prototype d'un **pipeline de forecasting automatisé** capable de prédire les volumes de ventes, pour chaque couple magasin (**Agency**) / produit (**SKU**), sur un horizon de **4 mois**. Cet outil permet aux gestionnaires d'assortiment d'anticiper les tendances saisonnières et d'ajuster les approvisionnements de manière proactive.

## 2. Approche de la Solution

Nous alons se reposer sur une approche avec du **Machine Learning supervisé**, notamment l'algorithme **LightGBM**, choix est motivé par sa rapidité d'entraînement, sa robustesse, entre autres.

### Stratégie de prédiction : l'inférence récursive

Pour répondre au besoin de prévision à 4 mois, le projet utilise une stratégie **récursive** : 

1. Le modèle prédit le volume pour le mois **M+1**.
2. Cette prédiction est ensuite réinjectée comme donnée d'entrée, pour prédire le mois **M+2**.
3. Le processus se répète jusqu'à atteindre l'horizon cible de 4 mois.

#### Cette stratégie  permet de n'utiliser qu'un seul modèle robuste.

#### Avantages :
* **Cohérence temporelle :** Le modèle conserve une "mémoire" des tendances qu'il vient de prédire.
* **Simplicité opérationnelle :** Un seul modèle à entraîner, monitorer et à déployer en production.
* **Flexibilité :** Cela permet de changer l'horizon de prévision (passer de 4 à 6 mois par exemple) sans avoir à ré-entraîner de nouveaux modèles.

### Stratégie des jeux de données de training, de validation, et d'inference 

Pour garantir la robustesse du modèle et sa capacité à généraliser sur des périodes futures, une stratégie de **split temporel** stricte a été adoptée. Contrairement à un découpage aléatoire, cette méthode respecte la chronologie des ventes et évite toute fuite de données dans le temps.

| Set | Période | Objectif Business |
| :--- | :--- | :--- |
| **Entraînement** | 2013 — Nov. 2016 | Apprentissage des tendances de fond et des comportements historiques. |
| **Validation** | Déc. 2016 — Nov. 2017 | **Cycle annuel complet** : Test de la performance du modèle sur une année entière pour valider la gestion de la saisonnalité (Noël, été, promos), pour éviter toute fuite de données (Data Leakage) et garantir la fiabilité des prévision futures. |
| **Inférence** | Décembre 2017 | Point de départ de la donnée fraîche pour projeter les prévisions récursives sur l'horizon 2018. |

#### Pourquoi ce choix pour le set d'inférence?
Vue que sur le jeu de données, il n'y a qu'une seule ligne par date/agency/sku, et que Décembre 2017 est la dernière date renseignée, alors, ce choix simule exactement le comportement du pipeline en production : utiliser le dernier mois consolidé pour prévoir les mois à venir.

### Architecture du Projet
Le projet adopte une structure modulaire pour garantir une industrialisation propre :
* `src/preprocessing.py` : Nettoyage et ingénierie des caractéristiques (features).
* `src/training.py` : Logique d'entraînement et sérialisation du modèle.
* `src/inference.py` : Moteur de prédiction récursive à 4 mois.
* `main.py` : Orchestrateur central du pipeline.

---

## 3. Preprocessing, training, inference

### Preprocessing & Feature Engineering

La phase de preprocessing transforme les données brutes en signaux exploitables. Compte tenu de l'approche récursive choisie, le périmètre des variables a été sélectionné selon leur capacité à être projetées dans le futur (itérations $M+n$).

#### Variables retenues

Pour expliquer le volume future de ventes, on a retenu que les variables **volume** et **date**.

#### Feature Engineering
* **Variables Temporelles :** Extraction du **mois** et de l'**année** pour capturer les cycles saisonniers. Pour faire simple et ne pas avoir des problèmes de gestion des classes dans l'industrialisation, je les ai laissé telles quelles.
* **Variables de Décalage (lags) :** 
    * `Lag_1` : Volume du mois précédent (tendance immédiate).
    * `Lag_12` : Volume du même mois l'année précédente (saisonnalité annuelle).
    Celles ci sont calculées par couple sku/agency, pour garder certaine cohérence. Des règles de gestions de nulls très simples ont été mises en place. Dans un contexte industriel, d'autres méthodes plus réalistes seraient utilisés.
* **Variables Calendaires :** Flags binaires pour les événements majeurs (is_winter, is_springtime, etc).

#### Arbitrages sur le périmètre des variables
Certaines variables présentes dans le dataset n'ont pas été incluses dans ce prototype pour des raisons de conception :

* **Variables Commerciales (price, discount) :** bien que critiques, leur utilisation en mode récursif nécessiterait de connaître le plan promotionnel futur. À défaut de ces données prospectives, elles ont été écartées pour éviter de biaiser les itérations futures avec des valeurs statiques.

* **Événements Spécifiques (Fêtes, Calendrier Sportif) :** ces variables du type **temps Forts** sont très impactantes mais nécessitent des règles de gestion précises (dates mobiles comme Pâques). Par souci de robustesse pour ce test, nous avons privilégié des variables calendaires déterministes.

* **Identifiants (sku, agency) :** Plutôt que d'utiliser des IDs bruts (qui posent des problèmes de scalabilité et de gestion des nouvelles références en production), j'ai privilégié une approche basée sur le comportement des séries temporelles. En milieu industriel, il est préférable de remplacer ces IDs par des caractéristiques métiers (catégories, formats de magasin) pour favoriser la généralisation du modèle.

### Stratégie d'apprentissage (training)

La variable cible est le **volume de ventes mensuel**. C'est à dire :

Plutôt que d'entraîner un modèle par produit/magasin, nous utilisons un modèle unique pour l'ensemble du catalogue. Cette approche permet au modèle de capter des comportements communs (saisonnalités partagées) et d'être plus robuste face aux séries ayant peu d'historique.

**Hypothèse de distribution :** En n'incluant pas de caractéristiques spécifiques (métadonnées) aux produits ou aux agences, le modèle apprend une dynamique de vente "moyenne" basée sur les historiques (lags) et le calendrier. C'est une approche qui privilégie la stabilité de la prévision.

**Considérations techniques :** Un modèle de ML traite chaque ligne de manière indépendante. Pour compenser cette absence de structure temporelle native (contrairement à un modèle ARIMA ou LSTM), nous avons injecté cette structure via le **feature engineering** (lags, variables calendaires). Bien que simplifiée, cette méthode permet de transformer un problème de série temporelle en un problème de régression supervisée classique, en profitant du pouvoir du ML.

#### Hyperparamètres et évaluation du modèle

**Hyperparamètres et régularisation** : dans l'optique de livrer un prototype fonctionnel dans des délais restreints, aucun ajustement automatique (tuning) n'a été réalisé. Cependant, une configuration manuelle robuste a été appliquée au modèle LightGBM : avec des hypermaramètres qui controlent la compléxité et l'overfitting du modèle (ainsi qu'un early stopping).

**Métrique de performance** : j'ai sélectionné le RMSE comme fonction de perte, par sa propriété à pénaliser plus sévèrement les erreurs importantes, un point critique en logistique pour éviter les ruptures de stock majeures, tout en restant une métrique stable et interprétable pour l'optimisation mathématique.

** Gestion des artefacts : à l'issue de l'entraînement, un objet structuré ("artifact") est sauvegardé, avec toutes les composants nécessaires à l'inférence : **le model, le data preprocessor, et le schema des features**.

### Stratégie d'inférence
L'inférence est conçue pour produire des prévisions sur un horizon de 4 mois. Étant donné que le modèle utilise des variables retardées (lags), nous appliquons une stratégie itérative :

* Le dernier état connu du jeu de données,
* L'artefact d'entrainement(le modèle, le preprocessing, et le schema de features),

Ainsi, itérativement, avec un horizon qui va de 1 à 4,
* génère les features (avec le **preprocessing**) pour **M+H**,
* génère les prédictions (avec le modèle, les features préalablement générées, et la liste des features) les prédictions pour **M+H**,
* on utilise la prediction de **M+H** comme volume base pour l'itération suivante.

**Remarque :**
Dans cette version de l'inférence, on ne reçoit que les données de décembre 2017. Pour un calcul rigoureux des varialbes lags (à M-1 et à M-12), le pipeline devrait en réalité être connecté à la base originale avec l'historique. Bien que la gestion acutelle des valeurs nulles permette l'exécution du script, une mise en production nécessiterait l'extraction de l'historique glissant. 

#### Rendu final

Fichier CSV avec le format

| reference_date | agency | sku | last_volume | predict_m1 | predict_m2 | predict_m3 | predict_m4 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| xxxx-xx-xx | yyy | zzz | aaa | bb | ccc | d | eee |

---

## 4. Installation et execution

### Pré-requis
La configuration est centralisée dans un pyproject.toml. Cela permet de gérer les dépendances d'une façon moderne, ainsi que standardiser les outils de qualité de code comme pytest dans un seul fichier de configuration.

* **Environnement virtuel** recommandé.

1. **Quick start :**
   ```bash
   git clone https://github.com/Donluisdavid/assortiment_laz.git
   pip install .

### **Exécution**
Elle est divisé en 3 : preprocessing, training, et inference.

1. **Preprocessing**

    ```
    bash python main.py
    *(Transforme le fichier d'entrée, pour obtenir les jeux de données de train, validation, et inference dans le dossier data)*

2. **Training**

    ```
    bash python main.py training
    *(Génère l'artifact du modèle dans le dossier models)*

3. **Inference**

    ```
    bash python main.py inference
    *(prend le fichier d'inférence, génère les prédictions, et les dépose dans le dossier data)

## 5. Tests et Qualité

Le projet suit les standards de développement avec une suite de tests automatisés. La configuration est centralisée dans le fichier `pyproject.toml`.

### Exécution des Tests
Nous utilisons `pytest` pour valider la logique des transformations (preprocessing) et la robustesse de l'inférence.

    ```
    bash
    pytestco
    *(Lance tous les codes unitaires)*

Grâce à pytest-cov, un rapport de couverture est généré automatiquement pour s'assurer que les parties critiques du pipeline sont bien testées.

Pour l'instant, j'ai réalisé qu'un test pour le pre processing, vue les contraintes temporaires. Les tests pour le training et l'inférence viendront bientôt. 

## 7. Axes d'amélioration et perspectives

### Segmentation et stratégies de modélisation 

Dans un contexte industriel réel, traiter l'ensemble des références avec un modèle unique est sous-optimal. Une approche plus fine consisterait à segmenter le catalogue pour adapter la granularité et l'algorithme :

* **Articles "Top Ventes" & Forte Fréquence (Permanent/Stable) :**

    * Stratégie : Modélisation individuelle à la maille journalière.
    * Objectif : Précision maximale sur les produits qui génèrent le plus de flux.
* **Articles "Top Ventes" & Faible Fréquence (Intermittent) :**
    * Stratégie : Agrégation à la maille sous-famille ou cluster pour stabiliser le signal.
* **Articles à Faible Volume (Slow-movers) :**
    * Stratégie : Utilisation du Clustering pour regrouper les profils de vente similaires et entraîner un modèle global par cluster (maille hebdomadaire).
* **Articles "Erratiques" (Faibles ventes/Faible fréquence) :**
    * Stratégie : Modélisation simplifiée ou gestion par stock de sécurité (approche statistique de Poisson) plutôt que par ML pur.

### Enrichissement du périmètre des features.
Des familles plus larges de signaux peuvent être considérées dans la modélisation, pour accroître la précision de la solution. 

La méthode itérative a fait qu'on a du éviter d'utiliser certaines des variables dans le jeu de données. Cependant, c'est totalement possible de regarder variable par variable si on peut connaitre en avance ses valeurs, avoir un proxy, voire si on peut les prédire. Mais peut être cette donnée peut déjà exister dans d'autres projets en interne. En tout cas, vu que très probablement dans la réalité on devoir donner les prédictions des 4 prochains mois (plutôt ça sera des prédictions des ventes dans les jours à venir), ça sera plus simple. 

* **Leviers promotionnels et prix** : intégrer le plan commercial (prix, reemises, promotions). Ces données sont prédictibles, car planifiées généralement à l'avance.   
* **Sources externes** : données socio-démographiques type INSEE, ou données météorologiques (réel et prévision), surtout pour les produits météo-sensibles. Contexte économique, comme l'inflation alimentaire.
* **Caractéristiques des produits et des magasins** . 
 * **Variables provenant du calendrier** : Au-delà de la saisonnalité mensuelle, intégrer les "temps forts" du retail : vacances scolaires, jours fériés mobiles (Pâques, Ramadan) et événements sportifs majeurs.
 * **Calendrier concurrenciel** : Identifier les ouvertures/fermetures de concurrents directs ou les opérations commerciales majeures de la zone de chalandise.

### Evolution stratégique de la modélisation

L'approche actuelle repose sur un modèle de Machine Learning (Gradient Boosting). Bien que performante pour capter des relations non linéaires complexes, cette approche peut être enrichie pour mieux appréhender la structure temporelle des ventes.

#### 1. Approche Statistique & Séries Temporelles (TS)
Au-delà du ML pur, l'utilisation de modèles de séries temporelles classiques (ARIMA, ETS) ou de librairies comme Prophet permettrait de mieux décomposer les signaux :

* **Tendance** : Capturer les évolutions de fond du marché à long terme.
* **Saisonnalité** : Modéliser explicitement les cycles annuels, mensuels ou hebdomadaires inhérents au retail.

#### 2. Modélisation Hybride (Two-Step Modeling)
Une approche prometteuse consisterait à fusionner les deux mondes via un modèle à deux couches :

* **Couche 1 (Statistique) :** Extraction des composantes de tendance et de saisonnalité pour chaque couple Magasin/Produit.
* **Couche 2 (Machine Learning) :** Utilisation des résidus de la première couche comme cible, ou intégration des composantes extraites comme features dans le modèle ML. Cela permet de déléguer la structure temporelle aux statistiques et la complexité des variables externes (promos, météo) au ML.

#### 3. Deep Learning pour les Séries Temporelles
Pour gérer des volumes de données massifs, l'exploration d'architectures de Deep Learning spécialisées pourrait être envisagée :

* **LSTM / GRU :** Pour capter des dépendances temporelles à long terme.
* Temporal Fusion Transformers (TFT) : Pour une approche à l'état de l'art capable de gérer des horizons de prédiction multiples tout en restant interprétable.

### Entrainement

Pour transformer ce prototype en un modèle de production performant, plusieurs axes d'amélioration technique sont envisagés :

#### Fine-tuning et feature selection
* Optimistaion de hyperparamètres via **Optuna**, pour ajuste finement les paramètres du LightGBM,
* Réduction de dimensionnalité, avec des méthodes des réduction de variables, pour éliminer les bruits et améliorer la capacité de généralisation du modèle. 

### Raffinement des métriques de performance
Le RMSE, bien que robuste mathématiquement, doit être complété par des indicateurs plus proches des réalités logistiques :

* **WAPE (Weighted Absolute Percentage Error) :** pour offrir une mesure de l'erreur en pourcentage, facilitant l'interprétation par les équipes métier.

* **Analyse du Biais de Prévision :** mesurer systématiquement si le modèle tend vers le sur-stockage (Over-forecasting) ou la rupture (Under-forecasting).

* **Intervalles de Confiance (Quantile Regression) :** Au lieu d'une prédiction ponctuelle, produire des intervalles (ex: Quantiles 10% et 90%) pour permettre aux gestionnaires de stock de définir des stocks de sécurité basés sur le risque.

### Vers une industrialisation MLOps
Pour garantir la pérennité et la fiabilité de la solution en production, le pipeline devrait évoluer vers une architecture MLOps complète, structurée autour de l'automatisation et de l'observabilité :

#### Automatisation du cycle de vie (CI/CD/CT)
* **Continuous Training (CT) :** Mise en place d'un déclencheur automatique pour le réentraînement du modèle, soit périodiquement, soit lorsque la performance descend en dessous d'un seuil prédéfini.
* **Model Registry :** Utilisation d'un outil pour versionner les artefacts, suivre les hyperparamètres et faciliter le "rollback" en cas de régression.

#### Monitoring et Observabilité
Le succès d'un modèle en retail dépend de sa capacité à s'adapter aux changements de comportement des consommateurs :

* **Data Drift Analysis :** Surveillance des dérives de distribution des variables d'entrée via des tests statistiques (KS Test, PSI).
* **Concept Drift :** suivi de l'évolution de la relation entre les caractéristiques et les ventes.

#### Suivi de la Performance Métier en Temps Réel
Intégrer un dashboard de monitoring focalisé sur la santé opérationnelle du modèle :

* **Analyse de l'asymétrie :** uivi temps réel des taux d'**over-forecasting** et d'**under-forecasting** pour alerter les équipes logistiques.
* **Backtesting glissant :** Évaluation continue de la fiabilité des intervalles de confiance pour ajuster dynamiquement les stocks de sécurité.confiance, etc.  

### Aide à la décision et opérationnalisation
Pour favoriser l'adoption du modèle par les équipes métier (gestionnaires d'assortiment et approvisionneurs), la restitution des résultats doit évoluer :

#### Transition vers des prévisions probabilistes
Fournir une valeur unique (moyenne) est souvent insuffisant pour la gestion des stocks. L'introduction d'intervalles de confiance (ex: 80% ou 90%) permet de quantifier l'incertitude :
* **Prise de décision éclairée :** les équipes peuvent choisir de stocker selon la borne haute (**prévenir la rupture**) ou la borne basse (**limiter le gaspillage**), selon la criticité du produit.
* **Réassurance Métier** : Un intervalle backtesté et fiable renforce la confiance des utilisateurs dans les prédictions du modèle.

#### Dashboards d'exception (management by exception)
Plutôt que de demander aux équipes de vérifier chaque ligne, le système peut alerter uniquement sur les anomalies de prévision ou les changements brusques de tendance, permettant aux opérationnels de se concentrer sur les références à fort enjeu.

## Architecture Cible sur Google Cloud Platform (GCP)

Pour industrialiser ce projet sur GCP, je m'appuierais sur Vertex AI.

* **Orchestration & Entraînement :** Vertex AI Pipelines pour automatiser l'enchaînement des tâches (préparation des données -> entraînement -> inférence). Le déclenchement serait planifié (*scheduling*) pour s'exécuter périodiquement (ex: chaque jour ou chaque semaine).

* ** Stockage :**

    * **Données :** Idéalement, toute la data serait stockés dans BigQuery.
    * **Modèles :** Le modèle final (.pkl) serait sauvegardé dans le Vertex AI Model Registry, pour gérer les différentes versions.

* **Exposition des Prédictions :** on va assumer qu'une exposition en temps réel n'est pas nécessaire (rare dans des cas de grand distribution comme les supermarchés). Alors, j'utiliserais des **Batch Predictions** : le modèle lit les données dans BigQuery et réécrit les prévisions dans une table finale, consultable par les outils de dashboarding (comme Looker).

* **Monitoring :** J'utiliserais Vertex AI Model Monitoring.




