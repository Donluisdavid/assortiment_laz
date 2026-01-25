# Projet Store Adaptor : Prévision d'Assortiment (mon trigramme LAZ)

[![Python Version](https://img.shields.io/badge/python-3.12.10-blue.svg)](https://www.python.org/downloads/release/python-31210/)
[![Framework](https://img.shields.io/badge/Model-LightGBM-orange.svg)](https://lightgbm.readthedocs.io/)
[![CI/CD](https://img.shields.io/badge/CI-GitHub%20Actions-green.svg)](https://github.com/features/actions)

Ce projet implémente un moteur de prévision de ventes, dont l'objectif est d'optimiser l'assortiment magasin en anticipant les volumes de ventes à 4 mois.

---

## 1. Contexte et objectifs

Dans le secteur de la grande distribution, l'optimisation de l'assortiment est un levier majeur de performance. Une mauvaise estimation du volume de ventes entraîne deux risques majeurs :
* **La Rupture de Stock** : Perte directe de chiffre d'affaires et dégradation de l'expérience client.
* **Le Sur-stockage** : Immobilisation de capital et risque de gaspillage (démarque), particulièrement sur les produits frais.

Ce projet propose un prototype d'un **pipeline de forecasting automatisé** capable de prédire les volumes de ventes, pour chaque couple magasin (**Agency**) / produit (**SKU**), sur un horizon de **4 mois**. Cet outil permet aux gestionnaires d'assortiment d'anticiper les tendances saisonnières et d'ajuster les approvisionnements de manière proactive.

## 2. Approche de la Solution

Nous alons se reposer sur une approche avec du **Machine Learning supervisé**, notamment l'algorithme **LightGBM**, choix est motivé par sa rapidité d'entraînement, sa robustesse, entre autres.

### Stratégie de Prédiction : L'Inférence Récursive

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

La phase de preprocessing transforme les données brutes en signaux exploitables par le modèle.

Vue l'approach recursirve avec un modèle de ML, nous allons nous resteindre que sur des variables qu'on pourrait générer à chaque itération facilement. Alors, des variables basées que sur : la variable **date**, et la variable **volume**.

**Remarques du périmètre de variables, en considérant l'approach itérative avec un modèle de ML:** 
* On aurait pu inclure **price_regular, price_actual, discount, avg_population, avg_yearly_household_income_2017, discount_in_percent** (très importantes pour indiquer au modèle les variations en prix, ainsi que des effets locaux des magasins). Sauf que on n'a pas moyen simple de les regénerer pour les itération pour les prochains mois. 
* On aurait plus facilement inclure **easter_day, good_friday, new_year,	christmas, labor_day, independence_day,	revolution_day_memorial,	regional_games, fifa_u_17_world_cup, football_gold_cup,	beer_capital, music_fest** (très importantes pour considérer les effets des temps forts du calendrier). Celles si peuvent facilment plus se générer à chaque itération, à condition d'avoir les règles de gestion pour les construire. Vue qu'on ne les a pas, je les ai omise. Cependant, d'autres variables plus simples basées sur le calendrier sont incluses.
* On aurait pu transformer **sku** et **agency** en variables catégorielles(et faire un hot enconding, un embeding ou autre), mais ça aurait fait plus lourde l'exercise dans la prod (gestion des nouvelles classes). Puis, dans la vrai vie, mettre l'id de l'article et le magasin comme feature, personellement je ne le conseille pas.

#### Feature Engineering
* **Variables Temporelles :** Extraction du **mois** et de l'**année** pour capturer les cycles saisonniers. Pour faire simple et ne pas avoir des problèmes de gestion des classes dans l'industrialisation, je les ai laissé telles quelles.
* **Variables de Décalage (lags) :** 
    * `Lag_1` : Volume du mois précédent (tendance immédiate).
    * `Lag_12` : Volume du même mois l'année précédente (saisonnalité annuelle).
    Celles ci sont calculées par couple sku/agency, pour garder certaine cohérence. Des règles de gestions de nulls très simples ont été mises en place. Dans un contexte industriel, d'autres méthodes plus réalistes seraient utilisés.
* **Variables Calendaires :** Flags binaires pour les événements majeurs (is_winter, is_springtime, etc).

Celui-ci c'est un approach très simple du preprocessing et feature engineering, mais il reste factuel et pragmatique, vue le contexte du developpement du cas d'usage.

### Training du modèle

La variable  à apprendre c'est **volume**.

Vue que :
* date/agency/sku represente une ligne à apprendre ce volume correspondant, 
* on ne considère aucune caractéristique des produits dans les features pour le modèle, 

Alors, on considère que les volumes de ventes de tous les produits, partoutes les magasins  se distribuent de la même façon. C'est un approach simpliste, mais efficace vue les contraintes temporaires.

**Remarque :**
Un modèle de ML va considérer chaque ligne comme indépendente l'un de l'autre. Malgré le fait d'avoir inclut des variables qui essayent de considérer la structure temporaire, saisonnaire, et des temps forts, elles sont très simples. 

#### Hyperparamètres et métrique d'évaluation 
Aucun tunning n'a été effectué, vue le context très short ainsi que les hypothèses très lourdes faites sur le cas d'usage. Cependant, nous avons inclut pour l'entrainement des **hyperparamètres** fixes qui jouent sur la compléxité du modèle et sur son sur apprentissage (avec un early stopping inclut aussi).

Pour l'entrainement du modèle, nous avons choisi le **RMSE** car il pénalise plus sévèrement les fortes erreurs de prévision, tout en conservant une métrique facile à gérer pendant l'entrainement, et à interpréter.

Les résultats sont gardés dans un dictionnaire (lartifact), avec : **le model, le data preprocessor, et la liste des features dans la modélisation**.

### Inférence

On prend : 
* Notre jeu de données d'inférence, avec la dernière date renseignée de nos données,
* L'artifact du modele (avec le modèle, le preprocessing, et la liste de features),

Ainsi, itérativement, avec un horizon qui va de 1 à 4,
* génère les features (avec le **preprocessing**) pour **M+H**,
* génère les prédictions (avec le modèle, les features préalablement générées, et la liste des features) les prédictions pour **M+H**,
* on utilise la prediction de **M+H** comme volume base pour l'itération suivante.

**Remarque :**
Dans cette version de l'inférence, on ne reçoit que les lignes de décembre 2017. Ce qui est insuffissant pour générer les variables **lag**. Même si ma gestion de nulls gère cela,en réalité, pour éviter ça, dans la vraie vi pour l'inférence on devrait interoger des bases avec l'historique nécessaire pour avoir nos lags. 

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

## 7. Evolutions possibles




