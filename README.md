# Projet Store Adaptor : Prévision d'Assortiment (mon trigramme LAZ)

[![Python Version](https://img.shields.io/badge/python-3.12.10-blue.svg)](https://www.python.org/downloads/release/python-31210/)
[![Framework](https://img.shields.io/badge/Model-LightGBM-orange.svg)](https://lightgbm.readthedocs.io/)
[![CI/CD](https://img.shields.io/badge/CI-GitHub%20Actions-green.svg)](https://github.com/features/actions)

Ce projet implémente un moteur de prévision de ventes, dont l'objectif est d'optimiser l'assortiment magasin en anticipant les volumes de ventes à 4 mois.

---

## 1. Contexte et Problématique

Dans le secteur de la grande distribution, l'optimisation de l'assortiment est un levier majeur de performance. Un mauvais calibrage entraîne soit des ruptures de stock, soit du surstock.

**Le défi :** Prédire avec précision les volumes de ventes à horizon 4 mois pour chaque couple magasin (**Agency**) / produit (**SKU**), en tenant compte de.

## 2. Approche de la Solution

Nous alons se reposer sur une approche avec du **Machine Learning supervisé**, notamment l'algorithme **LightGBM**, choix est motivé par sa rapidité d'entraînement, sa robustesse, entre autres.

### Stratégie de Prédiction : L'Inférence Récursive

Pour répondre au besoin de prévision à 4 mois, le projet utilise une stratégie **récursive** (ou *Iterative Forecasting*). 

#### Pourquoi ce choix ?
Cela  permet de n'utiliser qu'un seul modèle robuste.

Le principe est le suivant :
1. Le modèle prédit le volume pour le mois **M+1**.
2. Cette prédiction est ensuite réinjectée comme donnée d'entrée, pour prédire le mois **M+2**.
3. Le processus se répète jusqu'à atteindre l'horizon cible de 4 mois.

#### Avantages :
* **Cohérence temporelle :** Le modèle conserve une "mémoire" des tendances récentes qu'il vient de prédire.
* **Simplicité opérationnelle :** Un seul modèle à entraîner, monitorer et déployer en production.
* **Flexibilité :** Cette approche permet de changer l'horizon de prévision (passer de 4 à 6 mois par exemple) sans avoir à ré-entraîner de nouveaux modèles.

### Stratégie des jeux de données de training, de validation, et d'inference 

Pour garantir la robustesse du modèle et sa capacité à généraliser sur des périodes futures, une stratégie de **split temporel** stricte a été adoptée. Contrairement à un découpage aléatoire, cette méthode respecte la chronologie des ventes et évite toute fuite de données dans le temps.

| Set | Période | Objectif Business |
| :--- | :--- | :--- |
| **Entraînement (Train)** | 2013 — Nov. 2016 | Apprentissage des tendances de fond et des comportements historiques. |
| **Validation** | Déc. 2016 — Nov. 2017 | **Cycle annuel complet** : Test de la performance du modèle sur une année entière pour valider la gestion de la saisonnalité (Noël, été, promos). |
| **Inférence (Test)** | Décembre 2017 | Point de départ ("Seed") de la donnée fraîche pour projeter les prévisions récursives sur l'horizon 2018. |

#### Pourquoi ce choix ?
1. **Validation sur 12 mois** : En utilisant une fenêtre de validation d'un an, nous nous assurons que les hyperparamètres du modèle ne sont pas sur-optimisés pour une seule saison, mais pour la réalité annuelle du secteur retail.
2. **Réalisme opérationnel** : Le passage au set d'inférence en Décembre 2017 simule exactement le comportement du pipeline en production dans la grande distribution : utiliser le dernier mois consolidé pour prévoir les mois à venir.

### Architecture du Projet
Le projet adopte une structure modulaire pour garantir une industrialisation propre :
* `src/preprocessing.py` : Nettoyage et ingénierie des caractéristiques (Features).
* `src/training.py` : Logique d'entraînement et sérialisation du modèle.
* `src/inference.py` : Moteur de prédiction récursive à 4 mois.
* `main.py` : Orchestrateur central du pipeline.

---

## 3. Focus : Preprocessing & Feature Engineering

La phase de preprocessing transforme les données brutes en signaux exploitables par le modèle.

### Feature Engineering
* **Variables Temporelles :** Extraction du mois et de l'année pour capturer les cycles saisonniers.
* **Variables de Décalage (Lags) :** * `Lag_1` : Volume du mois précédent (tendance immédiate).
    * `Lag_12` : Volume du même mois l'année précédente (saisonnalité annuelle).
* **Variables Calendaires :** Flags binaires pour les événements majeurs (Noël, Labour Day).

"hypothèse de remplissage des données manquantes"


---

## 4. Installation 

### Pré-requis
La configuration est centralisée dans un pyproject.toml. Cela permet de gérer les dépendances, ainsi que standardiser les outils de qualité de code comme pytest dans un seul fichier de configuration.
* **Environnement virtuel** recommandé.

### Configuration
1. **Cloner le projet :**
   ```bash
   git clone [https://github.com/Donluisdavid/assortiment_laz.git](https://github.com/Donluisdavid/assortiment_laz.git)
   cd assortiment_laz

2. **Créer et activer l'environnement :**

    ```bash
    python -m venv .venv
    source .venv/Scripts/activate

3. **Installer les dépendances :** 

    ```bash 
    pip install .

4. **Installer avec les outils de dveloppement (testt, linting) :**
    ```bash
    pip install -e .[dev]

## 5. **Exécution**
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

## 6. Tests et Qualité

Le projet suit les standards de développement avec une suite de tests automatisés. La configuration est centralisée dans le fichier `pyproject.toml`.

### 1. Exécution des Tests
Nous utilisons `pytest` pour valider la logique des transformations (preprocessing) et la robustesse de l'inférence.

    ```
    bash
    pytestco
    *(Lance tous les codes unitaires)*

Grâce à pytest-cov, un rapport de couverture est généré automatiquement pour s'assurer que les parties critiques du pipeline sont bien testées.




