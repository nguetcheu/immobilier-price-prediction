# 📋 TODO - Projet Prédiction Prix Immobiliers

## 🎯 Répartition des Tâches

### Membre 1 : NGUETCHEU KUINSI Dominique
**Responsable** : Phases 1 - 6

### Membre 2 : WENJI PASCAL Victor
**Responsable** : Phases 1 - 6

### Collaboration : Phase 7

---

## Phase 1 : Exploration (EDA) 🔍 
**Script** : `scripts/01_exploration.py`  
**Responsable** : Membre 1

- [ ] Charger le dataset (`pandas.read_csv`)
- [ ] Afficher les premières lignes et info du dataset
- [ ] Statistiques descriptives (`describe()`)
- [ ] Vérifier les types de données
- [ ] Créer histogrammes pour toutes les variables numériques
- [ ] Créer boxplots pour détecter les outliers
- [ ] Calculer matrice de corrélation
- [ ] Visualiser heatmap des corrélations
- [ ] Identifier corrélations avec la cible (prix)
- [ ] Compter et visualiser les valeurs manquantes
- [ ] Documenter les outliers identifiés
- [ ] Rédiger conclusions préliminaires

**Livrables** :
- Notebook complété
- 5-10 visualisations sauvegardées dans `results/figures/`
- Fichier `data/processed/eda_summary.csv` avec statistiques

---

## Phase 2 : Feature Engineering 🛠️  
**Notebook** : `02_feature_engineering.ipynb`  
**Responsable** : Membre 1

### 2.1 Création de Features
- [ ] `surface_par_piece` = surface totale / nombre de pièces
- [ ] `prix_par_m2` = prix / surface (si disponible)
- [ ] `age_propriete` = année actuelle - année construction
- [ ] `ratio_chambres` = chambres / pièces totales
- [ ] Features d'interaction (ex: `surface * nb_pieces`)

### 2.2 Variables Catégoriques
- [ ] Identifier toutes les variables catégoriques
- [ ] One-Hot Encoding pour variables avec peu de modalités (<10)
- [ ] Label Encoding ou Target Encoding pour variables nombreuses
- [ ] Vérifier absence de multicolinéarité après encoding

### 2.3 Valeurs Manquantes
- [ ] Stratégie pour chaque colonne (médiane, mode, prédiction)
- [ ] Implémenter imputation intelligente
- [ ] Créer indicatrices de valeurs manquantes si pertinent
- [ ] Documenter choix d'imputation

### 2.4 Normalisation
- [ ] StandardScaler pour variables numériques
- [ ] Vérifier distribution après scaling
- [ ] Sauvegarder scaler pour réutilisation

**Livrables** :
- Notebook complété
- Dataset transformé : `data/processed/features_engineered.csv`
- Fichier `src/feature_engineering.py` avec fonctions réutilisables

---

## Phase 3 : Feature Selection 🎯
**Notebook** : `03_feature_selection.ipynb`  
**Responsable** : Membre 1

### 3.1 Méthode 1 : Tree-Based Importance
- [ ] Entraîner RandomForest sur toutes les features
- [ ] Extraire `feature_importances_`
- [ ] Visualiser top 20 features

### 3.2 Méthode 2 : Permutation Importance
- [ ] Utiliser `sklearn.inspection.permutation_importance`
- [ ] Calculer sur modèle Random Forest
- [ ] Comparer avec méthode 1

### 3.3 Méthode 3 : Corrélation
- [ ] Calculer corrélation Pearson avec cible
- [ ] Sélectionner features avec |corr| > 0.3
- [ ] Identifier features redondantes (corr entre elles > 0.9)

### 3.4 Sélection Finale
- [ ] Croiser les 3 méthodes
- [ ] Sélectionner top 15 features
- [ ] Entraîner modèle avec toutes features (baseline)
- [ ] Entraîner modèle avec 15 features sélectionnées
- [ ] Comparer performances (RMSE, R², temps d'entraînement)

**Livrables** :
- Liste finale de 15 features dans `results/selected_features.txt`
- Graphiques de comparaison
- Dataset réduit : `data/processed/features_selected.csv`

---

## Phase 4 : Modélisation 🤖
**Notebook** : `04_modelisation.py`  
**Responsable** : Membre 2

### 4.1 Préparation
- [ ] Charger dataset avec features sélectionnées
- [ ] Split train/test (70/30)
- [ ] Définir fonction d'évaluation (RMSE, R², MAE)

### 4.2 Modèles à Entraîner
- [ ] **Linear Regression** (baseline)
  - Entraînement
  - Cross-validation 5-fold
  - Métriques

- [ ] **Ridge Regression**
  - Tester alpha = [0.1, 1, 10, 100]
  - Cross-validation 5-fold
  - Métriques

- [ ] **Random Forest**
  - Paramètres par défaut
  - Cross-validation 5-fold
  - Métriques

- [ ] **Gradient Boosting** (XGBoost ou LightGBM)
  - Paramètres par défaut
  - Cross-validation 5-fold
  - Métriques

### 4.3 Comparaison
- [ ] Tableau comparatif des 4 modèles
- [ ] Graphique barplot des métriques
- [ ] Identifier le meilleur modèle
- [ ] Analyser temps d'entraînement

**Livrables** :
- Notebook complété
- Fichier `results/metrics/models_comparison.csv`
- 4 modèles sauvegardés dans `models/`

---

## Phase 5 : Optimisation ⚙️
**Deadline** : [Date]  
**Notebook** : `05_optimisation.py`  
**Responsable** : Membre 2

### 5.1 Choix du Modèle
- [ ] Sélectionner le meilleur modèle de la Phase 4
- [ ] Documenter pourquoi ce modèle

### 5.2 GridSearchCV
- [ ] Définir grille d'hyperparamètres (3-4 paramètres clés)
  - Exemple Random Forest : `n_estimators`, `max_depth`, `min_samples_split`
  - Exemple Gradient Boosting : `learning_rate`, `n_estimators`, `max_depth`
- [ ] Configurer GridSearchCV (cv=5, scoring='neg_mean_squared_error')
- [ ] Lancer optimisation (peut prendre du temps !)
- [ ] Extraire meilleurs paramètres

### 5.3 Évaluation
- [ ] Entraîner modèle avec paramètres par défaut
- [ ] Entraîner modèle avec paramètres optimisés
- [ ] Comparer métriques avant/après
- [ ] Calculer gain de performance (%)

**Livrables** :
- Meilleurs hyperparamètres dans `results/best_params.json`
- Modèle optimisé : `models/best_model_tuned.pkl`
- Comparaison avant/après

---

## Phase 6 : Pipeline & Validation 🔄
**Deadline** : [Date]  
**Notebook** : `06_pipeline_validation.py`  
**Responsable** : Membre 2

### 6.1 Pipeline Complet
- [ ] Créer Pipeline sklearn :
  ```
  Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(k=15)),
    ('model', BestModelTuned())
  ])
  ```
- [ ] Tester pipeline sur données brutes

### 6.2 Évaluation Test Set
- [ ] Charger données de test (30% séparés au début)
- [ ] Prédictions avec pipeline
- [ ] Calculer métriques finales (RMSE, R², MAE)
- [ ] Comparer avec métriques de validation

### 6.3 Analyse des Erreurs
- [ ] Calculer résidus (y_true - y_pred)
- [ ] Graphique Actual vs Predicted (scatter plot)
- [ ] Graphique résidus vs prédictions
- [ ] Identifier top 10 pires prédictions
- [ ] Analyser pourquoi le modèle se trompe

### 6.4 Visualisations Finales
- [ ] Distribution des résidus (histogramme)
- [ ] QQ-plot des résidus
- [ ] Feature importance du modèle final
- [ ] Courbes d'apprentissage (learning curves)

**Livrables** :
- Pipeline complet : `models/final_pipeline.pkl`
- Fichier `results/metrics/final_evaluation.csv`
- 5-8 visualisations dans `results/figures/`

---

## Phase 7 : Rapport & Recommandations 📊
**Deadline** : [Date]  
**Document** : `reports/rapport_final.md` ou `.pdf`  
**Responsable** : **COLLABORATION MEMBRE 1 + MEMBRE 2**

### 7.1 Structure du Rapport (3-5 pages)

#### Introduction
- [ ] Contexte du projet
- [ ] Problématique
- [ ] Objectifs

#### Méthodologie
- [ ] Description du dataset
- [ ] Approche feature engineering
- [ ] Modèles testés
- [ ] Méthode d'évaluation

#### Résultats
- [ ] Performances des modèles (tableau)
- [ ] Meilleur modèle et paramètres
- [ ] Métriques finales sur test set
- [ ] Visualisations clés (3-4 graphiques)

#### Analyse
- [ ] Features les plus importantes
- [ ] Cas d'usage bien prédits
- [ ] Cas problématiques
- [ ] Résidus et erreurs

#### Recommandations pour l'Agence Immobilière
- [ ] Comment utiliser le modèle
- [ ] Fourchette de confiance des prédictions
- [ ] Facteurs clés influençant le prix
- [ ] Stratégies d'évaluation immobilière

#### Limitations et Perspectives
- [ ] Limites du modèle actuel
- [ ] Données supplémentaires souhaitables
- [ ] Améliorations futures
- [ ] Risques et précautions

#### Conclusion
- [ ] Synthèse des résultats
- [ ] Réponse à la problématique

### 7.2 Annexes
- [ ] Code source principal
- [ ] Graphiques complémentaires
- [ ] Références

**Livrables** :
- Rapport final (PDF)
- Présentation PowerPoint (10-15 slides)
- Code source propre et commenté

---

### Documentation
- [ ] README.md à jour
- [ ] Chaque notebook a une introduction claire
- [ ] `requirements.txt` complet
- [ ] Commentaires dans le code
