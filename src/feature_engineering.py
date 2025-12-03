import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

def create_features(df: pd.DataFrame) -> pd.DataFrame:
  """
  Créer de nouvelles features :
  - surface_par_piece = surface_habitable / nb_pieces
  - prix_m2 = prix / surface_habitable
  """
  df = df.copy()
    
  if "surface_habitable" in df.columns and "nb_pieces" in df.columns:
    df["surface_par_piece"] = df["surface_habitable"] / df["nb_pieces"]
    
  if "prix" in df.columns and "surface_habitable" in df.columns:
    df["prix_m2"] = df["prix"] / df["surface_habitable"]
    
  # Arrondir à 2 décimales les nouvelles features
  for col in ["surface_par_piece", "prix_m2"]:
    if col in df.columns:
      df[col] = df[col].round(2)
    
  return df

## Corrélation avec la cible
def correlation_importance(df: pd.DataFrame, target: str) -> pd.Series:
    """Importance via corrélation absolue."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()[target].abs().sort_values(ascending=False)
    return corr.drop(target)

## Importance par un modèle d’arbres (Random Forest)
def tree_importance(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Importance basée sur RandomForest."""
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    return pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

## Importance par permutation
def permutation_importance_score(model, X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Permutation importance."""
    perm = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    return pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

## Sélection des top features
def select_top_features(tree_imp, corr_imp, perm_imp, top_n=15):
    """Fusionner les 3 méthodes et sélectionner les top features."""
    df = pd.DataFrame({
        "tree": tree_imp,
        "corr": corr_imp,
        "perm": perm_imp
    })

    df["mean_score"] = df.mean(axis=1)
    df_sorted = df.sort_values("mean_score", ascending=False)

    return df_sorted.head(top_n).index.tolist(), df_sorted