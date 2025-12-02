import pandas as pd

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
