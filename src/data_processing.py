import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_dataset(path: str = "data/raw/dataset.csv") -> pd.DataFrame:
    """Charger le dataset brut depuis le dossier data/raw"""
    return pd.read_csv(path)

def save_processed_data(df: pd.DataFrame, path: str = "data/processed/dataset_clean.csv"):
    """Sauvegarder le dataset nettoyé dans data/processed"""
    df.to_csv(path, index=False)

def show_data_info(df: pd.DataFrame):
    """Afficher les informations principales du dataset"""
    print("DataFrame Info:")
    print(df.info())
    print("\nMissing Values per Column:")
    print(df.isnull().sum())
    print("\nStatistical Summary:")
    print(df.describe(include='all'))

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyer et préparer les données pour ML :
    - Remplacer les valeurs manquantes numériques par la médiane
    - Arrondir certaines colonnes float à 2 décimales
    - Supprimer .0 pour les colonnes entières
    - Encoder les colonnes catégorielles pour ML
    - Transformer date_mise_vente en année et mois
    """
    df = df.copy()
    
    # 1️⃣ Remplacer les valeurs manquantes
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64', 'Int64']:
            median = df[col].median()
            df[col] = df[col].fillna(median)
        else:
            df[col] = df[col].fillna("Inconnu")
    
    # 2️⃣ Arrondir à 2 décimales pour certaines colonnes float
    round_2_cols = ["surface_habitable", "distance_centre", "distance_transport", "prix"]
    for col in round_2_cols:
        if col in df.columns:
            df[col] = df[col].round(2)
    
    # 3️⃣ Colonnes entières (supprimer .0)
    int_cols = ["nb_chambres", "annee_construction", "parking", "score_commerces"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: int(x) if pd.notna(x) else 0)
            df[col] = df[col].astype("Int64")
    
    # 4️⃣ Encoder les colonnes catégorielles
    cat_cols = ["ville", "quartier", "type_bien", "etat", "chauffage", "classe_energie"]
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    # 5️⃣ Transformer date_mise_vente en année et mois
    if "date_mise_vente" in df.columns:
        df["date_mise_vente"] = pd.to_datetime(df["date_mise_vente"], errors='coerce')
        df["annee_vente"] = df["date_mise_vente"].dt.year.fillna(df["date_mise_vente"].dt.year.median())
        df["mois_vente"] = df["date_mise_vente"].dt.month.fillna(df["date_mise_vente"].dt.month.median())
        df = df.drop(columns=["date_mise_vente"])
    
    return df
