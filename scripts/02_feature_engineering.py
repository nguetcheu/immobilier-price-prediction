from src.data_processing import load_dataset, clean_data, save_processed_data
from src.feature_engineering import create_features

# Charger le dataset
df = load_dataset("data/raw/dataset.csv")

# Nettoyer les données
df_clean = clean_data(df)

# Créer des features supplémentaires
df_features = create_features(df_clean)

# Sauvegarder le dataset prêt pour ML
save_processed_data(df_features, "data/processed/dataset_clean.csv")
