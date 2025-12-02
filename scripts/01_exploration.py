import pandas as pd
from scipy import stats
import numpy as np

from src.data_processing import load_dataset, show_data_info
from src.visualization import plot_histograms, plot_boxplots, plot_correlation_with_target, plot_categorical_counts

# 🔹 1️⃣ Charger le dataset brut
df = load_dataset("data/raw/dataset.csv")

# 🔹 2️⃣ Afficher les infos principales
show_data_info(df)

# 🔹 3️⃣ Identifier les colonnes numériques et catégorielles
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = ['ville', 'quartier', 'type_bien', 'etat', 'chauffage', 'classe_energie']

# 🔹 4️⃣ Visualiser les distributions numériques
plot_histograms(df, numeric_cols)

# 🔹 5️⃣ Boxplots pour détecter les outliers
plot_boxplots(df, numeric_cols)

# 🔹 6️⃣ Corrélations avec la cible 'prix'
if 'prix' in df.columns:
    plot_correlation_with_target(df, 'prix')

# 🔹 8️⃣ Valeurs manquantes
missing_values = df.isnull().sum()
print("\nValeurs manquantes par colonne :\n", missing_values)

# 🔹 9️⃣ Détection des outliers simples (z-score > 3)
for col in numeric_cols:
    z_scores = np.abs(stats.zscore(df[col].dropna()))
    outliers = np.where(z_scores > 3)
    print(f"{col} : {len(outliers[0])} outliers détectés")

print("\nExploration terminée !")
