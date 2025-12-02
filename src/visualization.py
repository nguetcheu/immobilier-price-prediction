import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 🔹 Histogrammes pour colonnes numériques
def plot_histograms(df: pd.DataFrame, numeric_cols: list):
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution de {col}")
        plt.xlabel(col)
        plt.ylabel("Fréquence")
        plt.show()

# 🔹 Boxplots pour détecter les outliers
def plot_boxplots(df: pd.DataFrame, numeric_cols: list):
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot de {col}")
        plt.show()

# 🔹 Heatmap de corrélation avec la cible
def plot_correlation_with_target(df, target: str):
    # Sélectionner uniquement les colonnes numériques
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if target in numeric_df.columns:
        corr_matrix = numeric_df.corr()
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(8,6))
        sns.heatmap(corr_matrix[[target]].sort_values(by=target, ascending=False),
                    annot=True, cmap="coolwarm")
        plt.title(f"Corrélations des features avec {target}")
        plt.show()

# 🔹 Graphiques pour features catégorielles
def plot_categorical_counts(df: pd.DataFrame, cat_cols: list):
    for col in cat_cols:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=col, data=df, order=df[col].value_counts().index)
        plt.title(f"Répartition de {col}")
        plt.xticks(rotation=45)
        plt.show()

