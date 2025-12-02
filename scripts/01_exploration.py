import pandas as pd
from src.data_processing import load_dataset, show_data_info
""" from src.visualization import (
    plot_histograms,
    plot_boxplots,
    plot_correlations,
)
 """
# 1. Charger les données
df = load_dataset("data/raw/dataset.csv")

# 2. Nettoyer les colonnes
df = show_data_info(df)
