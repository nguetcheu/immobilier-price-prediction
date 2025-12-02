import pandas as pd

def load_dataset(path: str = "data/raw/dataset.csv") -> pd.DataFrame:
  return pd.read_csv(path)

def save_processed_data(df, path: str = "data/processed/dataset_clean.csv"):
  df.to_csv(path, index=False)
  
def show_data_info(df: pd.DataFrame):
  print("DataFrame Info:")
  print(df.info())
  print("\nMissing Values per Column:")
  print(df.isnull().sum())
  print("\nStatistical Summary:")
  print(df.describe(include='all'))
  print("\Data types:")
  print(df.dtypes)
  
    
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
  # remplace les valeurs manquantes par la moyenne pour les colonnes numériques
  for column in df.select_dtypes(include=['float64', 'int64']).columns:
    df[column].fillna(df[column].mean(), inplace=True)
  
  