import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score

from src.data_processing import load_dataset, clean_data
from src.feature_engineering import create_features

TARGET = "prix"
FEATURES_PATH = "data/processed/features_selected.txt"
MODEL_SAVE_PATH = "models/best_model.pkl"
METRICS_PATH = "results/metrics/model_results.csv"


def load_selected_features():
    with open(FEATURES_PATH, "r") as f:
        return [line.strip() for line in f.readlines()]


def evaluate_model(model, X, y):
    scoring = {
        "rmse": make_scorer(mean_squared_error, squared=False),
        "mae": make_scorer(mean_absolute_error),
        "r2": make_scorer(r2_score),
    }

    scores = cross_validate(model, X, y, cv=5, scoring=scoring, n_jobs=-1, return_train_score=False)
    return {
        "RMSE": scores["test_rmse"].mean(),
        "MAE": scores["test_mae"].mean(),
        "R2": scores["test_r2"].mean(),
    }


def main():

    print("📌 PHASE 4 : MODELISATION\n")

    # 1️⃣ Charger et nettoyer les données
    df = load_dataset()
    df = clean_data(df)
    df = create_features(df)

    # 2️⃣ Charger les features sélectionnées
    top_features = load_selected_features()
    print("Features chargées :", top_features)

    X = df[top_features]
    y = df[TARGET]

    # 3️⃣ Modèles
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    results = {}

    # 4️⃣ Évaluation
    for name, model in models.items():
        print(f"\n⏳ Entraînement de {name}...")
        metrics = evaluate_model(model, X, y)
        results[name] = metrics
        print(f"✔ Résultats {name} :", metrics)

    # Convertir en DataFrame
    results_df = pd.DataFrame(results).T
    results_df.to_csv(METRICS_PATH, index=True)

    # 5️⃣ Sélection du meilleur modèle
    best_model_name = results_df["RMSE"].idxmin()
    best_model = models[best_model_name]

    # Réentraîner sur tout le dataset
    best_model.fit(X, y)
    joblib.dump(best_model, MODEL_SAVE_PATH)

    print(f"\n🏆 Meilleur modèle : {best_model_name}")
    print(f"📁 Sauvegardé dans : {MODEL_SAVE_PATH}")
    print(f"📈 Résultats détaillés dans : {METRICS_PATH}")


if __name__ == "__main__":
    main()
