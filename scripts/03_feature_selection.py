import pandas as pd
from src.data_processing import load_dataset, clean_data
from src.feature_engineering import create_features
from src.feature_engineering import (
    correlation_importance,
    tree_importance,
    permutation_importance_score,
    select_top_features
)
from sklearn.ensemble import RandomForestRegressor

OUTPUT_PATH = "data/processed/features_selected.txt"
TARGET = "prix"


def main():

    print("📌 PHASE 3 : FEATURE SELECTION\n")

    # 1️⃣ Charger dataset brut
    df = load_dataset()
    print("Données chargées :", df.shape)

    # 2️⃣ Nettoyage
    df = clean_data(df)

    # 3️⃣ Feature Engineering
    df = create_features(df)

    # 4️⃣ Séparer X et y
    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    # 5️⃣ Corrélation
    corr_imp = correlation_importance(df, TARGET)

    # 6️⃣ Tree-based importance
    tree_imp = tree_importance(X, y)

    # 7️⃣ Permutation Importance
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X, y)
    perm_imp = permutation_importance_score(rf, X, y)

    # 8️⃣ Fusion + sélection
    top_features, df_scores = select_top_features(tree_imp, corr_imp, perm_imp, top_n=15)

    print("\n🎯 TOP 15 FEATURES SÉLECTIONNÉES :")
    print(top_features)

    print("\n📊 Importance fusionnée :")
    print(df_scores.head(15))

    # 9️⃣ Sauvegarde
    with open(OUTPUT_PATH, "w") as f:
        for feat in top_features:
            f.write(feat + "\n")

    print(f"\n📁 Features sauvegardées dans : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
