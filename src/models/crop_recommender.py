"""
Crop recommendation model.

Trains a RandomForestClassifier on the Kaggle "Crop Recommendation Dataset"
(columns: N, P, K, temperature, humidity, ph, rainfall, label) and exposes
a simple predict() API returning the top-k most suitable crops.

Expected CSV at: data/datasets/crop_recommendation.csv
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import config

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
LABEL_COLUMN = "label"
MODEL_PATH = config.MODELS_DIR / "crop_recommender.pkl"


def train(csv_path: Path | None = None) -> RandomForestClassifier:
    csv_path = csv_path or (config.DATASETS_DIR / "crop_recommendation.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Download the Kaggle crop recommendation "
            "dataset and place it there."
        )

    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Validation accuracy: {acc:.3f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    return model


def load_model() -> RandomForestClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No trained model at {MODEL_PATH}. Run train() first.")
    return joblib.load(MODEL_PATH)


def predict_top_k(soil_inputs: dict, k: int = 3) -> list[tuple[str, float]]:
    """
    soil_inputs: dict with keys N, P, K, temperature, humidity, ph, rainfall
    Returns: list of (crop_name, probability) sorted descending, length k.
    """
    model = load_model()
    row = pd.DataFrame([soil_inputs])[FEATURE_COLUMNS]
    probs = model.predict_proba(row)[0]
    classes = model.classes_
    ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    return ranked[:k]


if __name__ == "__main__":
    train()
