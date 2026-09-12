import os

import joblib

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from catboost import CatBoostClassifier

# Folder to save trained models
MODEL_DIR = "models"


def train_catboost(data_path):
    """
    Train a CatBoost model on a preprocessed dataset.

    Parameters
    ----------
    data_path : str
        Path to the preprocessed dataset.

    Returns
    -------
    dict
        Evaluation metrics.
    """

    os.makedirs(MODEL_DIR, exist_ok=True)

    # -----------------------------
    # Load Dataset
    # -----------------------------
    df = pd.read_csv(data_path)

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Remove rows with missing target
    df = df.dropna(subset=[df.columns[-1]])

    # -----------------------------
    # Features and Target
    # -----------------------------
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # -----------------------------
    # Encode Features
    # -----------------------------
    X = pd.get_dummies(X)

    # Fill missing values
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)

    # Encode target labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    num_classes = len(set(y))

    # -----------------------------
    # Train/Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # Create Model
    # -----------------------------
    if num_classes == 2:

        model = CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            loss_function="Logloss",
            verbose=0,
            random_seed=42
        )

    else:

        model = CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            loss_function="MultiClass",
            verbose=0,
            random_seed=42
        )

    # -----------------------------
    # Train Model
    # -----------------------------
    model.fit(X_train, y_train)

    # -----------------------------
    # Prediction
    # -----------------------------
    predictions = model.predict(X_test)

    if len(predictions.shape) > 1:
        predictions = predictions.flatten()

    predictions = predictions.astype(int)

    # -----------------------------
    # Evaluation
    # -----------------------------
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    # -----------------------------
    # Save Model
    # -----------------------------
    model_path = os.path.join(
        MODEL_DIR,
        "catboost.pkl"
    )

    encoder_path = os.path.join(
        MODEL_DIR,
        "label_encoder.pkl"
    )

    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)

    # -----------------------------
    # Return Results
    # -----------------------------
    return {
        "Model": "CatBoost",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Model Path": model_path
    }


if __name__ == "__main__":

    result = train_catboost(
        "datasets/heart_disease.csv"
    )

    print("\n" + "=" * 50)
    print("CATBOOST RESULTS")
    print("=" * 50)

    for key, value in result.items():

        if isinstance(value, float):
            print(f"{key:<12}: {value:.4f}")
        else:
            print(f"{key:<12}: {value}")

    print("\nCatBoost model trained successfully.")