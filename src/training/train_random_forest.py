import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Folder to save trained models
MODEL_DIR = "models"


def train_random_forest(data_path):
    """
    Train a Random Forest model on a preprocessed dataset.

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

    # -----------------------------
    # Features and Target
    # -----------------------------
    # Assume the last column is the target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # -----------------------------
    # Encode Categorical Features
    # -----------------------------
    X = pd.get_dummies(X)

    # Encode target labels if categorical
    if y.dtype == object:
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

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
    # Create Random Forest Model
    # -----------------------------
    model = RandomForestClassifier(
        n_estimators=200,
        criterion="gini",
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    # -----------------------------
    # Train Model
    # -----------------------------
    model.fit(X_train, y_train)

    # -----------------------------
    # Prediction
    # -----------------------------
    predictions = model.predict(X_test)

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
        "random_forest.pkl"
    )

    joblib.dump(model, model_path)

    # -----------------------------
    # Return Results
    # -----------------------------
    return {
        "Model": "Random Forest",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Model Path": model_path
    }


if __name__ == "__main__":

    result = train_random_forest(
        "datasets/heart_disease.csv"
    )

    print("\n" + "=" * 50)
    print("RANDOM FOREST RESULTS")
    print("=" * 50)

    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key:<12}: {value:.4f}")
        else:
            print(f"{key:<12}: {value}")

    print("\nRandom Forest model trained successfully.")