import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

MODEL_DIR = "models"


def train_decision_tree(data_path):
    """
    Train Decision Tree on a preprocessed dataset.

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

    # Last column is assumed to be the target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Convert categorical columns
    X = pd.get_dummies(X)

    # Encode target if categorical
    if y.dtype == object:
        y = y.astype("category").cat.codes

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
    # Train Model
    # -----------------------------
    model = DecisionTreeClassifier(
        criterion="gini",
        random_state=42
    )

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
        "decision_tree.pkl"
    )

    joblib.dump(model, model_path)

    return {
        "Model": "Decision Tree",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Model Path": model_path
    }


if __name__ == "__main__":

    result = train_decision_tree(
        "datasets/heart_disease.csv"
    )

    print("\n" + "=" * 50)
    print("DECISION TREE RESULTS")
    print("=" * 50)

    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key:<12}: {value:.4f}")
        else:
            print(f"{key:<12}: {value}")

    print("\nDecision Tree model trained successfully.")