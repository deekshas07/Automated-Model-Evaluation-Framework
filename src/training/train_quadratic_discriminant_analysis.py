import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

MODEL_DIR = "models"


def train_quadratic_discriminant_analysis(data_path):
    """
    Train a Quadratic Discriminant Analysis model.
    """

    os.makedirs(MODEL_DIR, exist_ok=True)

    # -----------------------------
    # Load Dataset
    # -----------------------------
    df = pd.read_csv(data_path)

    df = df.dropna(how="all")
    df = df.dropna(subset=[df.columns[-1]])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # One-hot encoding
    X = pd.get_dummies(X)

    # Fill missing values
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)

    # Encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    try:
        # QDA model
        model = QuadraticDiscriminantAnalysis(reg_param=0.99)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

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

        model_path = os.path.join(
            MODEL_DIR,
            "quadratic_discriminant_analysis.pkl"
        )

        encoder_path = os.path.join(
            MODEL_DIR,
            "qda_label_encoder.pkl"
        )

        joblib.dump(model, model_path)
        joblib.dump(encoder, encoder_path)

        return {
            "Model": "Quadratic Discriminant Analysis",
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Model Path": model_path
        }

    except Exception as e:

        print("\nQDA could not be trained.")
        print("Reason:", e)

        return {
            "Model": "Quadratic Discriminant Analysis",
            "Accuracy": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "F1 Score": 0.0,
            "Model Path": "Not Saved"
        }


if __name__ == "__main__":

    result = train_quadratic_discriminant_analysis(
        "datasets/heart_disease.csv"
    )

    print("\n" + "=" * 50)
    print("QUADRATIC DISCRIMINANT ANALYSIS RESULTS")
    print("=" * 50)

    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key:<12}: {value:.4f}")
        else:
            print(f"{key:<12}: {value}")