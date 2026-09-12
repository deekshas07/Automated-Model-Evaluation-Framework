import os

import joblib

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Folder to save trained models
MODEL_DIR = "models"


def train_linear_regression(data_path):
    """
    Train a Linear Regression model on a preprocessed dataset.

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

    # -----------------------------
    # Train/Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # -----------------------------
    # Create Model
    # -----------------------------
    model = LinearRegression()

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
    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = mse ** 0.5

    r2 = r2_score(y_test, predictions)

    # -----------------------------
    # Save Model
    # -----------------------------
    model_path = os.path.join(
        MODEL_DIR,
        "linear_regression.pkl"
    )

    joblib.dump(model, model_path)

    # -----------------------------
    # Return Results
    # -----------------------------
    return {
        "Model": "Linear Regression",
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2,
        "Model Path": model_path
    }


if __name__ == "__main__":

    result = train_linear_regression(
        "datasets/california_housing.csv"
    )

    print("\n" + "=" * 50)
    print("LINEAR REGRESSION RESULTS")
    print("=" * 50)

    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key:<12}: {value:.4f}")
        else:
            print(f"{key:<12}: {value}")

    print("\nLinear Regression model trained successfully.")