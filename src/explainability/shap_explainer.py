import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt


def generate_shap(model_path, dataset_path):
    """
    Generate SHAP summary plot for the trained model.

    Parameters
    ----------
    model_path : str
        Path to the trained model (.pkl)

    dataset_path : str
        Path to the preprocessed dataset
    """

    # -----------------------------------
    # Load trained model
    # -----------------------------------
    model = joblib.load(model_path)

    # -----------------------------------
    # Load dataset
    # -----------------------------------
    df = pd.read_csv(dataset_path)

    # Features only
    X = df.iloc[:, :-1]

    # Apply same preprocessing used during training
    X = pd.get_dummies(X)

    # Fill missing values
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)

    # -----------------------------------
    # Use a small sample (faster)
    # -----------------------------------
    X_sample = X.sample(
        min(100, len(X)),
        random_state=42
    )

    # -----------------------------------
    # SHAP Explainer
    # (works with AdaBoost)
    # -----------------------------------
    explainer = shap.Explainer(model.predict, X_sample)

    shap_values = explainer(X_sample)

    # -----------------------------------
    # Save SHAP Summary Plot
    # -----------------------------------
    os.makedirs("reports", exist_ok=True)

    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        shap_values,
        X_sample,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        "reports/shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\n" + "=" * 60)
    print("EXPLAINABLE AI (SHAP)")
    print("=" * 60)
    print("SHAP Summary Plot saved at:")
    print("reports/shap_summary.png")