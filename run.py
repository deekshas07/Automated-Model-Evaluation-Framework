"""
run.py

Complete AURORA Pipeline

1. Ask user for dataset
2. Analyze dataset
3. Save preprocessing report
4. Save preprocessed dataset
5. Train all ML models
6. Compare results
7. Display best model
"""
import joblib

from src.evaluation.cross_validation import perform_cross_validation
from src.explainability.shap_explainer import generate_shap
import os
from collections import Counter

import pandas as pd

from src.core.preprocessor import get_preprocessor

# ==========================
# Import All Models
# ==========================

from src.training.train_decision_tree import train_decision_tree
from src.training.train_naive_bayes import train_naive_bayes
from src.training.train_random_forest import train_random_forest
from src.training.train_xgboost import train_xgboost
from src.training.train_catboost import train_catboost
from src.training.train_adaboost import train_adaboost
from src.training.train_gradient_boosting import train_gradient_boosting
from src.training.train_hist_gradient_boosting import train_hist_gradient_boosting
from src.training.train_lightgbm import train_lightgbm
from src.training.train_logistic_regression import train_logistic_regression
from src.training.train_knn import train_knn
from src.training.train_svm import train_svm
from src.training.train_linear_discriminant_analysis import (
    train_linear_discriminant_analysis,
)
from src.training.train_quadratic_discriminant_analysis import (
    train_quadratic_discriminant_analysis,
)
from src.training.train_mlp import train_mlp

PREPROCESSED_FOLDER = "preprocessed"
REPORT_FOLDER = "reports"

os.makedirs(PREPROCESSED_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)


def preprocess_dataset(dataset_path):
    """
    Analyze and preprocess dataset using AURORA.
    """

    print("\nLoading dataset...")

    df = pd.read_csv(dataset_path)

    print("\nDataset Loaded Successfully")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    preprocessor = get_preprocessor()

    processed_df = df.copy()

    action_counter = Counter()

    report = []

    print("\nAnalyzing columns...")

    # Ignore last column (Target)
    for column in df.columns[:-1]:

        result = preprocessor.preprocess_column(
            df[column],
            column_name=column
        )

        action_counter[result.action.value] += 1

        report.append({
            "Column": column,
            "Recommended Action": result.action.value,
            "Confidence": round(result.confidence, 2),
            "Source": result.source
        })

        # Future:
        # Apply preprocessing action here.

    report_df = pd.DataFrame(report)

    report_path = os.path.join(
        REPORT_FOLDER,
        "preprocessing_report.csv"
    )

    report_df.to_csv(
        report_path,
        index=False
    )

    output_path = os.path.join(
        PREPROCESSED_FOLDER,
        "preprocessed_dataset.csv"
    )

    processed_df.to_csv(
        output_path,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)

    for action, count in sorted(action_counter.items()):

        action_name = action.replace("_", " ").title()

        print(f"{action_name:<35}: {count}")

    print("\nDetailed preprocessing report saved at:")
    print(report_path)

    print("\nPreprocessed dataset saved at:")
    print(output_path)

    return output_path


def print_result(result):

    print(
        f"{result['Model']:<40}"
        f"{result['Accuracy']:<12.4f}"
        f"{result['Precision']:<12.4f}"
        f"{result['Recall']:<12.4f}"
        f"{result['F1 Score']:<12.4f}"
    )


def main():

    print("=" * 70)
    print("AURORA PREPROCESSING + MODEL TRAINING")
    print("=" * 70)

    dataset_path = input("\nEnter dataset path : ").strip()

    if not os.path.exists(dataset_path):
        print("\nDataset not found.")
        return

    processed_dataset = preprocess_dataset(dataset_path)

    print("\n")
    print("=" * 70)
    print("TRAINING MODELS")
    print("=" * 70)

    results = []

    print("\nTraining Decision Tree...")
    results.append(train_decision_tree(processed_dataset))

    print("Training Naive Bayes...")
    results.append(train_naive_bayes(processed_dataset))

    print("Training Random Forest...")
    results.append(train_random_forest(processed_dataset))

    print("Training XGBoost...")
    results.append(train_xgboost(processed_dataset))

    print("Training CatBoost...")
    results.append(train_catboost(processed_dataset))

    print("Training AdaBoost...")
    results.append(train_adaboost(processed_dataset))
    print("Training Gradient Boosting...")
    results.append(train_gradient_boosting(processed_dataset))

    print("Training HistGradientBoosting...")
    results.append(train_hist_gradient_boosting(processed_dataset))

    print("Training LightGBM...")
    results.append(train_lightgbm(processed_dataset))

    print("Training Logistic Regression...")
    results.append(train_logistic_regression(processed_dataset))

    print("Training KNN...")
    results.append(train_knn(processed_dataset))

    print("Training SVM...")
    results.append(train_svm(processed_dataset))

    print("Training Linear Discriminant Analysis...")
    results.append(train_linear_discriminant_analysis(processed_dataset))

    print("Training Quadratic Discriminant Analysis...")
    results.append(train_quadratic_discriminant_analysis(processed_dataset))

    print("Training Multi Layer Perceptron...")
    results.append(train_mlp(processed_dataset))

    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    print("\n")
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        f"{'Model':<40}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1 Score':<12}"
    )

    for result in results:
        print_result(result)

    # =====================================================
    # BEST MODEL
    # =====================================================

    best = max(
        results,
        key=lambda x: x["Accuracy"]
    )

    print("\n")
    print("=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(f"Model      : {best['Model']}")
    print(f"Accuracy   : {best['Accuracy']:.4f}")
    print(f"Precision  : {best['Precision']:.4f}")
    print(f"Recall     : {best['Recall']:.4f}")
    print(f"F1 Score   : {best['F1 Score']:.4f}")
    print(f"Saved At   : {best['Model Path']}")
    # =====================================================
# CROSS VALIDATION
# =====================================================

    print("\n")
    print("=" * 70)
    print("CROSS VALIDATION")
    print("=" * 70)

    best_model = joblib.load(best["Model Path"])

    perform_cross_validation(
        best_model,
        processed_dataset
    )

# =====================================================
# EXPLAINABLE AI (SHAP)
# =====================================================

    print("\n")
    print("=" * 70)
    print("EXPLAINABLE AI (SHAP)")
    print("=" * 70)

    generate_shap(
        best["Model Path"],
        processed_dataset
    )
    print("\nTraining completed successfully.")
    


if __name__ == "__main__":
    main()
