
"""
run.py

Complete AURORA Pipeline

1. Ask user for dataset
2. Analyze dataset
3. Save preprocessing report
4. Save preprocessed dataset
5. Approach 1 - Train all 15 models and select Model A
6. Approach 2 - Rule-based selection, train and evaluate Model B
7. Compare Model A and Model B
8. Cross-validation and SHAP on final selected model
"""

import joblib

from src.evaluation.cross_validation import perform_cross_validation
from src.explainability.shap_explainer import generate_shap
import os
from collections import Counter

import pandas as pd

from src.core.preprocessor import get_preprocessor

from src.model_selection.rule_based_selector import select_model_using_rules

from src.model_selection.hyperparameter_tuner import tune_selected_model

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
os.makedirs("models", exist_ok=True)

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

    # =====================================================
    # STEP 1 - PREPROCESSING
    # =====================================================

    processed_dataset = preprocess_dataset(dataset_path)

    # =====================================================
    # APPROACH 1 - PERFORMANCE-BASED MODEL SELECTION
    # =====================================================

    print("\n")
    print("=" * 70)
    print("APPROACH 1 - PERFORMANCE-BASED MODEL SELECTION")
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
    # MODEL COMPARISON - APPROACH 1
    # =====================================================

    print("\n")
    print("=" * 70)
    print("APPROACH 1 - MODEL COMPARISON")
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

    # Select Model A

    best = max(
        results,
        key=lambda x: x["Accuracy"]
    )

    print("\n")
    print("=" * 70)
    print("APPROACH 1 - SELECTED MODEL A")
    print("=" * 70)

    print(f"Model     : {best['Model']}")
    print(f"Accuracy  : {best['Accuracy']:.4f}")
    print(f"Precision : {best['Precision']:.4f}")
    print(f"Recall    : {best['Recall']:.4f}")
    print(f"F1 Score  : {best['F1 Score']:.4f}")
    # Tune the best model from Approach 1 using RandomizedSearchCV

    print("\n")
    print("=" * 70)
    print("APPROACH 1 - HYPERPARAMETER TUNING")
    print("=" * 70)

    tuned_model_A = tune_selected_model(
        model_name=best["Model"],
        data_path=processed_dataset,
        stage="approach_1"
    )

    print("\nAPPROACH 1 TUNED RESULTS")
    print("Model:", tuned_model_A["Model"])
    print("Best Parameters:", tuned_model_A["Best Parameters"])
    print("Accuracy:", tuned_model_A["Accuracy"])
    print("F1 Score:", tuned_model_A["F1 Score"])
    print("Model Path:", tuned_model_A["Model Path"])

        
    # =====================================================
    # STEP 3 - APPROACH 2 - RULE-BASED MODEL SELECTION
    # =====================================================

    print("\n")
    print("=" * 70)
    print("APPROACH 2 - RULE-BASED MODEL SELECTION")
    print("=" * 70)

    rule_df = pd.read_csv(processed_dataset)

    rule_df = rule_df.dropna(how="all")
    rule_df = rule_df.dropna(subset=[rule_df.columns[-1]])

    X = rule_df.iloc[:, :-1]
    y = rule_df.iloc[:, -1]

    model_b_result = select_model_using_rules(
        X,
        y,
        task_type="classification"
    )

    model_b = model_b_result["selected_model"]

    model_b_training_result = None

    if model_b is not None:

        print(f"\nSelected Model B: {model_b}")

        print("\nReasons:")
        for reason in model_b_result["reasons"]:
            print(f"- {reason}")

        # =================================================
        # TRAIN THE SELECTED MODEL B
        # =================================================

        print("\n")
        print("=" * 70)
        print("TRAINING APPROACH 2 SELECTED MODEL")
        print("=" * 70)

        model_trainers = {
            "Decision Tree": train_decision_tree,
            "Random Forest": train_random_forest,
            "Naive Bayes": train_naive_bayes,
            "Logistic Regression": train_logistic_regression,
            "KNN": train_knn,
            "SVM": train_svm,
            "LDA": train_linear_discriminant_analysis,
            "Linear Discriminant Analysis": train_linear_discriminant_analysis,
            "QDA": train_quadratic_discriminant_analysis,
            "Quadratic Discriminant Analysis": train_quadratic_discriminant_analysis,
            "AdaBoost": train_adaboost,
            "Gradient Boosting": train_gradient_boosting,
            "HistGradientBoosting": train_hist_gradient_boosting,
            "Hist Gradient Boosting": train_hist_gradient_boosting,
            "XGBoost": train_xgboost,
            "CatBoost": train_catboost,
            "LightGBM": train_lightgbm,
            "MLP": train_mlp,
            "Multi Layer Perceptron": train_mlp
        }

        selected_trainer = model_trainers.get(model_b)

        if selected_trainer is None:
            raise ValueError(
                f"No training function mapped for recommended model: {model_b}"
            )

        print(f"\nTraining recommended model: {model_b}")

        model_b_training_result = selected_trainer(processed_dataset)
        print(f"\nTraining recommended model: {model_b}")

        model_b_training_result = selected_trainer(processed_dataset)

        # Display original Model B training results
        print("\nMODEL B TRAINING RESULTS")
        print_result(model_b_training_result)
        # Tune Model B using RandomizedSearchCV

        print("\n")
        print("=" * 70)
        print("APPROACH 2 - HYPERPARAMETER TUNING")
        print("=" * 70)

        tuned_model_B = tune_selected_model(
            model_name=model_b_training_result["Model"],
            data_path=processed_dataset,
            stage="approach_2"
        )

        print("\nAPPROACH 2 TUNED RESULTS")
        print("Model:", tuned_model_B["Model"])
        print("Best Parameters:", tuned_model_B["Best Parameters"])
        print("Accuracy:", tuned_model_B["Accuracy"])
        print("F1 Score:", tuned_model_B["F1 Score"])
        print("Model Path:", tuned_model_B["Model Path"])

        # Tune Model B using RandomizedSearchCV
        print("\n")
        print("=" * 70)
        print("APPROACH 2 - HYPERPARAMETER TUNING")
        print("=" * 70)

        tuned_model_B = tune_selected_model(
            model_name=model_b_training_result["Model"],
            data_path=processed_dataset
        )

        print("\nApproach 2 Tuned Results:")
        print("Model:", tuned_model_B["Model"])
        print("Best Parameters:", tuned_model_B["Best Parameters"])
        print("Accuracy:", tuned_model_B["Accuracy"])
        print("F1 Score:", tuned_model_B["F1 Score"])
        print("Model Path:", tuned_model_B["Model Path"])
    else:
        print("\nNo model was recommended by the evaluation rules.")
        
    # =====================================================
    # STEP 4 - FINAL COMPARISON
    # =====================================================

    print("\n")
    print("=" * 70)
    print("FINAL MODEL COMPARISON - MODEL A vs MODEL B")
    print("=" * 70)

    print("\nMODEL A - PERFORMANCE-BASED")
    print_result(best)

    if model_b_training_result is not None:

        print("\nMODEL B - RULE-BASED AND TRAINED")
        print_result(model_b_training_result)

        final_best = max(
            [best, model_b_training_result],
            key=lambda x: x["Accuracy"]
        )

        print("\n")
        print("=" * 70)
        print("FINAL SELECTED MODEL")
        print("=" * 70)

        if final_best is best:
            print("Selected through Approach 1.")
        else:
            print("Selected through Approach 2.")

    else:
        final_best = best

        print("\nApproach 2 did not produce a trained model.")
        print("Selecting Model A as the final model.")

    print(f"\nFinal Model     : {final_best['Model']}")
    print(f"Accuracy        : {final_best['Accuracy']:.4f}")
    print(f"Precision       : {final_best['Precision']:.4f}")
    print(f"Recall          : {final_best['Recall']:.4f}")
    print(f"F1 Score        : {final_best['F1 Score']:.4f}")
    print(f"Saved At        : {final_best['Model Path']}")

    # =====================================================
    # STEP 5 - CROSS VALIDATION
    # =====================================================

    print("\n")
    print("=" * 70)
    print("CROSS VALIDATION - FINAL SELECTED MODEL")
    print("=" * 70)

    final_model = joblib.load(final_best["Model Path"])

    perform_cross_validation(
        final_model,
        processed_dataset
    )

    # =====================================================
    # EXPLAINABLE AI - SHAP
    # =====================================================

    print("\n")
    print("=" * 70)
    print("EXPLAINABLE AI (SHAP)")
    print("=" * 70)

    generate_shap(
        final_best["Model Path"],
        processed_dataset
    )

    print("\nAURORA Pipeline completed successfully.")


if __name__ == "__main__":
    main()
