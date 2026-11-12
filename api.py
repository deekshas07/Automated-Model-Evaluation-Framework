"""
api.py

FastAPI Backend Server for AURORA — Intelligent Model Recommendation System.
Connects the web application directly to the existing AURORA backend functions
in run.py and src/ without modifying or replacing any existing ML or rule logic.
"""

import os
import sys
import json
import time
import uuid
import shutil
import traceback
import threading
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional

# Configure writable Matplotlib cache dir and non-GUI backend before importing matplotlib/shap
os.makedirs(".matplotlib_cache", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# =====================================================================
# Import Existing AURORA Backend Functions
# =====================================================================
from src.core.preprocessor import get_preprocessor
from src.core.column_profiler import ColumnProfiler
from src.core.robust_parser import RobustCSVParser
from src.utils.target_detector import get_target_detector
from src.utils.universal_type_detector import get_type_detector

from src.model_selection.rule_based_selector import select_model_using_rules
from src.model_selection import model_evaluation_rules
from src.model_selection.hyperparameter_tuner import tune_selected_model

from src.evaluation.cross_validation import perform_cross_validation

# Import all 15 classification trainers (as in run.py) + 1 regression trainer
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
from src.training.train_linear_regression import train_linear_regression

# Ensure required folders exist
PREPROCESSED_FOLDER = "preprocessed"
REPORT_FOLDER = "reports"
MODEL_FOLDER = "models"
UPLOAD_FOLDER = "uploads"

for folder in [PREPROCESSED_FOLDER, REPORT_FOLDER, MODEL_FOLDER, UPLOAD_FOLDER]:
    os.makedirs(folder, exist_ok=True)

CURRENT_DATASET_META_PATH = os.path.join(REPORT_FOLDER, "current_dataset_meta.json")
ANALYSIS_HISTORY_PATH = os.path.join(REPORT_FOLDER, "analysis_history.json")

# =====================================================================
# Candidate Model Registries (Matching run.py & src/training/)
# =====================================================================

CLASSIFICATION_CANDIDATE_REGISTRY = [
    ("Decision Tree", train_decision_tree),
    ("Naive Bayes", train_naive_bayes),
    ("Random Forest", train_random_forest),
    ("XGBoost", train_xgboost),
    ("CatBoost", train_catboost),
    ("AdaBoost", train_adaboost),
    ("Gradient Boosting", train_gradient_boosting),
    ("HistGradientBoosting", train_hist_gradient_boosting),
    ("LightGBM", train_lightgbm),
    ("Logistic Regression", train_logistic_regression),
    ("KNN", train_knn),
    ("SVM", train_svm),
    ("Linear Discriminant Analysis", train_linear_discriminant_analysis),
    ("Quadratic Discriminant Analysis", train_quadratic_discriminant_analysis),
    ("Multi Layer Perceptron", train_mlp),
]

REGRESSION_CANDIDATE_REGISTRY = [
    ("Linear Regression", train_linear_regression),
]

# Trainer lookup map (matching run.py lines 336-356 + regression trainer)
MODEL_TRAINERS_MAP = {
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
    "Multi Layer Perceptron": train_mlp,
    "Linear Regression": train_linear_regression,
}

# Hyperparameter tuning parameter search space metadata (from src/model_selection/hyperparameter_tuner.py)
TUNING_SEARCH_SPACES: Dict[str, Dict[str, Any]] = {
    "Decision Tree": {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300, 500],
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    },
    "CatBoost": {
        "iterations": [100, 200, 300, 500],
        "depth": [4, 5, 6, 7, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "l2_leaf_reg": [1, 3, 5, 7, 9],
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 150, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        "estimator__max_depth": [1, 2, 3, 4],
        "estimator__min_samples_split": [2, 5, 10],
        "estimator__min_samples_leaf": [1, 2, 4],
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 150, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4, 5],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "max_features": ["None", "sqrt", "log2"],
    },
    "HistGradientBoosting": {
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_iter": [100, 150, 200, 300],
        "max_depth": [None, 3, 5, 6, 8, 10],
        "min_samples_leaf": [10, 20, 30, 50],
        "l2_regularization": [0.0, 0.1, 0.5, 1.0],
        "max_bins": [100, 150, 200, 255],
    },
    "LightGBM": {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [-1, 3, 5, 6, 8, 10],
        "num_leaves": [15, 31, 50, 70],
        "min_child_samples": [10, 20, 30, 50],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    },
    "Logistic Regression": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs"],
        "class_weight": ["None", "balanced"],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
        "p": [1, 2],
    },
    "SVM": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"],
        "degree": [2, 3, 4],
    },
    "Linear Discriminant Analysis": {
        "solver": ["svd", "lsqr", "eigen"],
        "shrinkage": ["None", "auto", 0.1, 0.5, 0.9],
    },
    "LDA": {
        "solver": ["svd", "lsqr", "eigen"],
        "shrinkage": ["None", "auto", 0.1, 0.5, 0.9],
    },
    "Quadratic Discriminant Analysis": {
        "reg_param": [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
    },
    "QDA": {
        "reg_param": [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
    },
    "Multi Layer Perceptron": {
        "hidden_layer_sizes": ["(50,)", "(100,)", "(100, 50)", "(100, 50, 25)"],
        "activation": ["relu", "tanh"],
        "solver": ["adam"],
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "learning_rate": ["constant", "adaptive"],
        "learning_rate_init": [0.001, 0.01],
    },
    "MLP": {
        "hidden_layer_sizes": ["(50,)", "(100,)", "(100, 50)", "(100, 50, 25)"],
        "activation": ["relu", "tanh"],
        "solver": ["adam"],
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "learning_rate": ["constant", "adaptive"],
        "learning_rate_init": [0.001, 0.01],
    },
    "Naive Bayes": {
        "var_smoothing": [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
    },
}

# =====================================================================
# In-Memory & Disk-Backed State Store
# =====================================================================

STATE: Dict[str, Any] = {
    "current_dataset": None,
    "jobs": {},
    "latest_analysis": None,
    "analysis_history": [],
}


def _safe_json_value(val: Any) -> Any:
    """Recursively sanitize numpy/pandas values for JSON serialization."""
    if val is None:
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    if isinstance(val, (str,)):
        return val
    if isinstance(val, dict):
        return {str(k): _safe_json_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set, np.ndarray, pd.Series)):
        return [_safe_json_value(x) for x in val]
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    return str(val)


def _load_persisted_state():
    """Load persisted dataset metadata and analysis history if present."""
    if os.path.exists(CURRENT_DATASET_META_PATH):
        try:
            with open(CURRENT_DATASET_META_PATH, "r", encoding="utf-8") as f:
                STATE["current_dataset"] = json.load(f)
        except Exception:
            pass

    if os.path.exists(ANALYSIS_HISTORY_PATH):
        try:
            with open(ANALYSIS_HISTORY_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
                if isinstance(history, list):
                    STATE["analysis_history"] = history
                    if history:
                        STATE["latest_analysis"] = history[0]
        except Exception:
            pass


def _save_current_dataset_meta(meta: Dict[str, Any]):
    clean_meta = _safe_json_value(meta)
    STATE["current_dataset"] = clean_meta
    try:
        with open(CURRENT_DATASET_META_PATH, "w", encoding="utf-8") as f:
            json.dump(clean_meta, f, indent=2)
    except Exception:
        pass


def _save_analysis_to_history(analysis_result: Dict[str, Any]):
    clean_res = _safe_json_value(analysis_result)
    STATE["latest_analysis"] = clean_res
    # Replace if same analysis_id exists, else prepend
    existing_idx = next(
        (
            i
            for i, item in enumerate(STATE["analysis_history"])
            if item.get("analysis_id") == clean_res.get("analysis_id")
        ),
        None,
    )
    if existing_idx is not None:
        STATE["analysis_history"][existing_idx] = clean_res
    else:
        STATE["analysis_history"].insert(0, clean_res)
        STATE["analysis_history"] = STATE["analysis_history"][:25]

    try:
        with open(ANALYSIS_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(STATE["analysis_history"], f, indent=2)
    except Exception:
        pass


_load_persisted_state()


# =====================================================================
# Helper Functions: Task Detection, Preprocessing, Overview Generation
# =====================================================================

def detect_dataset_task_type(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identify whether a dataset represents a classification or regression problem
    by inspecting the target column (last column, per AURORA convention).
    """
    target_col = df.columns[-1]
    y_raw = df[target_col].dropna()
    n_samples = len(y_raw)
    if n_samples == 0:
        return {
            "task_type": "classification",
            "target_dtype": str(df[target_col].dtype),
            "unique_values": 0,
            "unique_ratio": 0.0,
            "reason": "Empty target series; defaulting to classification.",
        }

    # Check if values are strings like '?' that represent missing values
    if y_raw.dtype == object or pd.api.types.is_string_dtype(y_raw):
        cleaned = y_raw[~y_raw.astype(str).str.strip().isin(["?", "NA", "N/A", "null", "None", ""])]
        if len(cleaned) > 0:
            y_raw = cleaned
            n_samples = len(y_raw)

    is_bool = pd.api.types.is_bool_dtype(y_raw)
    is_numeric = pd.api.types.is_numeric_dtype(y_raw) and not is_bool

    # If object/string, check if >90% convert to numbers
    if not is_numeric and not is_bool:
        converted = pd.to_numeric(y_raw, errors="coerce")
        if converted.notna().mean() >= 0.90:
            y_raw = converted.dropna()
            n_samples = len(y_raw)
            is_numeric = True

    unique_count = int(y_raw.nunique())
    unique_ratio = float(unique_count / max(n_samples, 1))

    if is_bool:
        task_type = "classification"
        reason = f"Boolean target column '{target_col}' with {unique_count} classes."
    elif not is_numeric:
        task_type = "classification"
        reason = f"Categorical target column '{target_col}' with {unique_count} distinct classes."
    else:
        # Numeric target
        is_float = pd.api.types.is_float_dtype(y_raw) and not np.all(np.mod(y_raw, 1) == 0)
        if unique_count <= 15:
            task_type = "classification"
            reason = f"Discrete target column '{target_col}' with {unique_count} target classes."
        elif is_float and unique_count > 15:
            task_type = "regression"
            reason = f"Continuous floating-point target '{target_col}' with {unique_count} unique values."
        elif unique_count > 20 and unique_ratio >= 0.05:
            task_type = "regression"
            reason = f"Continuous/high-cardinality numeric target '{target_col}' ({unique_count} unique values across {n_samples} rows)."
        else:
            task_type = "classification"
            reason = f"Discrete numeric target '{target_col}' with {unique_count} classes."

    details: Dict[str, Any] = {
        "task_type": task_type,
        "target_column": str(target_col),
        "target_dtype": str(df[target_col].dtype),
        "unique_values": unique_count,
        "unique_ratio": round(unique_ratio, 4),
        "reason": reason,
    }

    if task_type == "classification":
        vc = y_raw.astype(str).value_counts().head(15)
        details["class_distribution"] = [
            {
                "label": str(k),
                "count": int(v),
                "percentage": round(float(v / max(n_samples, 1) * 100), 2),
            }
            for k, v in vc.items()
        ]
    else:
        y_num = pd.to_numeric(y_raw, errors="coerce").dropna()
        if len(y_num) > 0:
            details["target_statistics"] = {
                "min": round(float(y_num.min()), 4),
                "max": round(float(y_num.max()), 4),
                "mean": round(float(y_num.mean()), 4),
                "median": round(float(y_num.median()), 4),
                "std": round(float(y_num.std()) if len(y_num) > 1 else 0.0, 4),
            }

    return details


def run_aurora_preprocessing(dataset_path: str, dataset_display_name: str, file_size_bytes: int) -> Dict[str, Any]:
    """
    Execute the exact AURORA preprocessing analysis workflow from run.py:
    1. Parse CSV (handling delimiter/encoding edge cases cleanly)
    2. Analyze feature columns (df.columns[:-1]) using get_preprocessor().preprocess_column()
    3. Save reports/preprocessing_report.csv
    4. Save preprocessed/preprocessed_dataset.csv
    5. Build comprehensive dataset overview for the UI
    """
    # Step 1: Parse CSV
    try:
        # Check if semicolon or tab delimiter is used
        parser = RobustCSVParser()
        parse_res = parser.parse_file(dataset_path)
        df = parse_res.dataframe
    except Exception as parse_err:
        try:
            df = pd.read_csv(dataset_path, low_memory=False)
        except Exception as fallback_err:
            raise ValueError(f"Could not parse CSV file: {parse_err or fallback_err}")

    # Replace common missing-value placeholders and infinite values with NaN
    df = df.replace(
        ["?", " ? ", "? ", "NA", "N/A", "null", "NULL", "None", "none", "nan", "NaN", ""],
        np.nan,
    )
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(how="all")

    # Drop completely empty columns if any exist
    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        df = df.drop(columns=empty_cols)

    if df.shape[1] < 2:
        raise ValueError("Dataset must contain at least 1 input feature column and 1 target column (minimum 2 columns).")
    if df.shape[0] < 10:
        raise ValueError("Dataset must contain at least 10 valid rows for training and evaluation.")

    # Convert numeric-looking object columns (where >=90% of non-null values are valid numbers)
    for col in df.columns:
        if df[col].dtype == object:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                conv = pd.to_numeric(non_null, errors="coerce")
                if conv.notna().mean() >= 0.90:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure target column (last column) has no missing values
    target_col = df.columns[-1]
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    if df.shape[0] < 10:
        raise ValueError(
            f"Dataset must contain at least 10 valid rows with non-null target '{target_col}' (found {df.shape[0]})."
        )

    # Step 2: Run AURORA IntelligentPreprocessor on feature columns (matching run.py lines 77-127)
    preprocessor = get_preprocessor()
    type_detector = get_type_detector()
    profiler = ColumnProfiler(bins=10)

    processed_df = df.copy()
    action_counter = Counter()
    report_rows = []
    feature_details = []

    for idx, column in enumerate(df.columns[:-1]):
        col_series = df[column]
        result = preprocessor.preprocess_column(
            col_series,
            column_name=column,
        )
        action_val = result.action.value
        conf_val = round(float(result.confidence), 2)
        source_val = str(result.source)

        action_counter[action_val] += 1
        report_rows.append(
            {
                "Column": column,
                "Recommended Action": action_val,
                "Confidence": conf_val,
                "Source": source_val,
            }
        )

        # Semantic type & column statistics
        sem_res = type_detector.detect_semantic_type(col_series, column)
        null_count = int(col_series.isnull().sum())
        null_pct = round(float(null_count / max(len(df), 1) * 100), 2)
        unique_cnt = int(col_series.dropna().nunique())
        sample_vals = [str(v) for v in col_series.dropna().unique()[:4]]

        col_stat_summary: Dict[str, Any] = {}
        if pd.api.types.is_numeric_dtype(col_series) and not pd.api.types.is_bool_dtype(col_series):
            clean_num = col_series.replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean_num) > 0:
                raw_std = clean_num.std() if len(clean_num) > 1 else 0.0
                col_stat_summary = {
                    "mean": round(float(clean_num.mean()), 3),
                    "std": round(0.0 if pd.isna(raw_std) else float(raw_std), 3),
                    "min": round(float(clean_num.min()), 3),
                    "max": round(float(clean_num.max()), 3),
                }

        feature_details.append(
            {
                "name": str(column),
                "position": idx + 1,
                "is_target": False,
                "dtype": str(col_series.dtype),
                "semantic_type": sem_res.semantic_type.value,
                "missing_count": null_count,
                "missing_percentage": null_pct,
                "unique_count": unique_cnt,
                "sample_values": sample_vals,
                "recommended_action": action_val,
                "confidence": conf_val,
                "source": source_val,
                "explanation": getattr(result, "explanation", ""),
                "stats": col_stat_summary,
            }
        )

    # Also add Target Column entry for complete column overview
    target_series = df[target_col]
    target_sem = type_detector.detect_semantic_type(target_series, target_col)
    target_nulls = int(target_series.isnull().sum())
    feature_details.append(
        {
            "name": str(target_col),
            "position": len(df.columns),
            "is_target": True,
            "dtype": str(target_series.dtype),
            "semantic_type": target_sem.semantic_type.value,
            "missing_count": target_nulls,
            "missing_percentage": round(float(target_nulls / max(len(df), 1) * 100), 2),
            "unique_count": int(target_series.dropna().nunique()),
            "sample_values": [str(v) for v in target_series.dropna().unique()[:4]],
            "recommended_action": "target_variable",
            "confidence": 1.0,
            "source": "target_detector",
            "explanation": "Target column (last column of dataset) used as prediction label.",
            "stats": {},
        }
    )

    # Save reports/preprocessing_report.csv and preprocessed/preprocessed_dataset.csv (exact paths from run.py)
    report_df = pd.DataFrame(report_rows)
    report_path = os.path.join(REPORT_FOLDER, "preprocessing_report.csv")
    report_df.to_csv(report_path, index=False)

    output_path = os.path.join(PREPROCESSED_FOLDER, "preprocessed_dataset.csv")
    processed_df.to_csv(output_path, index=False)

    # Task detection
    task_info = detect_dataset_task_type(processed_df)

    # Missing values summary across features
    total_cells = int(processed_df.shape[0] * processed_df.shape[1])
    total_missing = int(processed_df.isnull().sum().sum())
    cols_with_missing = int((processed_df.isnull().sum() > 0).sum())

    # Data types summary across all columns
    numeric_cols = len(processed_df.select_dtypes(include=["number"]).columns)
    bool_cols = len(processed_df.select_dtypes(include=["bool"]).columns)
    cat_cols = len(processed_df.select_dtypes(include=["object", "category", "string"]).columns)

    overview = {
        "dataset_id": str(uuid.uuid4())[:8],
        "dataset_name": dataset_display_name,
        "file_size_bytes": file_size_bytes,
        "uploaded_at": datetime.now().isoformat(),
        "preprocessed_path": output_path,
        "report_path": report_path,
        "rows": int(processed_df.shape[0]),
        "columns": int(processed_df.shape[1]),
        "feature_count": int(processed_df.shape[1] - 1),
        "target_column": str(target_col),
        "task_type": task_info["task_type"],
        "task_detection": task_info,
        "missing_summary": {
            "total_cells": total_cells,
            "missing_cells": total_missing,
            "missing_percentage": round(float(total_missing / max(total_cells, 1) * 100), 2),
            "columns_with_missing": cols_with_missing,
            "is_complete": total_missing == 0,
        },
        "dtype_summary": {
            "numeric": int(numeric_cols),
            "categorical": int(cat_cols),
            "boolean": int(bool_cols),
        },
        "preprocessing_summary": [
            {
                "action": action,
                "label": action.replace("_", " ").title(),
                "count": count,
            }
            for action, count in sorted(action_counter.items())
        ],
        "preprocessing_report": report_rows,
        "features": feature_details,
        "preview_columns": [str(c) for c in processed_df.columns],
        "preview_rows": _safe_json_value(processed_df.head(20).to_dict(orient="records")),
    }

    _save_current_dataset_meta(overview)
    return overview


def _prepare_clean_training_csv_if_needed(preprocessed_path: str, task_type: str) -> str:
    """
    Some classification trainers in src/training/ (e.g. train_naive_bayes, train_decision_tree,
    train_random_forest) do not call X.fillna() internally, or fail if a class has only 1 sample
    during stratified split. We keep preprocessed/preprocessed_dataset.csv untouched for rule
    evaluation, and provide a training-safe CSV when needed.
    """
    df = pd.read_csv(preprocessed_path)
    df = df.dropna(how="all")
    target_col = df.columns[-1]
    df = df.dropna(subset=[target_col])

    needs_clean_copy = False

    # Check if there are missing values in features
    if df.iloc[:, :-1].isnull().any().any():
        needs_clean_copy = True

    # For classification, check if any class has < 2 samples (which would break stratify=y in train_test_split)
    if task_type == "classification":
        vc = df[target_col].value_counts()
        if (vc < 2).any():
            valid_classes = vc[vc >= 2].index
            df = df[df[target_col].isin(valid_classes)]
            needs_clean_copy = True

    if not needs_clean_copy:
        return preprocessed_path

    clean_df = df.copy()
    for col in clean_df.columns[:-1]:
        if pd.api.types.is_numeric_dtype(clean_df[col]) and not pd.api.types.is_bool_dtype(clean_df[col]):
            median_val = clean_df[col].median()
            clean_df[col] = clean_df[col].fillna(0 if pd.isna(median_val) else median_val)
        else:
            mode_vals = clean_df[col].mode(dropna=True)
            fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else "missing"
            clean_df[col] = clean_df[col].fillna(fill_val)

    clean_path = os.path.join(PREPROCESSED_FOLDER, "preprocessed_training_ready.csv")
    clean_df.to_csv(clean_path, index=False)
    return clean_path


# =====================================================================
# Background Analysis Execution (Approach 1, Approach 2, Combined)
# =====================================================================

def _execute_analysis_job(
    job_id: str,
    mode: str,
    task_type: str,
    selection_metric: Optional[str],
    train_model_b: bool,
):
    job = STATE["jobs"][job_id]

    def add_log(msg: str, level: str = "info"):
        job["logs"].append(
            {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "message": msg,
                "level": level,
            }
        )

    def set_stage(stage_key: str, status: str, detail: str = ""):
        for s in job["stages"]:
            if s["key"] == stage_key:
                s["status"] = status
                if detail:
                    s["detail"] = detail
                if status == "running":
                    s["started_at"] = datetime.now().isoformat()
                elif status in ("completed", "error", "skipped"):
                    s["completed_at"] = datetime.now().isoformat()
        job["current_stage"] = stage_key

    try:
        preprocessed_path = os.path.join(PREPROCESSED_FOLDER, "preprocessed_dataset.csv")
        if not os.path.exists(preprocessed_path):
            raise FileNotFoundError("No preprocessed dataset found. Please upload or select a dataset first.")

        dataset_meta = STATE.get("current_dataset") or {}
        dataset_name = dataset_meta.get("dataset_name", "preprocessed_dataset.csv")

        # -------------------------------------------------------------
        # STAGE 1: Dataset Verification & Task Identification
        # -------------------------------------------------------------
        set_stage("dataset_prep", "running", f"Loading {dataset_name} and verifying schema")
        add_log(f"Loaded preprocessed dataset: {dataset_name}")

        raw_df = pd.read_csv(preprocessed_path)
        raw_df = raw_df.dropna(how="all")
        raw_df = raw_df.dropna(subset=[raw_df.columns[-1]])

        n_rows, n_cols = raw_df.shape
        target_col = raw_df.columns[-1]
        add_log(f"Dataset verified: {n_rows} rows, {n_cols - 1} features, target='{target_col}', task='{task_type}'")

        training_data_path = _prepare_clean_training_csv_if_needed(preprocessed_path, task_type)
        set_stage("dataset_prep", "completed", f"{n_rows} rows × {n_cols - 1} features ({task_type.title()})")

        approach_1_output = None
        approach_2_output = None

        # Determine primary selection metric
        if not selection_metric:
            selection_metric = "Accuracy" if task_type == "classification" else "R2 Score"

        # -------------------------------------------------------------
        # STAGE 2: Approach 1 — Performance-Based Selection
        # -------------------------------------------------------------
        if mode in ("approach_1", "both"):
            set_stage("approach_1", "running", f"Evaluating candidate models for {task_type}")
            add_log(f"Starting Approach 1 — Performance-Based Model Selection ({task_type})")

            registry = (
                CLASSIFICATION_CANDIDATE_REGISTRY
                if task_type == "classification"
                else REGRESSION_CANDIDATE_REGISTRY
            )

            job["approach_1_progress"] = {
                "total_models": len(registry),
                "completed_models": 0,
                "current_model": None,
                "evaluated": [],
            }

            candidate_results = []
            for idx, (model_label, trainer_fn) in enumerate(registry):
                job["approach_1_progress"]["current_model"] = model_label
                set_stage(
                    "approach_1",
                    "running",
                    f"Training candidate {idx + 1}/{len(registry)}: {model_label}",
                )
                add_log(f"[Approach 1] Training {model_label} ({idx + 1}/{len(registry)})...")

                t0 = time.time()
                try:
                    res = trainer_fn(training_data_path)
                    elapsed = round(time.time() - t0, 3)
                    res_clean = _safe_json_value(res)
                    res_clean["training_time_sec"] = elapsed
                    res_clean["status"] = "success"
                    candidate_results.append(res_clean)
                    job["approach_1_progress"]["evaluated"].append(res_clean)

                    if task_type == "classification":
                        add_log(
                            f"[Approach 1] {res_clean['Model']} completed in {elapsed}s — "
                            f"Accuracy: {res_clean['Accuracy']:.4f}, F1: {res_clean['F1 Score']:.4f}"
                        )
                    else:
                        add_log(
                            f"[Approach 1] {res_clean['Model']} completed in {elapsed}s — "
                            f"R2 Score: {res_clean['R2 Score']:.4f}, RMSE: {res_clean['RMSE']:.4f}"
                        )
                except Exception as model_err:
                    elapsed = round(time.time() - t0, 3)
                    add_log(f"[Approach 1] {model_label} failed ({model_err})", level="warning")
                    err_entry = {
                        "Model": model_label,
                        "status": "failed",
                        "error": str(model_err),
                        "training_time_sec": elapsed,
                    }
                    if task_type == "classification":
                        err_entry.update(
                            {
                                "Accuracy": 0.0,
                                "Precision": 0.0,
                                "Recall": 0.0,
                                "F1 Score": 0.0,
                                "Model Path": "Not Saved",
                            }
                        )
                    else:
                        err_entry.update(
                            {
                                "MAE": 0.0,
                                "MSE": 0.0,
                                "RMSE": 0.0,
                                "R2 Score": -999.0,
                                "Model Path": "Not Saved",
                            }
                        )
                    candidate_results.append(err_entry)

                job["approach_1_progress"]["completed_models"] = idx + 1

            job["approach_1_progress"]["current_model"] = None

            valid_candidates = [
                r for r in candidate_results if r.get("status") != "failed" and r.get("Model Path") != "Not Saved"
            ]
            if not valid_candidates:
                raise RuntimeError("No candidate models succeeded during Approach 1 evaluation.")

            # Select best-performing Model A according to selection metric
            if selection_metric in ("MAE", "MSE", "RMSE"):
                model_a = min(valid_candidates, key=lambda x: float(x.get(selection_metric, float("inf"))))
                sorted_candidates = sorted(
                    valid_candidates,
                    key=lambda x: float(x.get(selection_metric, float("inf"))),
                )
            else:
                metric_key = selection_metric if selection_metric in valid_candidates[0] else (
                    "Accuracy" if task_type == "classification" else "R2 Score"
                )
                model_a = max(valid_candidates, key=lambda x: float(x.get(metric_key, float("-inf"))))
                sorted_candidates = sorted(
                    valid_candidates,
                    key=lambda x: float(x.get(metric_key, float("-inf"))),
                    reverse=True,
                )

            primary_val = model_a.get("Accuracy") if task_type == "classification" else model_a.get("R2 Score")
            add_log(
                f"[Approach 1] Selected Model A: {model_a['Model']} "
                f"({selection_metric}: {primary_val:.4f})",
                level="success",
            )
            set_stage(
                "approach_1",
                "completed",
                f"Model A: {model_a['Model']} ({selection_metric}: {primary_val:.4f})",
            )

            approach_1_output = {
                "selected_model_a": model_a,
                "selection_metric": selection_metric,
                "candidate_count": len(candidate_results),
                "candidate_models": sorted_candidates,
                "all_results": candidate_results,
            }
        else:
            set_stage("approach_1", "skipped", "Not selected for this run")

        # -------------------------------------------------------------
        # STAGE 3: Approach 2 — Rule-Based Recommendation
        # -------------------------------------------------------------
        if mode in ("approach_2", "both"):
            set_stage("approach_2", "running", f"Applying {task_type} evaluation rules")
            add_log(f"Starting Approach 2 — Rule-Based Model Recommendation ({task_type})")

            X_rule = raw_df.iloc[:, :-1]
            y_rule = raw_df.iloc[:, -1]

            # Call the exact rule-based selector from src/model_selection/rule_based_selector.py
            model_b_result = select_model_using_rules(
                X_rule,
                y_rule,
                task_type=task_type,
            )

            selected_model_b = model_b_result.get("selected_model")
            rec_score = model_b_result.get("recommendation_score", 0)
            reasons = model_b_result.get("reasons", [])
            message = model_b_result.get("message", "")

            add_log(
                f"[Approach 2] Rule Engine selected Model B: {selected_model_b} "
                f"(Recommendation Score: {rec_score}, {len(reasons)} rules triggered)",
                level="success",
            )

            # Check if trainer exists for Model B and if optional training was requested
            trainer_fn = MODEL_TRAINERS_MAP.get(selected_model_b) if selected_model_b else None
            trainer_available = trainer_fn is not None
            model_b_trained_metrics = None

            if train_model_b and trainer_fn is not None:
                add_log(f"[Approach 2] Training recommended Model B ({selected_model_b})...")
                try:
                    t0 = time.time()
                    b_res = _safe_json_value(trainer_fn(training_data_path))
                    b_res["training_time_sec"] = round(time.time() - t0, 3)
                    model_b_trained_metrics = b_res
                    add_log(f"[Approach 2] Model B ({selected_model_b}) trained and evaluated.")
                except Exception as b_err:
                    add_log(f"[Approach 2] Optional training of Model B failed: {b_err}", level="warning")

            set_stage(
                "approach_2",
                "completed",
                f"Model B: {selected_model_b or 'None'} (Rule Score: {rec_score})",
            )

            approach_2_output = {
                "selected_model_b": selected_model_b,
                "recommendation_score": float(rec_score) if rec_score is not None else 0.0,
                "reasons": reasons,
                "rules_triggered_count": len(reasons),
                "message": message,
                "trainer_available": trainer_available,
                "is_trained": model_b_trained_metrics is not None,
                "trained_metrics": model_b_trained_metrics,
                "rule_engine_type": (
                    "160 Classification Evaluation Heuristics (Rules 1–160)"
                    if task_type == "classification"
                    else "85 Regression Evaluation Heuristics (Rules R1–R85)"
                ),
            }
        else:
            set_stage("approach_2", "skipped", "Not selected for this run")

        # -------------------------------------------------------------
        # STAGE 4: Finalize Results & Summary
        # -------------------------------------------------------------
        set_stage("finalize", "running", "Compiling results and comparison summary")

        analysis_record = {
            "analysis_id": job_id,
            "completed_at": datetime.now().isoformat(),
            "dataset_name": dataset_name,
            "rows": n_rows,
            "columns": n_cols,
            "feature_count": n_cols - 1,
            "target_column": str(target_col),
            "task_type": task_type,
            "mode": mode,
            "selection_metric": selection_metric,
            "approach_1": approach_1_output,
            "approach_2": approach_2_output,
            "tuning_results": {},
            "cross_validation": None,
            "shap_available": False,
        }

        _save_analysis_to_history(analysis_record)
        set_stage("finalize", "completed", "Analysis complete")
        add_log("AURORA analysis workflow completed successfully.", level="success")

        job["result"] = _safe_json_value(analysis_record)
        job["status"] = "completed"

    except Exception as exc:
        err_msg = str(exc)
        add_log(f"Execution failed: {err_msg}", level="error")
        for s in job["stages"]:
            if s["status"] == "running":
                s["status"] = "error"
                s["detail"] = err_msg
        job["status"] = "error"
        job["error"] = err_msg
        job["traceback"] = traceback.format_exc()


# =====================================================================
# FastAPI Application Setup & Endpoints
# =====================================================================

app = FastAPI(
    title="AURORA — Intelligent Model Recommendation System API",
    description="Backend API connecting the AURORA web interface to the existing preprocessing, performance-based evaluation, rule-based recommendation, and hyperparameter tuning modules.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoadSampleRequest(BaseModel):
    filename: str


class TaskTypeUpdateRequest(BaseModel):
    task_type: str


class StartAnalysisRequest(BaseModel):
    mode: str = "both"  # "approach_1", "approach_2", or "both"
    task_type: Optional[str] = None
    selection_metric: Optional[str] = None
    train_model_b: bool = False


class TuneModelRequest(BaseModel):
    model_name: str
    stage: str = "approach_1"


class ModelEvaluationRequest(BaseModel):
    model_name: Optional[str] = None
    model_path: Optional[str] = None


@app.get("/api/health")
def health_check():
    """Return backend health status and registered model capabilities."""
    return {
        "status": "ok",
        "service": "AURORA — Intelligent Model Recommendation System",
        "classification_candidates": [name for name, _ in CLASSIFICATION_CANDIDATE_REGISTRY],
        "regression_candidates": [name for name, _ in REGRESSION_CANDIDATE_REGISTRY],
        "rule_classification_models": model_evaluation_rules.MODELS,
        "rule_regression_models": model_evaluation_rules.REGRESSION_MODELS,
        "tunable_models": list(TUNING_SEARCH_SPACES.keys()),
        "has_active_dataset": STATE["current_dataset"] is not None,
        "history_count": len(STATE["analysis_history"]),
    }


@app.get("/api/datasets/samples")
def list_sample_datasets():
    """List built-in CSV datasets available in the workspace datasets/ directory."""
    samples = []
    if os.path.exists("datasets"):
        for fname in sorted(os.listdir("datasets")):
            if not fname.lower().endswith(".csv"):
                continue
            fpath = os.path.join("datasets", fname)
            if not os.path.isfile(fpath):
                continue
            try:
                fsize = os.path.getsize(fpath)
                # Quick read to get shape & target column
                df_head = pd.read_csv(fpath, nrows=250)
                if df_head.shape[1] < 2:
                    # Try semicolon delimiter
                    df_head = pd.read_csv(fpath, sep=";", nrows=250)
                # Count total lines quickly
                with open(fpath, "rb") as bf:
                    total_rows = max(0, sum(1 for _ in bf) - 1)
                task_info = detect_dataset_task_type(df_head)
                samples.append(
                    {
                        "filename": fname,
                        "file_size_bytes": fsize,
                        "rows": total_rows,
                        "columns": int(df_head.shape[1]),
                        "target_column": str(df_head.columns[-1]),
                        "task_type": task_info["task_type"],
                    }
                )
            except Exception:
                continue
    return {"samples": samples}


@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV file from the user's computer, validate, preprocess, and return Dataset Overview."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a valid .csv dataset.")

    content = await file.read()
    file_size = len(content)
    if file_size == 0:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")

    safe_name = os.path.basename(file.filename).strip()
    upload_path = os.path.join(UPLOAD_FOLDER, safe_name)
    with open(upload_path, "wb") as f:
        f.write(content)

    try:
        overview = run_aurora_preprocessing(upload_path, safe_name, file_size)
        return _safe_json_value(overview)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV dataset: {exc}")


@app.post("/api/datasets/load-sample")
def load_sample_dataset(req: LoadSampleRequest):
    """Load a benchmark dataset from datasets/ through the exact AURORA preprocessing pipeline."""
    sample_path = os.path.join("datasets", os.path.basename(req.filename))
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail=f"Dataset '{req.filename}' not found in datasets/ directory.")

    file_size = os.path.getsize(sample_path)
    try:
        overview = run_aurora_preprocessing(sample_path, req.filename, file_size)
        return _safe_json_value(overview)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to process sample dataset: {exc}")


@app.get("/api/datasets/current")
def get_current_dataset():
    """Return the active dataset overview, or rebuild from preprocessed_dataset.csv if present."""
    if STATE["current_dataset"] is not None:
        return _safe_json_value(STATE["current_dataset"])

    preprocessed_path = os.path.join(PREPROCESSED_FOLDER, "preprocessed_dataset.csv")
    if os.path.exists(preprocessed_path):
        fsize = os.path.getsize(preprocessed_path)
        overview = run_aurora_preprocessing(preprocessed_path, "preprocessed_dataset.csv", fsize)
        return _safe_json_value(overview)

    raise HTTPException(status_code=404, detail="No active dataset loaded. Please upload a CSV dataset first.")


@app.patch("/api/datasets/task-type")
def update_task_type(req: TaskTypeUpdateRequest):
    """Allow updating the task type ('classification' or 'regression') for the active dataset."""
    task = req.task_type.lower().strip()
    if task not in ("classification", "regression"):
        raise HTTPException(status_code=400, detail="Task type must be 'classification' or 'regression'.")
    if STATE["current_dataset"] is None:
        raise HTTPException(status_code=404, detail="No active dataset loaded.")

    STATE["current_dataset"]["task_type"] = task
    if "task_detection" in STATE["current_dataset"]:
        STATE["current_dataset"]["task_detection"]["task_type"] = task
    _save_current_dataset_meta(STATE["current_dataset"])
    return _safe_json_value(STATE["current_dataset"])


@app.post("/api/analysis/start")
def start_analysis(req: StartAnalysisRequest):
    """Launch an asynchronous AURORA model-selection workflow job."""
    mode = req.mode.lower().strip()
    if mode not in ("approach_1", "approach_2", "both"):
        raise HTTPException(status_code=400, detail="Mode must be 'approach_1', 'approach_2', or 'both'.")

    preprocessed_path = os.path.join(PREPROCESSED_FOLDER, "preprocessed_dataset.csv")
    if not os.path.exists(preprocessed_path):
        raise HTTPException(status_code=400, detail="No preprocessed dataset found. Please upload a dataset first.")

    current_ds = STATE.get("current_dataset") or {}
    task_type = (req.task_type or current_ds.get("task_type") or "classification").lower().strip()
    if task_type not in ("classification", "regression"):
        raise HTTPException(status_code=400, detail="Invalid task_type. Use 'classification' or 'regression'.")

    job_id = str(uuid.uuid4())[:10]
    stages = [
        {
            "key": "dataset_prep",
            "title": "Dataset Verification & Schema Analysis",
            "description": "Verify preprocessed dataset, target variable, and task type configuration.",
            "status": "pending",
            "detail": "",
        },
        {
            "key": "approach_1",
            "title": "Approach 1 — Performance-Based Model Selection",
            "description": "Train and evaluate configured candidate models to identify Model A.",
            "status": "pending" if mode in ("approach_1", "both") else "skipped",
            "detail": "Skipped" if mode not in ("approach_1", "both") else "",
        },
        {
            "key": "approach_2",
            "title": "Approach 2 — Rule-Based Recommendation",
            "description": "Apply AURORA task-specific heuristic rules to recommend single highest-scoring Model B.",
            "status": "pending" if mode in ("approach_2", "both") else "skipped",
            "detail": "Skipped" if mode not in ("approach_2", "both") else "",
        },
        {
            "key": "finalize",
            "title": "Results Compilation & Comparison",
            "description": "Aggregate model metrics, rule scores, and comparison insights.",
            "status": "pending",
            "detail": "",
        },
    ]

    STATE["jobs"][job_id] = {
        "job_id": job_id,
        "status": "running",
        "mode": mode,
        "task_type": task_type,
        "created_at": datetime.now().isoformat(),
        "current_stage": "dataset_prep",
        "stages": stages,
        "logs": [],
        "approach_1_progress": None,
        "result": None,
        "error": None,
    }

    worker = threading.Thread(
        target=_execute_analysis_job,
        args=(job_id, mode, task_type, req.selection_metric, req.train_model_b),
        daemon=True,
    )
    worker.start()

    return {"job_id": job_id, "status": "running", "mode": mode, "task_type": task_type}


@app.get("/api/analysis/status/{job_id}")
def get_analysis_status(job_id: str):
    """Get real-time status, stages, logs, and results for an analysis job."""
    job = STATE["jobs"].get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return _safe_json_value(job)


@app.get("/api/analysis/latest")
def get_latest_analysis():
    """Return the most recent completed analysis result."""
    if STATE["latest_analysis"] is None:
        raise HTTPException(status_code=404, detail="No completed analysis available yet.")
    return _safe_json_value(STATE["latest_analysis"])


@app.get("/api/analysis/history")
def get_analysis_history():
    """Return the list of completed analyses from this backend."""
    return {"history": _safe_json_value(STATE["analysis_history"])}


@app.post("/api/analysis/train-model-b")
def train_recommended_model_b():
    """
    Optionally train and evaluate the rule-recommended Model B (as in run.py lines 336-374)
    if a trainer function exists for it in src/training/.
    """
    latest = STATE.get("latest_analysis")
    if not latest or not latest.get("approach_2"):
        raise HTTPException(status_code=400, detail="No Approach 2 recommendation available to train.")

    app2 = latest["approach_2"]
    model_b_name = app2.get("selected_model_b")
    if not model_b_name:
        raise HTTPException(status_code=400, detail="Approach 2 did not recommend a valid model.")

    trainer_fn = MODEL_TRAINERS_MAP.get(model_b_name)
    if trainer_fn is None:
        raise HTTPException(
            status_code=400,
            detail=f"No training function is configured in src/training/ for recommended model '{model_b_name}'. Predictive performance is not available for this recommendation.",
        )

    preprocessed_path = os.path.join(PREPROCESSED_FOLDER, "preprocessed_dataset.csv")
    if not os.path.exists(preprocessed_path):
        raise HTTPException(status_code=400, detail="Preprocessed dataset not found.")

    task_type = latest.get("task_type", "classification")
    training_path = _prepare_clean_training_csv_if_needed(preprocessed_path, task_type)

    t0 = time.time()
    metrics = _safe_json_value(trainer_fn(training_path))
    metrics["training_time_sec"] = round(time.time() - t0, 3)

    app2["is_trained"] = True
    app2["trained_metrics"] = metrics
    _save_analysis_to_history(latest)

    return _safe_json_value(latest)


@app.get("/api/tuning/configurations")
def get_tuning_configurations():
    """Return the hyperparameter tuning configurations supported by src/model_selection/hyperparameter_tuner.py."""
    return {
        "search_method": "RandomizedSearchCV",
        "n_iter": 20,
        "scoring": "f1_weighted",
        "cv_strategy": "StratifiedKFold (up to 5 splits, shuffle=True, random_state=42)",
        "supported_task_type": "classification",
        "models": TUNING_SEARCH_SPACES,
    }


@app.post("/api/tuning/optimize")
def optimize_hyperparameters(req: TuneModelRequest):
    """
    Execute AURORA's existing tune_selected_model(model_name, data_path, stage)
    from src/model_selection/hyperparameter_tuner.py.
    """
    preprocessed_path = os.path.join(PREPROCESSED_FOLDER, "preprocessed_dataset.csv")
    if not os.path.exists(preprocessed_path):
        raise HTTPException(status_code=400, detail="No preprocessed dataset found. Please upload a dataset first.")

    model_name = req.model_name.strip()
    if model_name not in TUNING_SEARCH_SPACES:
        raise HTTPException(
            status_code=400,
            detail=f"Hyperparameter tuning configuration is not defined in hyperparameter_tuner.py for '{model_name}'.",
        )

    training_path = _prepare_clean_training_csv_if_needed(preprocessed_path, "classification")

    try:
        t0 = time.time()
        raw_result = tune_selected_model(
            model_name=model_name,
            data_path=training_path,
            stage=req.stage,
        )
        elapsed = round(time.time() - t0, 3)

        # Strip non-JSON estimator object before returning
        tuned_response = {
            "Model": raw_result["Model"],
            "Best Parameters": _safe_json_value(raw_result["Best Parameters"]),
            "CV Score": round(float(raw_result["CV Score"]), 4),
            "Accuracy": round(float(raw_result["Accuracy"]), 4),
            "Precision": round(float(raw_result["Precision"]), 4),
            "Recall": round(float(raw_result["Recall"]), 4),
            "F1 Score": round(float(raw_result["F1 Score"]), 4),
            "Model Path": str(raw_result["Model Path"]),
            "tuning_time_sec": elapsed,
            "stage": req.stage,
            "search_method": "RandomizedSearchCV (n_iter=20, scoring='f1_weighted')",
        }

        # Attach baseline comparison if available in latest analysis
        latest = STATE.get("latest_analysis")
        if latest:
            baseline_metrics = None
            if latest.get("approach_1") and latest["approach_1"].get("candidate_models"):
                for cand in latest["approach_1"]["candidate_models"]:
                    if cand.get("Model") == model_name or (
                        model_name == "LDA" and cand.get("Model") == "Linear Discriminant Analysis"
                    ) or (
                        model_name == "QDA" and cand.get("Model") == "Quadratic Discriminant Analysis"
                    ) or (
                        model_name == "MLP" and cand.get("Model") == "Multi Layer Perceptron"
                    ):
                        baseline_metrics = {
                            "Accuracy": round(float(cand["Accuracy"]), 4),
                            "Precision": round(float(cand["Precision"]), 4),
                            "Recall": round(float(cand["Recall"]), 4),
                            "F1 Score": round(float(cand["F1 Score"]), 4),
                        }
                        break
            tuned_response["baseline_metrics"] = baseline_metrics

            if "tuning_results" not in latest or not isinstance(latest["tuning_results"], dict):
                latest["tuning_results"] = {}
            latest["tuning_results"][model_name] = tuned_response
            _save_analysis_to_history(latest)

        return _safe_json_value(tuned_response)

    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Hyperparameter tuning failed: {exc}")


@app.post("/api/evaluation/cross-validate")
def run_cross_validation(req: ModelEvaluationRequest):
    """
    Execute perform_cross_validation(model, dataset_path) from src/evaluation/cross_validation.py.
    """
    preprocessed_path = os.path.join(PREPROCESSED_FOLDER, "preprocessed_dataset.csv")
    if not os.path.exists(preprocessed_path):
        raise HTTPException(status_code=400, detail="No preprocessed dataset found.")

    model_path = req.model_path
    latest = STATE.get("latest_analysis")
    if not model_path and latest and latest.get("approach_1"):
        model_a = latest["approach_1"].get("selected_model_a") or {}
        model_path = model_a.get("Model Path")

    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail=f"Trained model file not found at '{model_path}'.")

    training_path = _prepare_clean_training_csv_if_needed(preprocessed_path, "classification")

    try:
        model = joblib.load(model_path)
        scores = perform_cross_validation(model, training_path)
        cv_result = {
            "model_path": model_path,
            "model_name": req.model_name or os.path.basename(model_path),
            "fold_scores": [round(float(s), 4) for s in scores],
            "mean_accuracy": round(float(np.mean(scores)), 4),
            "std_accuracy": round(float(np.std(scores)), 4),
            "n_splits": len(scores),
        }
        if latest:
            latest["cross_validation"] = cv_result
            _save_analysis_to_history(latest)
        return cv_result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"5-Fold Cross-Validation failed: {exc}")


@app.post("/api/evaluation/shap")
def run_shap_explainability(req: ModelEvaluationRequest):
    """
    Execute generate_shap(model_path, dataset_path) from src/explainability/shap_explainer.py.
    """
    from src.explainability.shap_explainer import generate_shap

    preprocessed_path = os.path.join(PREPROCESSED_FOLDER, "preprocessed_dataset.csv")
    if not os.path.exists(preprocessed_path):
        raise HTTPException(status_code=400, detail="No preprocessed dataset found.")

    model_path = req.model_path
    latest = STATE.get("latest_analysis")
    if not model_path and latest and latest.get("approach_1"):
        model_a = latest["approach_1"].get("selected_model_a") or {}
        model_path = model_a.get("Model Path")

    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail=f"Trained model file not found at '{model_path}'.")

    training_path = _prepare_clean_training_csv_if_needed(preprocessed_path, "classification")

    try:
        generate_shap(model_path, training_path)
        if latest:
            latest["shap_available"] = True
            _save_analysis_to_history(latest)
        return {
            "status": "completed",
            "model_path": model_path,
            "shap_image_url": f"/api/reports/shap-image?t={int(time.time())}",
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"SHAP explainability generation failed: {exc}")


@app.get("/api/reports/shap-image")
def get_shap_image():
    """Serve the generated SHAP summary plot from reports/shap_summary.png."""
    shap_path = os.path.join(REPORT_FOLDER, "shap_summary.png")
    if not os.path.exists(shap_path):
        raise HTTPException(status_code=404, detail="SHAP summary plot has not been generated yet.")
    return FileResponse(shap_path, media_type="image/png")


@app.get("/api/reports/preprocessing-csv")
def download_preprocessing_report():
    """Serve reports/preprocessing_report.csv."""
    report_path = os.path.join(REPORT_FOLDER, "preprocessing_report.csv")
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Preprocessing report not found.")
    return FileResponse(
        report_path,
        media_type="text/csv",
        filename="aurora_preprocessing_report.csv",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)

