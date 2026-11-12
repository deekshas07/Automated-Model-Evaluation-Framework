# AURORA — Backend Integration Map (`BACKEND_INTEGRATION_MAP.md`)

## 1. Executive Overview of Existing Backend

The existing **AURORA — Intelligent Model Recommendation System** backend is implemented in Python and organized around a hybrid symbolic/neural preprocessing engine, a performance-based model evaluation pipeline (**Approach 1**), a deterministic rule-based model recommendation engine (**Approach 2**), a `RandomizedSearchCV` hyperparameter tuning module, 5-Fold Cross-Validation, and SHAP Explainable AI.

### Existing Directory & Module Inventory

| Path | Role & Responsibility |
| :--- | :--- |
| [`run.py`](file:///c:/Users/vigne/Desktop/Deeksha/report%20mini/Backend-20260703T105439Z-3-001/Automated_ML_Model%20-%20Copy%20(2)/run.py) | CLI pipeline orchestrator: `preprocess_dataset(dataset_path)`, Approach 1 model training & selection, Approach 2 rule-based selection & optional training, hyperparameter tuning, final comparison, 5-Fold CV, and SHAP generation. |
| [`src/core/preprocessor.py`](file:///c:/Users/vigne/Desktop/Deeksha/report%20mini/Backend-20260703T105439Z-3-001/Automated_ML_Model%20-%20Copy%20(2)/src/core/preprocessor.py) | `IntelligentPreprocessor` & `get_preprocessor()`: 4-layer column preprocessing analysis (185+ Symbolic Rules -> NeuralOracle -> SafetyValidator -> Conservative Fallback). |
| [`src/core/column_profiler.py`](file:///c:/Users/vigne/Desktop/Deeksha/report%20mini/Backend-20260703T105439Z-3-001/Automated_ML_Model%20-%20Copy%20(2)/src/core/column_profiler.py) | `ColumnProfiler`: computes per-column numeric/categorical statistics, distributions, histograms, and IQR outliers. |
| [`src/core/robust_parser.py`](file:///c:/Users/vigne/Desktop/Deeksha/report%20mini/Backend-20260703T105439Z-3-001/Automated_ML_Model%20-%20Copy%20(2)/src/core/robust_parser.py) | `RobustCSVParser` & `parse_csv_robust`: encoding detection (`chardet`), delimiter detection, and dataframe validation. |
| [`src/utils/target_detector.py`](file:///c:/Users/vigne/Desktop/Deeksha/report%20mini/Backend-20260703T105439Z-3-001/Automated_ML_Model%20-%20Copy%20(2)/src/utils/target_detector.py) | `TargetDetector`: target column detection and statistical task characterization (classification vs. regression). |
| [`src/utils/universal_type_detector.py`](file:///c:/Users/vigne/Desktop/Deeksha/report%20mini/Backend-20260703T105439Z-3-001/Automated_ML_Model%20-%20Copy%20(2)/src/utils/universal_type_detector.py) | `UniversalTypeDetector`: semantic column type classification (`numeric`, `categorical`, `boolean`, `datetime`, `identifier`, `text`, etc.). |
| [`src/model_selection/rule_based_selector.py`](file:///c:/Users/vigne/Desktop/Deeksha/report%20mini/Backend-20260703T105439Z-3-001/Automated_ML_Model%20-%20Copy%20(2)/src/model_selection/rule_based_selector.py) | `select_model_using_rules(X, y, task_type)`: executes task-specific rules (`classification` or `regression`) and selects **exactly ONE highest-scoring model** (`selected_model`, `recommendation_score`, `reasons`, `message`). |
| [`src/model_selection/model_evaluation_rules.py`](file:///c:/Users/vigne/Desktop/Deeksha/report%20mini/Backend-20260703T105439Z-3-001/Automated_ML_Model%20-%20Copy%20(2)/src/model_selection/model_evaluation_rules.py) | Contains `MODELS` (15 classification models) + `apply_all_model_rules` (Rules 1–160) and `REGRESSION_MODELS` (15 regression models) + `apply_all_regression_rules` (Rules R1–R85). |
| [`src/model_selection/hyperparameter_tuner.py`](file:///c:/Users/vigne/Desktop/Deeksha/report%20mini/Backend-20260703T105439Z-3-001/Automated_ML_Model%20-%20Copy%20(2)/src/model_selection/hyperparameter_tuner.py) | `tune_selected_model(model_name, data_path, stage)`: runs `RandomizedSearchCV` (`n_iter=20`, `StratifiedKFold`, `scoring="f1_weighted"`) for all 15 classification models and saves `models/tuned_{stage}_{safe_name}.pkl`. |
| [`src/training/train_*.py`](file:///c:/Users/vigne/Desktop/Deeksha/report%20mini/Backend-20260703T105439Z-3-001/Automated_ML_Model%20-%20Copy%20(2)/src/training/) | 15 classification model training scripts + 1 regression training script (`train_linear_regression.py`). |
| [`src/evaluation/cross_validation.py`](file:///c:/Users/vigne/Desktop/Deeksha/report%20mini/Backend-20260703T105439Z-3-001/Automated_ML_Model%20-%20Copy%20(2)/src/evaluation/cross_validation.py) | `perform_cross_validation(model, dataset_path)`: 5-Fold Stratified Cross-Validation returning fold accuracy array. |
| [`src/explainability/shap_explainer.py`](file:///c:/Users/vigne/Desktop/Deeksha/report%20mini/Backend-20260703T105439Z-3-001/Automated_ML_Model%20-%20Copy%20(2)/src/explainability/shap_explainer.py) | `generate_shap(model_path, dataset_path)`: generates SHAP summary plot at `reports/shap_summary.png`. |

---

## 2. Core Backend Functions & Exact Signatures

### 2.1 Dataset Preprocessing & Analysis (`run.py` & `src/core/preprocessor.py`)
- **Target Column Convention**: As defined in `README.md` and `run.py`, the **last column** of the CSV dataset (`df.columns[-1]`) is treated as the target variable (`y = df.iloc[:, -1]`), and all preceding columns (`df.columns[:-1]`) are input features (`X = df.iloc[:, :-1]`).
- **Function**: `preprocess_dataset(dataset_path: str) -> str`
  - Reads CSV from `dataset_path`.
  - Runs `get_preprocessor().preprocess_column(df[column], column_name=column)` on each feature column `df.columns[:-1]`.
  - Writes column-level recommendations (`Column`, `Recommended Action`, `Confidence`, `Source`) to `reports/preprocessing_report.csv`.
  - Saves the working dataset to `preprocessed/preprocessed_dataset.csv` and returns its path.
- **Task Type Identification (`classification` vs. `regression`)**:
  - Determined from the target series `y = df.iloc[:, -1]` (using target data type and cardinality rules consistent with `TargetDetector._check_statistical_properties`):
    - Non-numeric dtype (`object`, `category`, `bool`, `string`) OR integer/numeric with `<= 20` unique values (or unique ratio `< 0.05` on larger datasets) $\rightarrow$ `"classification"`.
    - Continuous numeric target with `> 20` unique values and unique ratio $\ge 0.05$ $\rightarrow$ `"regression"`.
    - User can also inspect or explicitly confirm the detected task type before running model selection.

### 2.2 Approach 1 — Performance-Based Model Selection (Without Rules)
- **Classification Candidate Model Registry** (15 models configured in `run.py`):
  1. `train_decision_tree(data_path)` $\rightarrow$ `"Decision Tree"`
  2. `train_naive_bayes(data_path)` $\rightarrow$ `"Naive Bayes"`
  3. `train_random_forest(data_path)` $\rightarrow$ `"Random Forest"`
  4. `train_xgboost(data_path)` $\rightarrow$ `"XGBoost"`
  5. `train_catboost(data_path)` $\rightarrow$ `"CatBoost"`
  6. `train_adaboost(data_path)` $\rightarrow$ `"AdaBoost"`
  7. `train_gradient_boosting(data_path)` $\rightarrow$ `"Gradient Boosting"`
  8. `train_hist_gradient_boosting(data_path)` $\rightarrow$ `"HistGradientBoosting"`
  9. `train_lightgbm(data_path)` $\rightarrow$ `"LightGBM"`
  10. `train_logistic_regression(data_path)` $\rightarrow$ `"Logistic Regression"`
  11. `train_knn(data_path)` $\rightarrow$ `"KNN"`
  12. `train_svm(data_path)` $\rightarrow$ `"SVM"`
  13. `train_linear_discriminant_analysis(data_path)` $\rightarrow$ `"Linear Discriminant Analysis"`
  14. `train_quadratic_discriminant_analysis(data_path)` $\rightarrow$ `"Quadratic Discriminant Analysis"`
  15. `train_mlp(data_path)` $\rightarrow$ `"Multi Layer Perceptron"`
  - **Return Format per Classification Trainer**:
    ```json
    {
      "Model": "Random Forest",
      "Accuracy": 0.9672,
      "Precision": 0.9681,
      "Recall": 0.9672,
      "F1 Score": 0.9671,
      "Model Path": "models/random_forest.pkl"
    }
    ```
  - **Selection Policy (`run.py` line 256)**:
    `best = max(results, key=lambda x: x["Accuracy"])` selects **Model A**.

- **Regression Candidate Model Registry** (configured in `src/training/train_linear_regression.py`):
  1. `train_linear_regression(data_path)` $\rightarrow$ `"Linear Regression"`
  - **Return Format per Regression Trainer**:
    ```json
    {
      "Model": "Linear Regression",
      "MAE": 0.5321,
      "MSE": 0.5559,
      "RMSE": 0.7456,
      "R2 Score": 0.5758,
      "Model Path": "models/linear_regression.pkl"
    }
    ```
  - **Selection Policy**:
    `best = max(results, key=lambda x: x["R2 Score"])` selects **Model A**.

### 2.3 Approach 2 — Rule-Based Recommendation (`src/model_selection/rule_based_selector.py`)
- **Function**: `select_model_using_rules(X, y, task_type)`
  - Calls `model_evaluation_rules.apply_all_model_rules(X, y, task_type="classification")` (Rules 1–160 across `MODELS`) or `model_evaluation_rules.apply_all_regression_rules(X, y)` (Rules R1–R85 across `REGRESSION_MODELS`).
  - Filters models with positive score (`score > 0`) and selects **exclusively ONE highest-scoring model** (`max(eligible_models, key=lambda model: (eligible_models[model], model))`).
  - **Return Format**:
    ```json
    {
      "selected_model": "CatBoost",
      "recommendation_score": 16.0,
      "reasons": [
        "Categorical features are present; evaluate categorical-aware boosting",
        "Mixed feature types make categorical-aware boosting worth evaluating"
      ],
      "message": "Rule-based model recommendation completed."
    }
    ```
  - **Optional Model B Training (`run.py` lines 336–374)**:
    - When `model_b` maps to an implemented trainer in `src/training/` (all 15 classification models via `model_trainers` in `run.py`, or `"Linear Regression"` via `train_linear_regression` for regression), the backend can also train and evaluate Model B to obtain its measured metrics (`model_b_training_result`), while keeping `recommendation_score` strictly distinct from predictive metrics (`Accuracy`, `F1 Score`, `R2 Score`, etc.).
    - If Model B has not been trained and evaluated (e.g., when a regression rule recommends a regressor without a trainer in `src/training/`, or if rule-only evaluation is requested), `model_b_training_result` is `null` and the UI explicitly displays: `"Predictive performance is not available for this recommendation."`

### 2.4 Hyperparameter Tuning (`src/model_selection/hyperparameter_tuner.py`)
- **Function**: `tune_selected_model(model_name: str, data_path: str, stage: str = "approach") -> dict`
  - Uses `RandomizedSearchCV` (`n_iter=20`, `scoring="f1_weighted"`, `StratifiedKFold(n_splits=min(5, min_class_count), shuffle=True, random_state=42)`).
  - Supports all 15 classification models (`Decision Tree`, `Random Forest`, `XGBoost`, `CatBoost`, `AdaBoost`, `Gradient Boosting`, `HistGradientBoosting` / `Hist Gradient Boosting`, `LightGBM`, `Logistic Regression`, `KNN`, `SVM`, `LDA` / `Linear Discriminant Analysis`, `QDA` / `Quadratic Discriminant Analysis`, `MLP` / `Multi Layer Perceptron`, `Naive Bayes`).
  - **Return Format**:
    ```json
    {
      "Model": "Random Forest",
      "Best Parameters": {
        "n_estimators": 300,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_depth": 15,
        "criterion": "gini"
      },
      "CV Score": 0.8412,
      "Accuracy": 0.8689,
      "Precision": 0.8695,
      "Recall": 0.8689,
      "F1 Score": 0.8687,
      "Model Path": "models/tuned_approach_1_random_forest.pkl"
    }
    ```
    *(Note: `Best Estimator` is stripped or serialized as string representation before JSON response).*

### 2.5 Cross-Validation & Explainable AI (`src/evaluation/cross_validation.py` & `src/explainability/shap_explainer.py`)
- **Cross-Validation**: `perform_cross_validation(model, dataset_path)` returns `numpy.ndarray` of 5 fold accuracy scores (`scores`, `scores.mean()`, `scores.std()`).
- **SHAP Explainability**: `generate_shap(model_path, dataset_path)` saves `reports/shap_summary.png`.

---

## 3. FastAPI Backend Service Layer (`api.py`)

Because the existing project only has a CLI entry point (`run.py`) and already includes `fastapi`, `uvicorn`, `pydantic`, and `python-multipart` in `requirements.txt`, we create [`api.py`](file:///c:/Users/vigne/Desktop/Deeksha/report%20mini/Backend-20260703T105439Z-3-001/Automated_ML_Model%20-%20Copy%20(2)/api.py) as a non-invasive HTTP bridge over the exact existing functions in `run.py` and `src/`.

### API Endpoints Specification

| Method & Path | Purpose | Underlying Backend Functions Called |
| :--- | :--- | :--- |
| `GET /api/health` | Health check & backend capability metadata | Checks model registry & folders (`datasets/`, `preprocessed/`, `reports/`, `models/`) |
| `GET /api/datasets/samples` | List built-in CSV datasets in `datasets/` folder for quick demo or testing, alongside file upload | Scans `datasets/*.csv` |
| `POST /api/datasets/upload` | Upload a CSV file (`multipart/form-data`), parse, run `preprocess_dataset()`, detect task type, profile columns, and return full Dataset Overview | `preprocess_dataset()` (`run.py`), `get_preprocessor()`, `ColumnProfiler`, `UniversalTypeDetector`, `TargetDetector` |
| `POST /api/datasets/load-sample` | Load one of the existing datasets in `datasets/` through the exact same `preprocess_dataset()` and profiling pipeline | `preprocess_dataset()` (`run.py`), `get_preprocessor()` |
| `GET /api/datasets/current` | Retrieve current uploaded/active dataset overview, preprocessing report, column profiles, and preview rows | Reads `preprocessed/preprocessed_dataset.csv` and `reports/preprocessing_report.csv` |
| `POST /api/analysis/start` | Launch asynchronous analysis job for `approach_1`, `approach_2`, or `both` on the preprocessed dataset | Background thread executing Approach 1 (`train_*` loop in `run.py`) and/or Approach 2 (`select_model_using_rules` + trainer in `run.py`) with real-time stage/event logging |
| `GET /api/analysis/status/{job_id}` | Poll real-time execution status, current stage, completed models, live logs, and final results | Reads live job state from memory/disk store |
| `GET /api/analysis/latest` | Fetch the most recent completed analysis result (and session history) | Returns latest completed job & history list |
| `POST /api/tuning/optimize` | Run hyperparameter tuning (`RandomizedSearchCV`) on a selected model (`Model A`, `Model B`, or any candidate model) | `tune_selected_model(model_name, data_path, stage)` (`src/model_selection/hyperparameter_tuner.py`) |
| `POST /api/evaluation/cross-validate` | Run 5-Fold Stratified Cross-Validation on a trained model | `perform_cross_validation(model, dataset_path)` (`src/evaluation/cross_validation.py`) |
| `POST /api/evaluation/shap` | Generate SHAP feature importance summary plot for a trained model | `generate_shap(model_path, dataset_path)` (`src/explainability/shap_explainer.py`) |
| `GET /api/reports/shap-image` | Serve the generated `reports/shap_summary.png` image | `FileResponse("reports/shap_summary.png")` |
| `GET /api/reports/preprocessing-csv` | Download `reports/preprocessing_report.csv` | `FileResponse("reports/preprocessing_report.csv")` |

---

## 4. Frontend Integration Plan (`frontend/`)

- **Stack**: React 18 + TypeScript + Vite + Tailwind CSS + React Router v6 + Axios + Recharts + Lucide React.
- **Design System**:
  - Navy (`#0A1128`, `#0F172A`, `#1E293B`), Crisp White (`#FFFFFF`, `#F8FAFC`), and Electric Blue (`#2563EB`, `#3B82F6`, `#06B6D4`) palette.
  - Typography: `Inter` / `Plus Jakarta Sans` for UI & headings, `JetBrains Mono` for metrics, parameters, and logs.
- **Pages & Routes**:
  1. `/` — **Landing Page** (Hero with custom interactive AURORA workflow/model-intelligence visual, Key Features, 4-Step How It Works, CTA, Footer).
  2. `/dashboard` — **Application Dashboard** (Welcome, capability overview, workflow launcher, session analysis history).
  3. `/upload` — **Dataset Upload** (Drag-and-drop CSV upload, validation, plus quick-load option for workspace benchmark datasets).
  4. `/overview` — **Dataset Overview** (Rows, columns, target variable, task type badge & selector, missing values, data types, AURORA symbolic preprocessing report table, and interactive dataset preview).
  5. `/select-approach` — **Model Selection** (Option cards for Approach 1 — Performance-Based Selection, Approach 2 — Rule-Based Recommendation, or Combined Execution of both approaches).
  6. `/processing` — **Processing** (Live backend event stages, real-time model training log stream, actual candidate progress, error/recovery state).
  7. `/results` — **Results Dashboard** (Model A card & metrics, Model B card with Rule Recommendation Score & triggered rules, full candidate leaderboard chart & table, Cross-Validation & SHAP explainability integration).
  8. `/comparison` — **Model Comparison** (Side-by-side comparison of Approach 1 vs. Approach 2 methodology, selection basis, rule reasons, and measured performance—or explicit `"Predictive performance is not available for this recommendation."` notice when Model B is untrainable/untrained).
  9. `/tuning` — **Hyperparameter Tuning** (Interactive `RandomizedSearchCV` tuning trigger for Model A, Model B, or any candidate classification model, parameter search space inspection, before-vs-after metric comparison, and best parameters display).

