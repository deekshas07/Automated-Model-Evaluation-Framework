
import os
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def tune_selected_model(model_name, data_path, stage="approach"):

    # =========================================
    # LOAD DATASET
    # =========================================

    df = pd.read_csv(data_path)

    df = df.dropna(how="all")
    df = df.dropna(subset=[df.columns[-1]])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Encode categorical features
    X = pd.get_dummies(X)

    # Fill missing feature values
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)

    # Encode target labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # =========================================
    # TRAIN / TEST SPLIT
    # =========================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    num_classes = len(set(y))

    # =========================================
    # MODEL CONFIGURATIONS
    # =========================================

    # 1. DECISION TREE
    if model_name == "Decision Tree":

        model = DecisionTreeClassifier(random_state=42)

        param_dist = {
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8]
        }

    # 2. RANDOM FOREST
    elif model_name == "Random Forest":

        model = RandomForestClassifier(
            random_state=42,
            n_jobs=1
        )

        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }

    # 3. XGBOOST
    elif model_name == "XGBoost":

        from xgboost import XGBClassifier

        if num_classes == 2:
            objective = "binary:logistic"
            eval_metric = "logloss"
        else:
            objective = "multi:softprob"
            eval_metric = "mlogloss"

        model = XGBClassifier(
            objective=objective,
            eval_metric=eval_metric,
            num_class=num_classes if num_classes > 2 else None,
            random_state=42,
            n_jobs=1
        )

        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6, 8, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0]
        }

    # 4. CATBOOST
    elif model_name == "CatBoost":

        from catboost import CatBoostClassifier

        loss_function = (
            "Logloss" if num_classes == 2 else "MultiClass"
        )

        model = CatBoostClassifier(
            loss_function=loss_function,
            verbose=0,
            random_seed=42,
            thread_count=1
        )

        param_dist = {
            "iterations": [100, 200, 300, 500],
            "depth": [4, 5, 6, 7, 8, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "l2_leaf_reg": [1, 3, 5, 7, 9]
        }

    # 5. ADABOOST
    elif model_name == "AdaBoost":

        model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                random_state=42
            ),
            random_state=42
        )

        param_dist = {
            "n_estimators": [50, 100, 150, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
            "estimator__max_depth": [1, 2, 3, 4],
            "estimator__min_samples_split": [2, 5, 10],
            "estimator__min_samples_leaf": [1, 2, 4]
        }

    # 6. GRADIENT BOOSTING
    elif model_name == "Gradient Boosting":

        model = GradientBoostingClassifier(
            random_state=42
        )

        param_dist = {
            "n_estimators": [50, 100, 150, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [2, 3, 4, 5],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "max_features": [None, "sqrt", "log2"]
        }

    # 7. HIST GRADIENT BOOSTING
    elif model_name in [
        "HistGradientBoosting",
        "Hist Gradient Boosting"
    ]:

        model = HistGradientBoostingClassifier(
            random_state=42
        )

        param_dist = {
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_iter": [100, 150, 200, 300],
            "max_depth": [None, 3, 5, 6, 8, 10],
            "min_samples_leaf": [10, 20, 30, 50],
            "l2_regularization": [0.0, 0.1, 0.5, 1.0],
            "max_bins": [100, 150, 200, 255]
        }

    # 8. LIGHTGBM
    elif model_name == "LightGBM":

        from lightgbm import LGBMClassifier

        if num_classes == 2:
            objective = "binary"
        else:
            objective = "multiclass"

        model = LGBMClassifier(
            objective=objective,
            num_class=num_classes if num_classes > 2 else None,
            random_state=42,
            n_jobs=1,
            verbosity=-1
        )

        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [-1, 3, 5, 6, 8, 10],
            "num_leaves": [15, 31, 50, 70],
            "min_child_samples": [10, 20, 30, 50],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0]
        }

    # 9. LOGISTIC REGRESSION
    elif model_name == "Logistic Regression":

        model = LogisticRegression(
            max_iter=2000,
            random_state=42
        )

        param_dist = {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs"],
            "class_weight": [None, "balanced"]
        }

    # 10. KNN
    elif model_name == "KNN":

        model = KNeighborsClassifier()

        param_dist = {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"],
            "p": [1, 2]
        }

    # 11. SVM
    elif model_name == "SVM":

        model = SVC(
            probability=True,
            random_state=42
        )

        param_dist = {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"],
            "degree": [2, 3, 4]
        }

    # 12. LINEAR DISCRIMINANT ANALYSIS
    elif model_name in [
        "LDA",
        "Linear Discriminant Analysis"
    ]:

        model = LinearDiscriminantAnalysis()

        # Separate valid combinations:
        # shrinkage cannot be used with solver="svd"
        param_dist = [
            {
                "solver": ["svd"]
            },
            {
                "solver": ["lsqr", "eigen"],
                "shrinkage": [None, "auto", 0.1, 0.5, 0.9]
            }
        ]

    # 13. QUADRATIC DISCRIMINANT ANALYSIS
    elif model_name in [
        "QDA",
        "Quadratic Discriminant Analysis"
    ]:

        model = QuadraticDiscriminantAnalysis()

        param_dist = {
            "reg_param": [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
        }

    # 14. MULTI LAYER PERCEPTRON
    elif model_name in [
        "MLP",
        "Multi Layer Perceptron"
    ]:

        model = MLPClassifier(
            max_iter=800,
            random_state=42
        )

        param_dist = {
            "hidden_layer_sizes": [
                (50,),
                (100,),
                (100, 50),
                (100, 50, 25)
            ],
            "activation": ["relu", "tanh"],
            "solver": ["adam"],
            "alpha": [0.0001, 0.001, 0.01, 0.1],
            "learning_rate": ["constant", "adaptive"],
            "learning_rate_init": [0.001, 0.01]
        }

    # 15. NAIVE BAYES
    elif model_name == "Naive Bayes":

        model = GaussianNB()

        param_dist = {
            "var_smoothing": [
                1e-12,
                1e-11,
                1e-10,
                1e-9,
                1e-8,
                1e-7,
                1e-6
            ]
        }

    else:

        raise ValueError(
            f"Tuning configuration not added for: {model_name}"
        )

    # =========================================
    # CROSS VALIDATION
    # =========================================

    class_counts = pd.Series(y_train).value_counts()

    n_splits = min(5, int(class_counts.min()))

    if n_splits < 2:
        raise ValueError(
            "Not enough samples per class for stratified cross-validation."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    # =========================================
    # RANDOMIZED SEARCH CV
    # =========================================

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=20,
        scoring="f1_weighted",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        error_score="raise"
    )

    print("\n" + "=" * 60)
    print(f"RANDOMIZED SEARCH CV - {model_name}")
    print("=" * 60)

    with joblib.parallel_backend("threading", n_jobs=4):
        search.fit(X_train, y_train)

    # =========================================
    # BEST TUNED MODEL
    # =========================================

    best_model = search.best_estimator_

    # =========================================
    # TEST EVALUATION
    # =========================================

    predictions = best_model.predict(X_test)

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

    # =========================================
    # SAVE TUNED MODEL
    # =========================================

    os.makedirs("models", exist_ok=True)

    safe_name = model_name.lower().replace(" ", "_")

    model_path = os.path.join(
        "models",
        f"tuned_{stage}_{safe_name}.pkl"
    )

    joblib.dump(best_model, model_path)

    # =========================================
    # RETURN RESULTS
    # =========================================

    return {
        "Model": model_name,
        "Best Parameters": search.best_params_,
        "CV Score": search.best_score_,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Model Path": model_path,
        "Best Estimator": best_model
    }
