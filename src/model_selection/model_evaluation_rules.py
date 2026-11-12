def apply_categorical_rules(X, y, task_type, add_score):
    import numpy as np

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns

    n_rows, n_features = X.shape
    categorical_count = len(categorical_columns)
    numeric_count = len(numeric_columns)

    if n_features == 0 or n_rows == 0:
        return

    categorical_ratio = categorical_count / n_features
    feature_ratio = n_rows / n_features

    # Analyse categorical complexity
    high_cardinality_count = 0
    rare_category_count = 0

    for col in categorical_columns:
        values = X[col].dropna()

        if values.empty:
            continue

        unique_count = values.nunique()
        unique_ratio = unique_count / len(values)

        # Rule support: high-cardinality features
        if unique_count >= 20 and unique_ratio >= 0.05:
            high_cardinality_count += 1

        # Rule support: rare categories
        frequencies = values.value_counts(normalize=True)

        if (frequencies < 0.01).any():
            rare_category_count += 1

    high_cardinality_ratio = (
        high_cardinality_count / categorical_count
        if categorical_count else 0
    )

    rare_category_ratio = (
        rare_category_count / categorical_count
        if categorical_count else 0
    )

    # Rule 31: Categorical features exist
    if categorical_count > 0:
        add_score(
            "CatBoost", 2,
            "Categorical features are present; evaluate categorical-aware boosting"
        )

    # Rule 32: Mixed numerical and categorical features
    if categorical_count > 0 and numeric_count > 0:
        add_score(
            "CatBoost", 2,
            "Mixed feature types make categorical-aware boosting worth evaluating"
        )

    # Rule 33: High-cardinality categorical features
    if high_cardinality_ratio >= 0.30:
        add_score(
            "CatBoost", 3,
            "At least 30% of categorical columns have high cardinality"
        )

    # Rule 34: Rare categories
    if rare_category_ratio >= 0.30:
        add_score(
            "CatBoost", 2,
            "Several categorical columns contain rare categories"
        )

    # Rule 35: Categorical features dominate
    if categorical_ratio >= 0.50 and categorical_count >= 3:
        add_score(
            "CatBoost", 2,
            "Categorical columns form at least half of all features"
        )

    # Rule 36: Many categorical features relative to dataset size
    if categorical_count >= 5 and feature_ratio >= 10:
        add_score(
            "CatBoost", 2,
            "Multiple categorical features with sufficient observations per feature"
        )

    # Rule 37: Numerical features dominate
    if numeric_count > categorical_count and numeric_count >= 3:
        add_score(
            "XGBoost", 2,
            "Numerical features dominate, making gradient-boosted trees a candidate"
        )

    # Rule 38: Numerical features and high dimensionality
    if numeric_count >= 3 and feature_ratio < 20:
        add_score(
            "Random Forest", 2,
            "Numerical features are present in a high-dimensional dataset"
        )

    # Rule 39: Large, purely numerical dataset
    if numeric_count == n_features and n_rows >= 5000:
        add_score(
            "HistGradientBoosting", 2,
            "Large numerical dataset is suitable for evaluating histogram-based boosting"
        )

    # Rule 40: High-cardinality categorical features are prevalent
    if categorical_count >= 2 and high_cardinality_ratio >= 0.50:
        add_score(
            "CatBoost", 2,
            "At least half of categorical columns have high cardinality"
        )







def apply_classification_rules(X, y, task_type, add_score):

    if task_type != "classification":
        return

    n_samples = len(y)
    n_features = X.shape[1]
    n_classes = y.nunique()

    if n_samples == 0 or n_features == 0 or n_classes < 2:
        return

    samples_per_class = n_samples / n_classes
    feature_ratio = n_samples / n_features

    # Rule 51: Binary classification with enough samples
    if n_classes == 2 and samples_per_class >= 30:
        add_score(
            "Logistic Regression", 2,
            "Binary classification with sufficient samples supports evaluating Logistic Regression"
        )

    # Rule 52: Binary or multiclass classification
    if n_classes >= 2 and n_samples <= 5000:
        add_score(
            "SVM", 2,
            "Dataset size makes SVM a candidate for validation"
        )

    # Rule 53: Binary classification with nonlinear patterns possible
    if n_classes == 2 and n_features >= 3:
        add_score(
            "Random Forest", 2,
            "Random Forest can model nonlinear relationships in binary classification"
        )

    # Rule 54: Multiclass classification
    if n_classes > 2:
        add_score(
            "Random Forest", 2,
            "Random Forest supports multiclass classification"
        )

    # Rule 55: Multiclass classification with adequate samples
    if n_classes > 2 and samples_per_class >= 30:
        add_score(
            "XGBoost", 2,
            "Multiclass boosting is worth evaluating with adequate samples per class"
        )

    # Rule 56: Multiclass classification with sufficient observations
    if n_classes > 2 and n_samples >= 1000:
        add_score(
            "LightGBM", 2,
            "LightGBM is a candidate for multiclass datasets with many observations"
        )

    # Rule 57: Many target classes
    if n_classes >= 5 and samples_per_class >= 20:
        add_score(
            "Random Forest", 2,
            "Multiple classes with sufficient samples per class suit tree-ensemble evaluation"
        )

    # Rule 58: Many classes and a larger dataset
    if n_classes >= 5 and n_samples >= 5000:
        add_score(
            "LightGBM", 2,
            "Large multiclass dataset makes scalable boosting worth evaluating"
        )

    # Rule 59: Small binary dataset
    if n_classes == 2 and n_samples < 1000 and feature_ratio >= 5:
        add_score(
            "Logistic Regression", 2,
            "Small binary dataset with manageable dimensionality suits a simple baseline"
        )

    # Rule 60: Small dataset with limited feature count
    if n_samples < 1000 and n_features <= 50 and samples_per_class >= 10:
        add_score(
            "SVM", 2,
            "Limited feature count and sample size make SVM worth evaluating"
        )

    # Rule 61: Large binary dataset
    if n_classes == 2 and n_samples >= 10000:
        add_score(
            "LightGBM", 2,
            "Large binary dataset makes efficient gradient boosting a candidate"
        )

    # Rule 62: Large dataset with adequate samples per class
    if n_samples >= 10000 and samples_per_class >= 50:
        add_score(
            "XGBoost", 2,
            "Large dataset with adequate class representation supports boosting evaluation"
        )

    # Rule 63: Low-dimensional classification
    if n_features <= 20 and feature_ratio >= 10:
        add_score(
            "LDA", 2,
            "Low-dimensional dataset makes Linear Discriminant Analysis worth evaluating"
        )

    # Rule 64: Low-dimensional classification with sufficient samples
    if n_features <= 20 and feature_ratio >= 10 and samples_per_class >= 20:
        add_score(
            "QDA", 1,
            "Low-dimensional data with adequate class samples permits evaluating QDA"
        )

    # Rule 65: Multiclass data with sufficient observations
    if n_classes >= 3 and n_samples >= 1000 and feature_ratio >= 5:
        add_score(
            "MLP", 2,
            "Multiclass dataset with sufficient observations makes nonlinear neural learning a candidate"
        )








def apply_complexity_rules(X, y, task_type, add_score):

    n_samples, n_features = X.shape

    if n_samples == 0 or n_features == 0:
        return

    feature_ratio = n_samples / n_features

    # Rule 76: Moderate or high feature count
    if n_features >= 10 and feature_ratio >= 10:
        add_score(
            "Random Forest", 2,
            "Multiple features and sufficient observations make Random Forest worth evaluating"
        )

    # Rule 77: High-dimensional dataset
    if n_features >= 100 and feature_ratio < 20:
        add_score(
            "Random Forest", 2,
            "High dimensionality makes a tree ensemble worth evaluating"
        )

    # Rule 78: Nonlinear relationships may exist
    if n_features >= 10 and n_samples >= 1000:
        add_score(
            "Gradient Boosting", 2,
            "Dataset size and feature count support evaluating gradient-boosted trees"
        )

    # Rule 79: Large dataset with manageable dimensionality
    if n_samples >= 5000 and feature_ratio >= 10:
        add_score(
            "XGBoost", 2,
            "Large dataset with sufficient observations per feature suits boosting evaluation"
        )

    # Rule 80: Large dataset suitable for scalable boosting
    if n_samples >= 5000 and n_features >= 10:
        add_score(
            "LightGBM", 2,
            "Large dataset makes efficient gradient boosting worth evaluating"
        )

    # Rule 81: Large dataset with categorical or mixed features
    categorical_count = len(
        X.select_dtypes(
            include=["object", "category", "bool", "string"]
        ).columns
    )

    if n_samples >= 5000 and categorical_count > 0:
        add_score(
            "CatBoost", 2,
            "Large dataset containing categorical features makes CatBoost worth evaluating"
        )

    # Rule 82: Small feature space with enough observations
    if n_features <= 10 and n_samples >= 1000:
        add_score(
            "Decision Tree", 2,
            "Small feature space makes an interpretable tree model worth evaluating"
        )

    # Rule 83: Low-dimensional dataset
    if n_features <= 10 and feature_ratio >= 20 and n_samples <= 5000:
        add_score(
            "KNN", 2,
            "Low-dimensional data with sufficient observations makes distance-based learning worth evaluating"
        )

    # Rule 84: Dataset suitable for ensemble boosting
    if n_samples >= 3000 and n_features >= 5 and feature_ratio >= 10:
        add_score(
            "Gradient Boosting", 2,
            "Dataset has sufficient observations relative to features for boosting evaluation"
        )

    # Rule 85: Moderate dataset size for AdaBoost
    if 500 <= n_samples <= 10000 and 5 <= n_features <= 100:
        add_score(
            "AdaBoost", 2,
            "Dataset size and dimensionality make AdaBoost a candidate for validation"
        )













def apply_dataset_rules(X, y, task_type, add_score):

    n_samples, n_features = X.shape

    if n_samples == 0 or n_features == 0:
        return

    feature_ratio = n_samples / n_features

    # Rule 1: Small dataset with manageable dimensions
    if n_samples < 500 and n_features <= 50:
        add_score(
            "SVM", 2,
            "Small dataset with manageable dimensionality makes SVM worth evaluating"
        )

    # Rule 2: Small dataset with enough samples per feature
    if n_samples < 1000 and feature_ratio >= 10:
        add_score(
            "Logistic Regression", 2,
            "Limited dataset size and manageable dimensionality support a simple baseline"
        )

    # Rule 3: Small, low-dimensional dataset
    if n_samples < 2000 and n_features <= 10 and feature_ratio >= 20:
        add_score(
            "KNN", 2,
            "Low-dimensional dataset makes distance-based classification worth evaluating"
        )

    # Rule 4: Small-to-medium dataset
    if 500 <= n_samples < 2000 and n_features <= 100:
        add_score(
            "SVM", 2,
            "Small-to-medium dataset makes SVM a candidate for validation"
        )

    # Rule 5: Moderate sample count and manageable dimensions
    if 500 <= n_samples < 2000 and feature_ratio >= 10:
        add_score(
            "Logistic Regression", 2,
            "Sufficient observations relative to features support a linear baseline"
        )

    # Rule 6: Small-to-medium dataset with multiple features
    if 500 <= n_samples < 2000 and n_features >= 5:
        add_score(
            "Random Forest", 2,
            "Tree ensembles are worth evaluating on this dataset size and feature count"
        )

    # Rule 7: Medium-sized dataset
    if 2000 <= n_samples < 10000 and n_features >= 5:
        add_score(
            "Random Forest", 2,
            "Medium-sized dataset makes Random Forest worth evaluating"
        )

    # Rule 8: Medium dataset with manageable dimensionality
    if 2000 <= n_samples < 10000 and feature_ratio >= 10:
        add_score(
            "Gradient Boosting", 2,
            "Dataset has sufficient observations relative to features for boosting evaluation"
        )

    # Rule 9: Medium-to-large dataset
    if 2000 <= n_samples < 10000 and n_features >= 10:
        add_score(
            "XGBoost", 2,
            "Medium-sized dataset with multiple features makes XGBoost a candidate"
        )

    # Rule 10: Large dataset
    if n_samples >= 10000 and n_features >= 10:
        add_score(
            "LightGBM", 2,
            "Large dataset makes scalable gradient boosting worth evaluating"
        )

    # Rule 11: Large dataset with manageable dimensionality
    if n_samples >= 10000 and feature_ratio >= 10:
        add_score(
            "XGBoost", 2,
            "Large dataset with sufficient observations per feature supports boosting evaluation"
        )

    # Rule 12: Large dataset suitable for histogram-based learning
    if n_samples >= 10000 and n_features >= 5:
        add_score(
            "HistGradientBoosting", 2,
            "Large dataset makes histogram-based gradient boosting a candidate"
        )

    # Rule 13: Very large dataset
    if n_samples >= 50000 and n_features >= 10:
        add_score(
            "LightGBM", 3,
            "Very large dataset makes efficient boosting worth evaluating"
        )

    # Rule 14: Very large dataset with manageable dimensions
    if n_samples >= 50000 and feature_ratio >= 10:
        add_score(
            "HistGradientBoosting", 3,
            "Very large dataset with sufficient observations per feature suits histogram-based evaluation"
        )

    # Rule 15: Enough observations for ensemble learning
    if n_samples >= 3000 and n_features >= 5 and feature_ratio >= 5:
        add_score(
            "Random Forest", 1,
            "Dataset has sufficient observations relative to feature count for ensemble evaluation"
        )








def apply_feature_rules(X, y, task_type, add_score):

    n_samples, n_features = X.shape

    if n_samples == 0 or n_features == 0:
        return

    feature_sample_ratio = n_features / n_samples
    samples_per_feature = n_samples / n_features

    # Rule 16: Very low-dimensional data
    if n_features <= 5 and samples_per_feature >= 10:
        add_score(
            "KNN", 2,
            "Low-dimensional data with sufficient samples makes KNN worth evaluating"
        )

    # Rule 17: Few features and sufficient observations
    if n_features <= 5 and n_samples >= 100:
        add_score(
            "Decision Tree", 2,
            "Small feature space makes a simple tree model worth evaluating"
        )

    # Rule 18: Low-dimensional data
    if n_features <= 10 and samples_per_feature >= 10:
        add_score(
            "Logistic Regression", 2,
            "Low-dimensional data with sufficient observations supports a linear baseline"
        )

    # Rule 19: Moderate feature count
    if 5 < n_features <= 50 and samples_per_feature >= 10:
        add_score(
            "Random Forest", 2,
            "Moderate feature count and sufficient observations make Random Forest a candidate"
        )

    # Rule 20: Moderate-dimensional data
    if 5 < n_features <= 50 and n_samples >= 500:
        add_score(
            "Gradient Boosting", 2,
            "Dataset size and moderate dimensionality support evaluating gradient boosting"
        )

    # Rule 21: High-dimensional data with manageable sample size
    if n_features > 50 and feature_sample_ratio >= 0.01:
        add_score(
            "SVM", 2,
            "High-dimensional feature space makes SVM worth evaluating"
        )

    # Rule 22: High-dimensional data
    if n_features > 50 and feature_sample_ratio >= 0.01:
        add_score(
            "Logistic Regression", 2,
            "High-dimensional data makes regularized Logistic Regression a candidate"
        )

    # Rule 23: Very high-dimensional data
    if n_features > 500 and samples_per_feature < 20:
        add_score(
            "SVM", 3,
            "Very high dimensionality relative to sample count makes SVM worth evaluating"
        )

    # Rule 24: Very high-dimensional data with limited observations
    if n_features > 500 and samples_per_feature < 20:
        add_score(
            "Logistic Regression", 3,
            "Regularized linear classification is worth evaluating in high dimensions"
        )

    # Rule 25: Features are numerous relative to observations
    if feature_sample_ratio > 0.10 and n_features >= 10:
        add_score(
            "SVM", 2,
            "High feature-to-sample ratio makes SVM a candidate"
        )

    # Rule 26: High feature-to-sample ratio
    if feature_sample_ratio > 0.10 and n_features >= 10:
        add_score(
            "Logistic Regression", 2,
            "High feature-to-sample ratio supports evaluating regularized linear classification"
        )

    # Rule 27: Extremely high feature-to-sample ratio
    if feature_sample_ratio > 0.50:
        add_score(
            "SVM", 2,
            "Features substantially outnumber observations, making SVM worth evaluating"
        )

    # Rule 28: Extreme dimensionality
    if feature_sample_ratio > 0.50:
        add_score(
            "Logistic Regression", 2,
            "Regularized linear classification is a candidate when features outnumber samples"
        )

    # Rule 29: Multiple features and sufficient observations
    if n_features >= 10 and n_samples >= 2000:
        add_score(
            "XGBoost", 2,
            "Dataset size and feature count support evaluating boosted trees"
        )

    # Rule 30: Multiple features and sufficient observations
    if n_features >= 10 and n_samples >= 2000:
        add_score(
            "Random Forest", 2,
            "Dataset size and feature count support evaluating tree ensembles"
        )










def apply_imbalance_rules(X, y, task_type, add_score):

    if task_type != "classification":
        return

    counts = y.value_counts()

    if len(counts) < 2:
        return

    n_samples = len(y)
    majority = counts.max()
    minority = counts.min()

    if minority == 0:
        return

    imbalance_ratio = majority / minority
    minority_ratio = minority / n_samples

    # Rule 66: Relatively balanced classes
    if imbalance_ratio < 1.20:
        add_score(
            "Logistic Regression", 2,
            "Relatively balanced classes make Logistic Regression a candidate"
        )

    # Rule 67: Balanced classes with sufficient observations
    if imbalance_ratio < 1.20 and n_samples >= 500:
        add_score(
            "Random Forest", 2,
            "Balanced classes and sufficient observations support tree-ensemble evaluation"
        )

    # Rule 68: Mild imbalance
    if 1.20 <= imbalance_ratio < 1.50:
        add_score(
            "Random Forest", 2,
            "Mild class imbalance makes Random Forest worth evaluating"
        )

    # Rule 69: Moderate imbalance
    if 1.50 <= imbalance_ratio < 3:
        add_score(
            "XGBoost", 2,
            "Moderate class imbalance makes boosted trees a candidate for validation"
        )

    # Rule 70: Moderate imbalance with enough minority samples
    if 1.50 <= imbalance_ratio < 3 and minority >= 30:
        add_score(
            "LightGBM", 2,
            "Moderate imbalance and adequate minority-class representation support boosting evaluation"
        )

    # Rule 71: Strong imbalance
    if 3 <= imbalance_ratio < 10:
        add_score(
            "XGBoost", 2,
            "Strong class imbalance warrants evaluating boosting with imbalance-aware metrics"
        )

    # Rule 72: Strong imbalance in a sufficiently large dataset
    if imbalance_ratio >= 3 and n_samples >= 1000:
        add_score(
            "LightGBM", 2,
            "Imbalanced classes and a larger dataset make scalable boosting worth evaluating"
        )

    # Rule 73: Imbalance with adequate minority representation
    if imbalance_ratio >= 3 and minority >= 30:
        add_score(
            "Random Forest", 2,
            "Tree ensembles are worth evaluating when minority-class examples are available"
        )

    # Rule 74: Severe imbalance
    if imbalance_ratio >= 5 and minority_ratio <= 0.10:
        add_score(
            "CatBoost", 2,
            "Severe class imbalance makes CatBoost worth evaluating with appropriate class weighting"
        )

    # Rule 75: Extreme imbalance and enough minority examples
    if imbalance_ratio >= 10 and minority >= 30:
        add_score(
            "XGBoost", 2,
            "Extreme imbalance with enough minority examples warrants evaluating imbalance-aware boosting"
        )




















def apply_missing_value_rules(X, y, task_type, add_score):

    import numpy as np

    n_rows, n_features = X.shape
    total_cells = n_rows * n_features

    if total_cells == 0:
        return

    missing_ratio = X.isnull().sum().sum() / total_cells

    column_missing_ratio = X.isnull().mean()
    affected_columns = (column_missing_ratio > 0).sum()
    affected_ratio = affected_columns / n_features

    # Rule 41: Some missing values exist
    if missing_ratio > 0:
        add_score(
            "CatBoost", 1,
            "Missing values are present; evaluate CatBoost's missing-value handling"
        )

    # Rule 42: Missing values in a large dataset
    if missing_ratio > 0 and n_rows >= 5000:
        add_score(
            "LightGBM", 2,
            "Large dataset with missing values makes LightGBM worth evaluating"
        )

    # Rule 43: Missing values in numerical features
    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns

    if len(numeric_columns) > 0 and X[numeric_columns].isnull().any().any():
        add_score(
            "HistGradientBoosting", 2,
            "Numerical features contain missing values suitable for testing histogram-based boosting"
        )

    # Rule 44: Meaningful overall missingness
    if missing_ratio >= 0.01:
        add_score(
            "CatBoost", 2,
            "At least 1% of feature cells are missing"
        )

    # Rule 45: Moderate missingness
    if missing_ratio >= 0.05:
        add_score(
            "HistGradientBoosting", 2,
            "At least 5% of feature cells are missing; test native missing-value handling"
        )

    # Rule 46: Missing values affect multiple columns
    if affected_ratio >= 0.30:
        add_score(
            "CatBoost", 2,
            "Missing values occur across at least 30% of feature columns"
        )

    # Rule 47: Substantial missingness
    if missing_ratio >= 0.10:
        add_score(
            "LightGBM", 2,
            "Substantial missingness makes missing-value-aware boosting worth evaluating"
        )

    # Rule 48: Many cells are missing
    if missing_ratio >= 0.20 and affected_ratio >= 0.30:
        add_score(
            "CatBoost", 2,
            "High missingness across multiple columns warrants evaluating CatBoost"
        )

    # Rule 49: Complete dataset
    if missing_ratio == 0:
        add_score(
            "Random Forest", 1,
            "No missing values simplify preprocessing for Random Forest"
        )

    # Rule 50: Complete dataset with manageable dimensionality
    if missing_ratio == 0 and n_features <= 100:
        add_score(
            "Logistic Regression", 1,
            "Complete data and manageable dimensionality support evaluating Logistic Regression"
        )















def apply_model_specific_rules(X, y, task_type, add_score):

    n_samples, n_features = X.shape

    if n_samples == 0 or n_features == 0:
        return

    categorical_count = len(
        X.select_dtypes(
            include=["object", "category", "bool", "string"]
        ).columns
    )

    numeric_count = len(
        X.select_dtypes(include=["number"]).columns
    )

    feature_ratio = n_samples / n_features

    # Rule 86: Multiple categorical features
    if categorical_count >= 5:
        add_score(
            "CatBoost", 2,
            "Multiple categorical features make categorical-aware boosting worth evaluating"
        )

    # Rule 87: Mixed feature types
    if categorical_count > 0 and numeric_count > 0:
        add_score(
            "CatBoost", 2,
            "Mixed numerical and categorical features make CatBoost a candidate"
        )

    # Rule 88: Large dataset
    if n_samples >= 10000 and n_features >= 5:
        add_score(
            "LightGBM", 2,
            "Large dataset makes scalable gradient boosting worth evaluating"
        )

    # Rule 89: Large dataset with multiple features
    if n_samples >= 10000 and n_features >= 20 and feature_ratio >= 10:
        add_score(
            "XGBoost", 2,
            "Large dataset with multiple features and sufficient samples per feature supports boosting evaluation"
        )

    # Rule 90: High-dimensional data
    if n_features > 100 and feature_ratio < 20:
        add_score(
            "SVM", 2,
            "High dimensionality relative to sample count makes SVM worth evaluating"
        )

    # Rule 91: High-dimensional data with regularization potential
    if n_features > 100 and feature_ratio < 20:
        add_score(
            "Logistic Regression", 2,
            "Regularized Logistic Regression is worth evaluating for high-dimensional data"
        )

    # Rule 92: Low-dimensional data
    if n_features <= 10 and n_samples <= 5000 and feature_ratio >= 10:
        add_score(
            "KNN", 2,
            "Low-dimensional data with sufficient observations makes KNN a candidate"
        )

    # Rule 93: Limited feature count
    if n_features <= 20 and n_samples >= 100:
        add_score(
            "Decision Tree", 2,
            "Limited feature count makes an interpretable tree model worth evaluating"
        )

    # Rule 94: Larger dataset with multiple features
    if n_samples >= 5000 and n_features >= 10 and feature_ratio >= 5:
        add_score(
            "MLP", 2,
            "Dataset size and dimensionality make nonlinear neural classification worth evaluating"
        )

    # Rule 95: Numerical classification
    if task_type == "classification" and numeric_count >= 2:
        add_score(
            "Naive Bayes", 1,
            "Numerical classification data makes a suitable Naive Bayes variant worth evaluating"
        )

    # Rule 96: Low-dimensional numerical classification
    if (
        task_type == "classification"
        and numeric_count == n_features
        and 2 <= n_features <= 20
        and feature_ratio >= 10
    ):
        add_score(
            "LDA", 2,
            "Low-dimensional numerical data makes Linear Discriminant Analysis worth evaluating"
        )

    # Rule 97: Low-dimensional numerical classification
    if (
        task_type == "classification"
        and numeric_count == n_features
        and 2 <= n_features <= 20
        and feature_ratio >= 10
    ):
        add_score(
            "QDA", 1,
            "Low-dimensional numerical data makes Quadratic Discriminant Analysis worth evaluating"
        )

    # Rule 98: Dataset size for ensemble learning
    if task_type == "classification" and n_samples >= 1000:
        add_score(
            "AdaBoost", 1,
            "Dataset size makes AdaBoost a candidate for validation"
        )

    # Rule 99: Histogram-based boosting
    if (
        task_type == "classification"
        and n_samples >= 2000
        and n_features >= 5
    ):
        add_score(
            "HistGradientBoosting", 2,
            "Dataset size and feature count support evaluating histogram-based boosting"
        )

    # Rule 100: General-purpose tree ensemble
    if task_type == "classification" and n_samples >= 500:
        add_score(
            "Random Forest", 1,
            "Random Forest is a general-purpose candidate for classification"
        )



# Model list used by run.py and the evaluation module.
MODELS = [
    "Decision Tree", "Random Forest", "Naive Bayes", "Logistic Regression",
    "KNN", "SVM", "LDA", "QDA", "AdaBoost", "Gradient Boosting",
    "HistGradientBoosting", "XGBoost", "CatBoost", "LightGBM", "MLP"
]

RULE_FUNCTIONS = [
    apply_dataset_rules,
    apply_feature_rules,
    apply_categorical_rules,
    apply_missing_value_rules,
    apply_classification_rules,
    apply_imbalance_rules,
    apply_complexity_rules,
    apply_model_specific_rules,
]

def apply_all_model_rules(X, y, task_type="classification", add_score=None):
    """
    Execute the original rules 1-100 using their original function definitions.
    Returns (scores, reasons), compatible with run.py.
    """
    from collections import defaultdict

    scores = defaultdict(float)
    reasons = defaultdict(list)

    def collect(model, points, reason):
        scores[model] += points
        reasons[model].append(reason)
        if add_score is not None:
            add_score(model, points, reason)

    for rule_function in RULE_FUNCTIONS:
        rule_function(X, y, task_type, collect)

    # Include all named candidate models in the score table, even if score is 0.
    for model in MODELS:
        scores[model] += 0

    return dict(scores), dict(reasons)


# ---------------------------------------------------------------------------
# ADDITIONAL RULES 101-160
# The original rules 1-100 above are intentionally left unchanged.
# These additional rules use separate evidence signals and remain heuristics.
# ---------------------------------------------------------------------------
def apply_additional_rules(X, y, task_type, add_score):
    import numpy as np
    import pandas as pd

    if X is None or y is None or len(X) == 0 or X.shape[1] == 0:
        return
    n_samples, n_features = X.shape
    n_classes = y.nunique(dropna=True)
    numeric = X.select_dtypes(include=["number"]).copy()
    categorical = X.select_dtypes(include=["object", "category", "bool", "string"]).copy()
    numeric_count, categorical_count = len(numeric.columns), len(categorical.columns)
    sample_feature_ratio = n_samples / max(n_features, 1)
    class_counts = y.value_counts(dropna=True)
    min_class_count = int(class_counts.min()) if len(class_counts) else 0
    max_class_count = int(class_counts.max()) if len(class_counts) else 0
    imbalance_ratio = max_class_count / max(min_class_count, 1)
    missing_fraction = float(X.isna().to_numpy().mean()) if X.size else 0.0
    unique_counts = X.nunique(dropna=True)
    constant_count = int((unique_counts <= 1).sum())
    duplicate_row_fraction = float(X.duplicated().mean()) if n_samples > 1 else 0.0
    numeric_skew = numeric.skew(numeric_only=True).replace([np.inf, -np.inf], np.nan).dropna()
    high_skew_fraction = float((numeric_skew.abs() > 1.0).mean()) if len(numeric_skew) else 0.0
    high_corr_fraction = 0.0
    if numeric_count >= 2:
        corr = numeric.corr().abs().to_numpy()
        upper = corr[np.triu_indices_from(corr, k=1)]
        high_corr_fraction = float(np.mean(upper >= 0.90)) if len(upper) else 0.0
    binary_numeric_count = sum(numeric[c].dropna().nunique() <= 2 for c in numeric.columns)
    low_cardinality_numeric_count = sum(2 < numeric[c].dropna().nunique() <= 10 for c in numeric.columns)
    high_cardinality_categorical_count = sum(categorical[c].dropna().nunique() >= 50 for c in categorical.columns)
    numeric_outlier_fraction = 0.0
    if numeric_count:
        outlier_cols = 0
        valid_cols = 0
        for col in numeric.columns:
            vals = numeric[col].dropna()
            if len(vals) < 4:
                continue
            q1, q3 = vals.quantile([0.25, 0.75])
            iqr = q3 - q1
            valid_cols += 1
            if iqr > 0 and ((vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)).mean() > 0.05:
                outlier_cols += 1
        numeric_outlier_fraction = outlier_cols / max(valid_cols, 1)
    class_count_ratio = min_class_count / max(n_samples, 1)
    is_classification = task_type == "classification"

    # Rule 101: Constant columns should be removed before model fitting.
    if constant_count > 0:
        add_score("Decision Tree", 1, "Constant feature columns were detected; evaluate a tree after feature cleanup")

    # Rule 102: Many constant columns make feature selection important.
    if n_features >= 5 and constant_count / n_features >= 0.20:
        add_score("Random Forest", 1, "At least one fifth of the features are constant, so feature selection should be checked")

    # Rule 103: Duplicate observations may affect validation reliability.
    if duplicate_row_fraction >= 0.10:
        add_score("Decision Tree", 1, "Repeated feature rows are common; use duplicate-aware validation")

    # Rule 104: Very small minority class favors simple baselines for comparison.
    if is_classification and min_class_count < 10:
        add_score("Logistic Regression", 1, "The smallest class has fewer than 10 examples; include a simple baseline and stratified validation")

    # Rule 105: Very small minority class can make flexible neighbors unstable.
    if is_classification and min_class_count < 10:
        add_score("Naive Bayes", 1, "Few minority-class examples make a low-complexity probabilistic baseline worth testing")

    # Rule 106: A large number of classes raises the need for multiclass evaluation.
    if is_classification and n_classes >= 10:
        add_score("Random Forest", 2, "The target has at least 10 classes; evaluate a multiclass-capable ensemble")

    # Rule 107: Very high class count relative to samples is a difficult setting.
    if is_classification and n_classes >= 5 and n_samples / max(n_classes, 1) < 20:
        add_score("Naive Bayes", 1, "There are relatively few observations per class; compare a compact probabilistic model")

    # Rule 108: Many numeric binary indicators can suit linear decision boundaries.
    if numeric_count >= 3 and binary_numeric_count / max(numeric_count, 1) >= 0.60:
        add_score("Logistic Regression", 2, "Most numeric columns are binary indicators, making a linear classifier worth testing")

    # Rule 109: Low-cardinality numeric data may suit shallow trees.
    if numeric_count >= 3 and low_cardinality_numeric_count / max(numeric_count, 1) >= 0.60:
        add_score("Decision Tree", 2, "Most numeric columns have at most 10 distinct values, which may support interpretable tree splits")

    # Rule 110: Strongly skewed numeric columns suggest testing robust tree splits.
    if high_skew_fraction >= 0.50:
        add_score("Random Forest", 2, "At least half of numeric features are strongly skewed; compare a tree ensemble")

    # Rule 111: Strongly skewed numeric columns can also favor boosting validation.
    if high_skew_fraction >= 0.50 and n_samples >= 300:
        add_score("Gradient Boosting", 1, "Skewed numeric distributions warrant testing a boosting model")

    # Rule 112: Potential numeric outliers favor models less dependent on distance scales.
    if numeric_outlier_fraction >= 0.30:
        add_score("Random Forest", 2, "Several numeric columns contain many IQR-based outliers; tree ensembles are worth comparing")

    # Rule 113: Outliers make unscaled distance-based methods less attractive without preprocessing.
    if numeric_outlier_fraction >= 0.30 and numeric_count >= 2:
        add_score("Decision Tree", 1, "Outlier-heavy numeric features suggest testing a tree baseline before distance-based models")

    # Rule 114: Highly correlated numeric features make regularized linear models worth checking.
    if high_corr_fraction >= 0.25:
        add_score("Logistic Regression", 2, "Many numeric feature pairs are highly correlated; compare a regularized linear classifier")

    # Rule 115: Correlation-heavy data can challenge discriminant covariance estimates.
    if high_corr_fraction >= 0.40 and numeric_count >= 3:
        add_score("Random Forest", 1, "Strong numeric feature correlations make a tree ensemble a useful comparison")

    # Rule 116: Wide datasets with few rows need careful capacity control.
    if n_features > n_samples:
        add_score("Logistic Regression", 2, "There are more features than rows; a regularized linear model is worth evaluating")

    # Rule 117: Wide datasets also justify a margin-based classifier after scaling.
    if n_features > n_samples and numeric_count >= 2:
        add_score("SVM", 2, "The dataset is wider than it is tall; compare an SVM with suitable scaling and regularization")

    # Rule 118: Many samples per feature can support nonlinear model evaluation.
    if sample_feature_ratio >= 100 and n_samples >= 1000:
        add_score("Gradient Boosting", 2, "There are at least 100 observations per feature and 1,000 rows; evaluate boosting")

    # Rule 119: A broad dataset with many features supports bagged feature sampling.
    if n_samples >= 2000 and n_features >= 30:
        add_score("Random Forest", 2, "A large dataset with 30 or more features is suitable for testing bagged feature sampling")

    # Rule 120: Very small datasets call for low-capacity candidates and careful CV.
    if n_samples < 100:
        add_score("Naive Bayes", 2, "Fewer than 100 rows are available; include a compact probabilistic baseline")

    # Rule 121: Very small datasets call for a simple interpretable baseline too.
    if n_samples < 100 and n_features <= 20:
        add_score("Decision Tree", 1, "A small dataset with limited features supports checking a constrained interpretable tree")

    # Rule 122: Medium-sized datasets support testing nonlinear kernel boundaries.
    if is_classification and 200 <= n_samples <= 5000 and numeric_count >= 2:
        add_score("SVM", 1, "Dataset size and numeric features make a scaled SVM a candidate for validation")

    # Rule 123: Many categorical columns suggest comparing categorical-aware boosting.
    if categorical_count >= 8:
        add_score("CatBoost", 3, "At least eight categorical columns are present; compare categorical-aware boosting")

    # Rule 124: Very high-cardinality categorical columns warrant CatBoost evaluation.
    if high_cardinality_categorical_count >= 2:
        add_score("CatBoost", 2, "At least two categorical columns have 50 or more distinct values")

    # Rule 125: Very high-cardinality categories also warrant a preprocessing-aware baseline.
    if high_cardinality_categorical_count >= 2:
        add_score("Random Forest", 1, "High-cardinality categorical columns need careful encoding; compare a tree ensemble after preprocessing")

    # Rule 126: Mostly categorical data supports evaluating boosted trees after proper handling.
    if categorical_count >= 1 and categorical_count / max(n_features, 1) >= 0.70:
        add_score("CatBoost", 2, "At least 70% of features are categorical, so CatBoost is a strong candidate to test")

    # Rule 127: Mixed types and moderate row count suggest checking CatBoost.
    if categorical_count > 0 and numeric_count > 0 and n_samples >= 200:
        add_score("CatBoost", 1, "Mixed numeric and categorical columns with at least 200 rows warrant a CatBoost comparison")

    # Rule 128: Missingness concentrated in many columns makes imputation pipeline comparison important.
    missing_by_col = X.isna().mean()
    if len(missing_by_col) and (missing_by_col >= 0.20).sum() >= 3:
        add_score("HistGradientBoosting", 1, "At least three columns have 20% or more missing values; evaluate missing-value handling carefully")

    # Rule 129: Very low overall missingness supports a standard baseline pipeline.
    if 0 < missing_fraction <= 0.01:
        add_score("Logistic Regression", 1, "Overall missingness is at most 1%; a standard imputation and linear-model pipeline is worth testing")

    # Rule 130: High overall missingness calls for explicit preprocessing validation.
    if missing_fraction >= 0.25:
        add_score("HistGradientBoosting", 2, "At least 25% of feature cells are missing; evaluate a model with native missing-value handling")

    # Rule 131: Sparse one-hot-style data is often a useful linear-model setting.
    if numeric_count >= 5:
        zero_fractions = [(numeric[c].fillna(0) == 0).mean() for c in numeric.columns]
        if np.mean(zero_fractions) >= 0.70:
            add_score("Logistic Regression", 2, "Numeric features are mostly zero-valued, suggesting a sparse linear baseline after encoding")

    # Rule 132: Sparse features also warrant an SVM comparison.
    if numeric_count >= 5:
        zero_fractions = [(numeric[c].fillna(0) == 0).mean() for c in numeric.columns]
        if np.mean(zero_fractions) >= 0.70:
            add_score("SVM", 1, "Mostly zero-valued numeric features warrant testing a scaled or sparse-aware SVM pipeline")

    # Rule 133: Low class imbalance and enough data supports a multiclass linear baseline.
    if is_classification and n_classes >= 3 and imbalance_ratio < 1.5 and n_samples >= 300:
        add_score("Logistic Regression", 1, "A reasonably balanced multiclass target supports testing multinomial logistic regression")

    # Rule 134: Severe imbalance suggests class-weight-aware linear comparison.
    if is_classification and imbalance_ratio >= 5 and numeric_count >= 2:
        add_score("Logistic Regression", 1, "Class counts are uneven; test class weighting and threshold tuning with logistic regression")

    # Rule 135: Severe imbalance also suggests comparing a balanced tree ensemble.
    if is_classification and imbalance_ratio >= 5 and n_samples >= 300:
        add_score("Random Forest", 1, "Class imbalance is substantial; compare balanced sampling or class-weighted tree ensembles")

    # Rule 136: Tiny minority support makes per-class metrics more important than accuracy.
    if is_classification and class_count_ratio < 0.02:
        add_score("Naive Bayes", 1, "The smallest class is under 2% of rows; assess macro-F1 and per-class recall")

    # Rule 137: Balanced binary targets make probabilistic linear baselines informative.
    if is_classification and n_classes == 2 and imbalance_ratio <= 1.2 and n_samples >= 100:
        add_score("Logistic Regression", 1, "The binary target is close to balanced; logistic regression offers a useful probability baseline")

    # Rule 138: Binary targets with nonlinear feature interactions justify boosted trees.
    if is_classification and n_classes == 2 and n_features >= 5 and n_samples >= 500:
        add_score("XGBoost", 1, "A binary target with enough rows and features warrants comparing gradient-boosted trees")

    # Rule 139: Multiclass targets and adequate rows support boosted trees.
    if is_classification and n_classes >= 3 and n_samples >= 500:
        add_score("LightGBM", 1, "Multiclass data with at least 500 rows warrants testing LightGBM where installed")

    # Rule 140: Moderate-size multiclass data also supports general gradient boosting.
    if is_classification and n_classes >= 3 and n_samples >= 300 and n_features >= 5:
        add_score("Gradient Boosting", 1, "Multiclass data with several features warrants comparing gradient boosting")

    # Rule 141: High feature count can make neural nonlinear combinations worth testing.
    if n_features >= 50 and n_samples >= 2000:
        add_score("MLP", 1, "Many features and at least 2,000 rows provide a basis for testing a regularized multilayer perceptron")

    # Rule 142: A relatively large dataset supports neural-network evaluation after scaling.
    if n_samples >= 5000 and numeric_count >= 5:
        add_score("MLP", 2, "At least 5,000 rows and five numeric features support testing a scaled, regularized MLP")

    # Rule 143: Low feature count and enough samples can make KNN a useful benchmark.
    if n_features <= 10 and n_samples >= 200:
        add_score("KNN", 1, "A feature count of 10 or fewer makes a scaled KNN benchmark practical")

    # Rule 144: KNN should be checked only with enough rows per class.
    if is_classification and min_class_count >= 30 and n_features <= 20:
        add_score("KNN", 1, "Each class has at least 30 rows and feature count is modest; compare KNN after scaling")

    # Rule 145: A high feature count makes distance concentration a concern.
    if n_features >= 100:
        add_score("SVM", 1, "High-dimensional data warrants comparing margin-based classification against distance-based methods")

    # Rule 146: Moderate dimensions and mostly numeric features make LDA worth validating.
    if is_classification and numeric_count == n_features and 2 <= n_features <= 15 and n_samples >= 100:
        add_score("LDA", 1, "Moderate-dimensional numeric features make LDA worth testing as a linear discriminant baseline")

    # Rule 147: A small number of numeric features can support QDA if class counts suffice.
    if is_classification and numeric_count == n_features and 2 <= n_features <= 8 and min_class_count >= 20:
        add_score("QDA", 1, "A small all-numeric feature set with at least 20 rows per class makes QDA worth evaluating")

    # Rule 148: Very high dimensionality relative to rows suggests avoiding unconstrained QDA.
    if is_classification and numeric_count == n_features and n_features >= min_class_count:
        add_score("Logistic Regression", 1, "Feature count is at least as large as the smallest class size; compare regularized linear classification")

    # Rule 149: More than two classes and all numeric columns support multinomial logistic regression.
    if is_classification and n_classes > 2 and numeric_count == n_features and n_samples >= 200:
        add_score("Logistic Regression", 1, "All features are numeric and the target is multiclass; compare multinomial logistic regression")

    # Rule 150: Nonlinear relationships may be captured by boosted tree interactions.
    if n_features >= 5 and n_samples >= 1000:
        add_score("XGBoost", 1, "A dataset with several features and 1,000 rows warrants testing nonlinear boosted-tree interactions")

    # Rule 151: Dataset size supports histogram-based gradient boosting.
    if n_samples >= 10000 and n_features >= 5:
        add_score("HistGradientBoosting", 2, "At least 10,000 rows and five features make histogram-based boosting worth benchmarking")

    # Rule 152: Medium-sized numeric data can support LightGBM benchmarking.
    if numeric_count >= 5 and n_samples >= 1000:
        add_score("LightGBM", 1, "Numeric feature data with at least 1,000 rows warrants benchmarking LightGBM where installed")

    # Rule 153: Enough rows and columns support CatBoost comparison even with numeric data.
    if n_samples >= 1000 and n_features >= 10:
        add_score("CatBoost", 1, "Dataset scale and feature count warrant including CatBoost in the benchmark when available")

    # Rule 154: Modest data can benefit from AdaBoost as a compact ensemble comparison.
    if 100 <= n_samples < 2000 and n_features <= 50:
        add_score("AdaBoost", 1, "A modest dataset is suitable for comparing AdaBoost against a single decision tree")

    # Rule 155: Few numeric features and low correlation can support simple discriminant baselines.
    if is_classification and 2 <= numeric_count <= 10 and high_corr_fraction < 0.10:
        add_score("LDA", 1, "Numeric features have few highly correlated pairs; include LDA as a compact baseline")

    # Rule 156: A high proportion of categorical values with limited cardinality can suit encoded linear models.
    if categorical_count > 0 and all(categorical[c].dropna().nunique() <= 10 for c in categorical.columns):
        add_score("Logistic Regression", 1, "Categorical columns have at most 10 levels each; one-hot encoding plus logistic regression is worth testing")

    # Rule 157: High-cardinality labels are a warning for leakage-prone identifier columns.
    identifier_like = 0
    for col in X.columns:
        non_null = X[col].dropna()
        if len(non_null) >= 20 and non_null.nunique() / len(non_null) >= 0.95:
            identifier_like += 1
    if identifier_like > 0:
        add_score("Decision Tree", 1, "Identifier-like columns were detected; remove or validate them to avoid misleading model scores")

    # Rule 158: Numeric feature scale variation suggests explicit scaling before distance or margin models.
    if numeric_count >= 2:
        scales = numeric.std(numeric_only=True).replace(0, np.nan).dropna()
        if len(scales) >= 2 and scales.max() / max(scales.min(), 1e-12) >= 1000:
            add_score("SVM", 1, "Numeric feature standard deviations vary greatly; compare SVM only with a scaling pipeline")

    # Rule 159: Large scale differences also warrant a scaled KNN benchmark.
    if numeric_count >= 2:
        scales = numeric.std(numeric_only=True).replace(0, np.nan).dropna()
        if len(scales) >= 2 and scales.max() / max(scales.min(), 1e-12) >= 1000:
            add_score("KNN", 1, "Feature scales differ substantially; a standardized KNN pipeline is worth comparing")

    # Rule 160: Many feature columns relative to rows calls for regularization and careful CV.
    if n_features >= 30 and sample_feature_ratio < 5:
        add_score("Logistic Regression", 2, "There are fewer than five rows per feature; prioritize regularization and careful cross-validation")


# Add the new rules after the original rules without modifying rules 1-100.
RULE_FUNCTIONS = RULE_FUNCTIONS + [apply_additional_rules]



# ============================================================================
# REGRESSION_RULES_START
# Separate regression recommendation engine.
# The existing classification rules 1-160 above are intentionally unchanged.
# ============================================================================

REGRESSION_MODELS = [
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "Elastic Net",
    "Decision Tree Regressor",
    "Random Forest Regressor",
    "Extra Trees Regressor",
    "KNN Regressor",
    "Support Vector Regressor",
    "Gradient Boosting Regressor",
    "HistGradientBoosting Regressor",
    "XGBoost Regressor",
    "CatBoost Regressor",
    "LightGBM Regressor",
    "MLP Regressor",
]

def apply_regression_dataset_rules(X, y, add_score):
    """Regression Rules R1-R15: dataset size and dimensionality."""
    n_samples, n_features = X.shape
    if n_samples == 0 or n_features == 0:
        return
    ratio = n_samples / max(n_features, 1)

    # R1
    if n_samples < 500 and n_features <= 50:
        add_score("Ridge Regression", 2, "Small dataset: compare a regularized linear baseline")
    # R2
    if n_samples < 500 and n_features <= 20:
        add_score("Decision Tree Regressor", 1, "Small dataset: an interpretable tree is worth testing")
    # R3
    if 500 <= n_samples < 2000 and ratio >= 10:
        add_score("Linear Regression", 2, "Moderate sample-to-feature ratio supports a linear baseline")
    # R4
    if 500 <= n_samples < 5000 and n_features >= 5:
        add_score("Random Forest Regressor", 2, "Moderate dataset size supports evaluating a tree ensemble")
    # R5
    if n_samples >= 2000 and n_features >= 5:
        add_score("Gradient Boosting Regressor", 2, "Dataset size supports evaluating gradient boosting")
    # R6
    if n_samples >= 5000 and n_features >= 5:
        add_score("HistGradientBoosting Regressor", 2, "Larger dataset may benefit from histogram-based boosting")
    # R7
    if n_samples >= 10000 and n_features >= 5:
        add_score("LightGBM Regressor", 2, "Large dataset makes scalable boosting worth benchmarking")
    # R8
    if n_samples >= 10000 and n_features >= 10:
        add_score("XGBoost Regressor", 2, "Large dataset with multiple features supports boosted-tree evaluation")
    # R9
    if n_samples >= 1000 and n_features >= 10:
        add_score("CatBoost Regressor", 1, "Dataset size and feature count support benchmarking CatBoost")
    # R10
    if n_features > n_samples:
        add_score("Lasso Regression", 3, "More features than rows makes regularization and feature shrinkage important")
    # R11
    if ratio >= 100 and n_samples >= 1000:
        add_score("Linear Regression", 1, "Many observations per feature support a simple linear baseline")
    # R12
    if n_samples < 100:
        add_score("Ridge Regression", 2, "Very small dataset warrants a regularized baseline and careful validation")
    # R13
    if n_samples >= 300 and n_features <= 10:
        add_score("KNN Regressor", 1, "A modest feature space makes KNN regression worth comparing after scaling")
    # R14
    if n_samples >= 1000 and n_features >= 30:
        add_score("Extra Trees Regressor", 2, "Larger dataset with many features supports randomized tree ensembles")
    # R15
    if n_samples >= 200 and n_features <= 100:
        add_score("Support Vector Regressor", 1, "Dataset dimensions make SVR a candidate if scaling and runtime are managed")


def apply_regression_feature_rules(X, y, add_score):
    """Regression Rules R16-R30: feature count, scale, sparsity and redundancy."""
    import numpy as np
    n_samples, n_features = X.shape
    if n_samples == 0 or n_features == 0:
        return
    numeric = X.select_dtypes(include=["number"])
    categorical = X.select_dtypes(include=["object", "category", "bool", "string"])
    if n_features <= 10 and n_samples >= 100:
        add_score("Linear Regression", 2, "Low-dimensional features make a linear baseline interpretable")
    if n_features >= 50:
        add_score("Random Forest Regressor", 1, "Many features warrant comparison with a tree ensemble")
    if n_features > n_samples:
        add_score("Ridge Regression", 2, "High-dimensional setting favors regularization")
    if n_features > n_samples and n_features >= 20:
        add_score("Lasso Regression", 2, "Feature count exceeds sample count; sparse coefficients may be useful")
    if len(numeric.columns) >= 2:
        stds = numeric.std(numeric_only=True).replace(0, np.nan).dropna()
        if len(stds) >= 2 and stds.max() / max(stds.min(), 1e-12) >= 1000:
            add_score("Support Vector Regressor", 1, "Feature scales vary greatly; SVR should be tested in a scaling pipeline")
            add_score("KNN Regressor", 1, "Feature scales vary greatly; KNN should be tested after standardization")
        corr = numeric.corr().abs().to_numpy()
        if corr.shape[0] >= 2:
            upper = corr[np.triu_indices_from(corr, k=1)]
            high_corr = float(np.mean(upper >= 0.90)) if len(upper) else 0.0
            if high_corr >= 0.20:
                add_score("Ridge Regression", 2, "Highly correlated numeric features make regularized linear regression worth testing")
                add_score("Lasso Regression", 1, "Correlated features warrant checking coefficient shrinkage and stability")
    if len(numeric.columns) >= 5:
        zero_fracs = [(numeric[c].fillna(0) == 0).mean() for c in numeric.columns]
        if float(np.mean(zero_fracs)) >= 0.70:
            add_score("Ridge Regression", 1, "Mostly zero-valued numeric features warrant a regularized linear baseline")
    if len(categorical.columns) > 0:
        add_score("CatBoost Regressor", 2, "Categorical features are present; use a compatible categorical-handling pipeline")
    if len(categorical.columns) > 0 and len(numeric.columns) > 0:
        add_score("CatBoost Regressor", 1, "Mixed numeric and categorical features warrant categorical-aware boosting")
    if len(categorical.columns) >= 5:
        add_score("CatBoost Regressor", 2, "Several categorical columns make CatBoost worth benchmarking")
    if n_features <= 5 and n_samples >= 100:
        add_score("Linear Regression", 1, "Small feature set supports an interpretable linear baseline")
    if n_features <= 20 and n_samples >= 500:
        add_score("Decision Tree Regressor", 1, "Moderate data with limited features supports an interpretable tree benchmark")
    if n_features >= 10 and n_samples >= 1000:
        add_score("XGBoost Regressor", 1, "Multiple features and sufficient rows support boosted-tree evaluation")
    if n_features >= 10 and n_samples >= 1000:
        add_score("MLP Regressor", 1, "Nonlinear neural regression can be compared when scaling and regularization are applied")
    if n_features >= 100 and n_samples >= 1000:
        add_score("Extra Trees Regressor", 2, "High feature count and sufficient data support randomized tree ensembles")
    if n_features <= 20 and n_samples >= 200:
        add_score("KNN Regressor", 1, "Moderate feature count supports KNN benchmarking after scaling")


def apply_regression_target_rules(X, y, add_score):
    """Regression Rules R31-R45: target distribution and target quality."""
    import numpy as np
    target = y.dropna()
    if len(target) < 2:
        return
    nunique = target.nunique()
    if nunique <= 1:
        return
    skew = float(target.skew()) if target.dtype.kind in "iuf" else 0.0
    target_std = float(target.std()) if target.dtype.kind in "iuf" else 0.0
    target_mean = float(target.mean()) if target.dtype.kind in "iuf" else 0.0
    target_cv = abs(target_std / target_mean) if target_mean != 0 else np.inf
    # R31
    if nunique <= 10:
        add_score("Decision Tree Regressor", 1, "Target has few distinct values; inspect whether regression is the intended task")
    # R32
    if nunique >= 20:
        add_score("Random Forest Regressor", 1, "Target has many distinct numeric values, consistent with a continuous target")
    # R33
    if abs(skew) >= 1:
        add_score("Random Forest Regressor", 1, "Target is strongly skewed; compare a nonlinear tree ensemble")
    # R34
    if abs(skew) >= 1 and len(target) >= 500:
        add_score("Gradient Boosting Regressor", 1, "Skewed target distribution warrants evaluating boosting and suitable target transforms")
    # R35
    if target_cv >= 1 and np.isfinite(target_cv):
        add_score("Random Forest Regressor", 1, "Target variability is large relative to its mean; compare nonlinear models")
    # R36
    if len(target) >= 100 and target_std > 0:
        add_score("Ridge Regression", 1, "Nonconstant numeric target permits a regularized linear baseline")
    # R37
    if len(target) >= 1000 and nunique >= 50:
        add_score("HistGradientBoosting Regressor", 1, "Many distinct target values and sufficient rows support histogram boosting")
    # R38
    if len(target) < 100:
        add_score("Ridge Regression", 1, "Few target observations call for simple baselines and careful validation")
    # R39
    if target.isna().any():
        add_score("Ridge Regression", 1, "Target contains missing values; remove or resolve missing target rows before fitting")
    # R40
    if target_std == 0:
        add_score("Linear Regression", 1, "Constant target detected; verify the target and task before model training")
    # R41
    if target.dtype.kind in "iuf" and target.min() >= 0 and skew > 1:
        add_score("Gradient Boosting Regressor", 1, "Nonnegative, right-skewed target warrants comparing transformed-target and boosting approaches")
    # R42
    if target.dtype.kind in "iuf" and target.min() < 0 and target.max() > 0:
        add_score("Linear Regression", 1, "Target spans negative and positive values; a linear baseline can handle signed targets")
    # R43
    if len(target) >= 300 and target.nunique() >= 30:
        add_score("Support Vector Regressor", 1, "Continuous target and moderate sample size warrant testing SVR after scaling")
    # R44
    if len(target) >= 1000 and target.nunique() >= 30:
        add_score("XGBoost Regressor", 1, "Continuous target with adequate data supports boosted regression benchmarking")
    # R45
    if len(target) >= 500 and target.nunique() >= 30:
        add_score("CatBoost Regressor", 1, "Target variability and sample count support testing CatBoost regression")


def apply_regression_relationship_rules(X, y, add_score):
    """Regression Rules R46-R60: linearity, correlations and interactions."""
    import numpy as np
    numeric = X.select_dtypes(include=["number"])
    if len(numeric.columns) >= 1 and len(y) == len(X):
        aligned = pd_numeric = numeric.copy()
        target = y.reset_index(drop=True)
        aligned = aligned.reset_index(drop=True)
        if target.dtype.kind in "iuf":
            correlations = aligned.corrwith(target, method="pearson").replace([np.inf, -np.inf], np.nan).dropna()
            if len(correlations):
                max_abs = float(correlations.abs().max())
                strong_fraction = float((correlations.abs() >= 0.70).mean())
                weak_fraction = float((correlations.abs() < 0.10).mean())
                # R46
                if max_abs >= 0.70:
                    add_score("Linear Regression", 2, "At least one numeric feature has a strong linear target correlation")
                # R47
                if strong_fraction >= 0.30:
                    add_score("Ridge Regression", 2, "Several numeric features correlate strongly with the target; compare regularized linear models")
                # R48
                if weak_fraction >= 0.70:
                    add_score("Random Forest Regressor", 1, "Most individual linear correlations are weak; nonlinear relationships may still exist")
                # R49
                if strong_fraction < 0.10 and len(numeric.columns) >= 5:
                    add_score("Gradient Boosting Regressor", 1, "Weak individual correlations warrant testing models that can capture interactions")
                # R50
                if max_abs < 0.30 and len(X) >= 300:
                    add_score("Random Forest Regressor", 1, "No strong single-feature linear relationship was detected; compare nonlinear models")
    # R51-R60: feature interdependence and general relation patterns
    if len(numeric.columns) >= 2:
        corr = numeric.corr().abs().to_numpy()
        upper = corr[np.triu_indices_from(corr, k=1)]
        high_corr = float(np.mean(upper >= 0.85)) if len(upper) else 0.0
        if high_corr >= 0.10:
            add_score("Ridge Regression", 1, "Correlated predictors may benefit from coefficient regularization")
        if high_corr >= 0.30:
            add_score("Lasso Regression", 1, "Redundant predictors warrant checking regularized feature selection")
        if len(X) >= 1000 and len(numeric.columns) >= 5:
            add_score("XGBoost Regressor", 1, "Sufficient data and multiple numeric predictors support interaction-aware boosting")
        if len(X) >= 1000 and len(numeric.columns) >= 5:
            add_score("Random Forest Regressor", 1, "Multiple predictors and sufficient samples support nonlinear ensemble comparison")
        if len(X) >= 300 and len(numeric.columns) <= 10:
            add_score("Support Vector Regressor", 1, "Moderate feature count supports an SVR benchmark with scaling")
        if len(X) >= 200 and len(numeric.columns) <= 10:
            add_score("KNN Regressor", 1, "Low-to-moderate dimensional numeric data supports a scaled KNN benchmark")
        if len(X) >= 5000 and len(numeric.columns) >= 5:
            add_score("LightGBM Regressor", 1, "Large numeric dataset supports scalable boosting")
        if len(X) >= 2000 and len(numeric.columns) >= 5:
            add_score("HistGradientBoosting Regressor", 1, "Numeric predictors and sufficient rows support histogram boosting")
        if len(X) >= 500 and len(numeric.columns) >= 5:
            add_score("Extra Trees Regressor", 1, "Multiple numeric features make randomized tree ensembles worth testing")
        if len(X) >= 500 and len(numeric.columns) >= 5:
            add_score("MLP Regressor", 1, "Nonlinear feature interactions can be investigated with a scaled, regularized MLP")


def apply_regression_missing_data_rules(X, y, add_score):
    """Regression Rules R61-R70: missingness, duplicates and feature quality."""
    n_samples, n_features = X.shape
    if n_samples == 0 or n_features == 0:
        return
    missing_fraction = float(X.isna().to_numpy().mean())
    affected_fraction = float((X.isna().mean() > 0).mean())
    constant_count = int((X.nunique(dropna=True) <= 1).sum())
    duplicate_fraction = float(X.duplicated().mean()) if n_samples > 1 else 0.0
    if missing_fraction > 0:
        add_score("HistGradientBoosting Regressor", 1, "Missing feature values exist; validate imputation or supported native missing-value handling")
    if missing_fraction >= 0.10:
        add_score("CatBoost Regressor", 1, "Substantial missingness warrants careful preprocessing and a CatBoost comparison")
    if affected_fraction >= 0.30:
        add_score("Random Forest Regressor", 1, "Missingness affects many columns; compare a robust preprocessing pipeline")
    if constant_count > 0:
        add_score("Ridge Regression", 1, "Constant features should be removed or checked before fitting")
    if n_features and constant_count / n_features >= 0.20:
        add_score("Lasso Regression", 1, "Many constant columns indicate a need for feature cleanup")
    if duplicate_fraction >= 0.10:
        add_score("Random Forest Regressor", 1, "Duplicate feature rows are common; review duplicates and validation strategy")
    if y.isna().any():
        add_score("Ridge Regression", 1, "Missing target values must be handled before training")
    if n_samples < 50:
        add_score("Linear Regression", 1, "Very few rows warrant simple baselines and cautious validation")
    if n_samples >= 5000 and missing_fraction == 0:
        add_score("HistGradientBoosting Regressor", 1, "Large complete dataset provides a useful benchmark for histogram boosting")
    if n_features >= 10 and n_samples >= 500:
        add_score("Random Forest Regressor", 1, "Adequate data supports comparison after feature-quality checks")


def apply_regression_model_specific_rules(X, y, add_score):
    """Regression Rules R71-R85: model-specific suitability heuristics."""
    n_samples, n_features = X.shape
    numeric_count = len(X.select_dtypes(include=["number"]).columns)
    categorical_count = len(X.select_dtypes(include=["object", "category", "bool", "string"]).columns)
    if numeric_count == n_features and n_features <= 20 and n_samples >= 100:
        add_score("Linear Regression", 1, "All-numeric, low-dimensional data supports a linear regression baseline")
    if n_samples >= 100 and n_features >= 2:
        add_score("Decision Tree Regressor", 1, "A tree-based regression baseline is worth evaluating")
    if n_samples >= 300 and n_features >= 2:
        add_score("Random Forest Regressor", 1, "Tree ensemble is a general-purpose nonlinear regression candidate")
    if n_samples >= 500 and n_features >= 5:
        add_score("Gradient Boosting Regressor", 1, "Dataset size supports comparing boosting with bagged trees")
    if n_samples >= 5000 and n_features >= 5:
        add_score("HistGradientBoosting Regressor", 1, "Large dataset supports efficient histogram-based regression")
    if n_samples >= 1000 and n_features >= 5:
        add_score("XGBoost Regressor", 1, "Sufficient sample count supports XGBoost regression benchmarking")
    if n_samples >= 1000 and n_features >= 5:
        add_score("LightGBM Regressor", 1, "Sufficient sample count supports LightGBM regression benchmarking")
    if categorical_count > 0:
        add_score("CatBoost Regressor", 2, "Categorical features make CatBoost regression worth evaluating with proper input handling")
    if n_features > n_samples:
        add_score("Ridge Regression", 2, "High-dimensional regression warrants regularized coefficients")
    if n_features > n_samples:
        add_score("Lasso Regression", 1, "High-dimensional regression warrants testing sparse coefficient selection")
    if n_samples >= 200 and n_features <= 10:
        add_score("KNN Regressor", 1, "Low-dimensional data supports a scaled neighbor-regression benchmark")
    if n_samples >= 200 and n_samples <= 5000 and numeric_count >= 2:
        add_score("Support Vector Regressor", 1, "Moderate-sized numeric data supports an SVR benchmark after scaling")
    if n_samples >= 1000 and n_features >= 10:
        add_score("Extra Trees Regressor", 1, "Larger datasets with multiple features support Extra Trees benchmarking")
    if n_samples >= 1000 and n_features >= 10:
        add_score("MLP Regressor", 1, "Larger datasets may support neural regression after scaling and regularization")
    if n_features <= 10 and n_samples >= 100:
        add_score("Elastic Net", 1, "Low-dimensional data supports comparing combined L1/L2 regularization")


REGRESSION_RULE_FUNCTIONS = [
    apply_regression_dataset_rules,
    apply_regression_feature_rules,
    apply_regression_target_rules,
    apply_regression_relationship_rules,
    apply_regression_missing_data_rules,
    apply_regression_model_specific_rules,
]

def apply_all_regression_rules(X, y, add_score=None):
    """
    Apply 85 separate regression recommendation heuristics (R1-R85).
    Returns (scores, reasons). Does not modify or execute classification rules.
    """
    from collections import defaultdict
    scores = defaultdict(float)
    reasons = defaultdict(list)

    def collect(model, points, reason):
        scores[model] += points
        reasons[model].append(reason)
        if add_score is not None:
            add_score(model, points, reason)

    for rule_function in REGRESSION_RULE_FUNCTIONS:
        rule_function(X, y, collect)
    for model in REGRESSION_MODELS:
        scores[model] += 0
    return dict(scores), dict(reasons)

# ============================================================================
# REGRESSION_RULES_END
# ============================================================================

