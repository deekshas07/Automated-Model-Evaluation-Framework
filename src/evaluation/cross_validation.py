import pandas as pd

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def perform_cross_validation(model, dataset_path):
    """
    Perform 5-Fold Cross Validation on the selected model.
    """

    # -----------------------------------
    # Load Dataset
    # -----------------------------------
    df = pd.read_csv(dataset_path)

    df = df.dropna(how="all")
    df = df.dropna(subset=[df.columns[-1]])

    # -----------------------------------
    # Features and Target
    # -----------------------------------
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # -----------------------------------
    # Same preprocessing used in training
    # -----------------------------------
    X = pd.get_dummies(X)

    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # -----------------------------------
    # 5-Fold Stratified Cross Validation
    # -----------------------------------
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    # -----------------------------------
    # Results
    # -----------------------------------
    print("\n" + "=" * 60)
    print("5-FOLD CROSS VALIDATION")
    print("=" * 60)
    print("Scores :", scores)
    print(f"Average Accuracy   : {scores.mean():.4f}")
    print(f"Standard Deviation : {scores.std():.4f}")

    return scores