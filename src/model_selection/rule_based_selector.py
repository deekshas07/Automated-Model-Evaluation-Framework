
from src.model_selection import model_evaluation_rules


def select_model_using_rules(X, y, task_type):
    """
    Select exactly one model using AURORA model evaluation rules.

    Parameters:
        X: Preprocessed input features
        y: Target column
        task_type: 'classification' or 'regression'

    Returns:
        Dictionary containing the selected model,
        its recommendation score, and reasons.
    """

    if X is None or y is None:
        raise ValueError("Features X and target y are required.")

    if len(X) == 0 or X.shape[1] == 0:
        raise ValueError("The dataset contains no usable features.")

    task_type = task_type.lower().strip()

    # Execute classification or regression model evaluation rules
    if task_type == "classification":

        scores, reasons = model_evaluation_rules.apply_all_model_rules(
            X,
            y,
            task_type="classification"
        )

    elif task_type == "regression":

        scores, reasons = model_evaluation_rules.apply_all_regression_rules(
            X,
            y
        )

    else:
        raise ValueError(
            "Invalid task type. Use classification or regression."
        )

    # Keep only models with positive recommendation scores
    eligible_models = {
        model: score
        for model, score in scores.items()
        if score > 0
    }

    if not eligible_models:
        return {
            "selected_model": None,
            "recommendation_score": 0,
            "reasons": [],
            "message": "No model received a positive recommendation score."
        }

    # Select exactly one highest-scoring model
    selected_model = max(
        eligible_models,
        key=lambda model: (eligible_models[model], model)
    )

    return {
        "selected_model": selected_model,
        "recommendation_score": eligible_models[selected_model],
        "reasons": reasons.get(selected_model, []),
        "message": "Rule-based model recommendation completed."
    }
