# AI-Based Automated Model Identification and Hyperparameter Tuning

## 📌 Project Overview

AI-Based Automated Model Identification and Hyperparameter Tuning is a generic automated machine learning framework developed for **CSV-based tabular classification datasets**.

The framework is not designed for one particular dataset. It can be applied to different suitable CSV datasets containing input features and a target variable.

The main purpose of the project is to automate important stages of the machine learning workflow, starting from dataset analysis and preprocessing and continuing through model training, model evaluation, model comparison, best model identification, cross-validation, and Explainable AI.

The framework uses a **rule-based/symbolic preprocessing engine containing more than 100 deterministic preprocessing rules**. These rules analyze the characteristics of the dataset and determine appropriate preprocessing operations.

After preprocessing, the framework trains **16 different machine learning classification models**. Each model is evaluated using Accuracy, Precision, Recall, and F1-Score.

The performance of all models is compared automatically, and the best-performing model is identified.

The selected best model is then validated using **5-Fold Cross-Validation** to evaluate its performance across different portions of the dataset.

Finally, **Explainable AI using SHAP (SHapley Additive exPlanations)** is applied to the selected best model to understand feature importance and how different attributes contribute to the model's predictions.

Overall, the project combines automated preprocessing, multiple machine learning algorithms, model evaluation, model comparison, best model selection, cross-validation, and explainability into a single machine learning framework.

---

# 🎯 Objectives

The main objectives of the project are:

1. To develop a generic automated machine learning framework for CSV-based tabular classification datasets.
2. To automatically analyze the input dataset.
3. To automate dataset preprocessing using a rule-based approach.
4. To implement more than 100 deterministic preprocessing rules.
5. To handle different types of data and dataset characteristics.
6. To train multiple machine learning classification models.
7. To evaluate the performance of each model.
8. To compare the performance of different machine learning models.
9. To automatically identify the best-performing model.
10. To save the trained models for reuse.
11. To validate the selected best model using 5-Fold Cross-Validation.
12. To implement Explainable AI using SHAP.
13. To identify important features influencing model predictions.
14. To reduce the amount of manual effort required in the machine learning workflow.

---

# 📂 Dataset Support

The framework is designed to work with **generic CSV-based tabular classification datasets**.

There is no particular dataset fixed to the project.

A suitable CSV dataset can be provided as input, provided that it contains:

- Multiple input features/attributes
- A target variable
- Tabular data
- A classification problem
- A valid CSV format

The framework does not depend on specific feature names.

Different datasets can contain different attributes, and the preprocessing engine analyzes the characteristics of those attributes before applying preprocessing operations.

For example, the framework can be applied to suitable classification datasets from domains such as:

- Healthcare
- Finance
- Education
- Business
- Customer analytics
- Other tabular classification applications

### Current Target Column Requirement

In the current implementation, the **last column of the CSV dataset is treated as the target variable**.

The remaining columns are treated as input features.

For example:

```text
Age | Gender | BloodPressure | Cholesterol | Target