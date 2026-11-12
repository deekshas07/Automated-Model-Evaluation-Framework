<div align="center">

  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20,30,40&height=200§ion=header&text=Automated%20Model%20Evaluation%20Framework&fontSize=30&fontColor=ffffff&animation=fadeIn&fontAlignY=38" width="100%" />

  <p align="center">
    <img src="https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge&logo=appveyor" />
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/License-MIT-orange?style=for-the-badge" />
    <img src="https://img.shields.io/badge/Framework-AMEF-informational?style=for-the-badge" />
  </p>

  <h3>🚀 An intelligent, generic end-to-end Automated Machine Learning (AutoML) framework designed for CSV-based tabular classification datasets.</h3>

</div>

---

## 📌 Project Overview

The **Automated Model Evaluation Framework (AMEF)** is engineered to take the heavy lifting and manual trial-and-error out of the machine learning lifecycle. Rather than being tied to a single specific dataset, the framework is fully generic and capable of processing any valid CSV tabular classification dataset containing input features and a target variable (where the last column is treated as the target class).

By merging symbolic rule-based intelligence with empirical model training, AMEF automates everything from schema profiling and data preparation to model arbitration, 5-fold cross-validation, hyperparameter tuning, and explainable AI.

---

## 🎯 Core Objectives

1. **Generic Tabular Support:** Process arbitrary CSV classification datasets across domains (healthcare, finance, business, etc.) without relying on hardcoded feature names.
2. **Deterministic Preprocessing:** Automatically analyze input structures and apply over **100+ deterministic data preparation rules** for missing values, outliers, and scaling.
3. **Dual-Track Evaluation Engine:** 
   * **Approach 1:** Systematically train and evaluate **16 diverse machine learning classification models**.
   * **Approach 2:** Utilize **100 classification rules and 83 preparation rules** to score dataset meta-features and recommend optimal heuristic variants.
4. **Hybrid Arbitration & Selection:** Compare empirical results against rule-based recommendations to crown the champion model architecture.
5. **Robust Validation:** Validate top models using **5-Fold Stratified Cross-Validation** to guarantee generalizability and stability.
6. **Explainable AI (XAI):** Integrate **SHAP (SHapley Additive exPlanations)** to decipher feature importance and model decision pathways.

---

## 📊 System Workflow Architecture

```mermaid
graph TD
    A[Raw CSV Dataset] --> B[Schema Profiling & Target Extraction Last Column]
    B --> C[Symbolic Preprocessing: 100+ Rules]
    C
