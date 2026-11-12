graph TD
    classDef process fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#fff;
    classDef track1 fill:#0f172a,stroke:#34d399,stroke-width:2px,color:#fff;
    classDef track2 fill:#0f172a,stroke:#f472b6,stroke-width:2px,color:#fff;
    classDef decision fill:#312e81,stroke:#818cf8,stroke-width:2px,color:#fff;

    A[RAW DATASET: CSV Input] --> B[DATA PREPROCESSING]:::process
    B --> C["Apply 100+ Data Preparation Rules<br/>• Handling Missing Values<br/>• Outlier Detection<br/>• Normalization<br/>• Feature Engineering"]:::process
    C --> D[PREPROCESSED DATASET: Ready for Modeling]:::process
    
    D --> E[APPROACH 1: ML Model Training & Selection]:::track1
    D --> F[APPROACH 2: Rule-Based Model Evaluation]:::track2
    
    E --> G["TRAIN 16 ML MODELS<br/>Random Forest, XGBoost, LightGBM, SVM, etc."]:::track1
    F --> H["APPLY 100 CLASSIFICATION RULES & 83 PREPARATION RULES"]:::track2
    
    G --> I[SELECT BEST MODEL FROM APPROACH 1]:::track1
    H --> J[SCORE DATASET BASED ON RULES]:::track2
    J --> K[SELECT & TRAIN BEST MODEL FROM APPROACH 2]:::track2
    
    I --> L[HYBRID COMPARISON METHOD: Approach 1 vs Approach 2]:::decision
    K --> L:::decision
    
    L --> M[ULTIMATE BEST CHAMPION MODEL SELECTED]:::decision
    M --> N["POST-PROCESSING & OPTIMIZATION<br/>• Hyperparameter Tuning<br/>• 5-Fold Cross-Validation<br/>• SHAP Explainability Analysis"]:::process
    N --> O[FINAL DEPLOYABLE PRODUCTION MODEL]:::process
