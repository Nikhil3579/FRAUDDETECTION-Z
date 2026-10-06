"""
train_baselines.py
==================
STEP 5: TRAIN BASELINE MODELS

Purpose
-------
Before hyperparameter tuning, we train a few baseline models.

The goal is to:
1. Verify that the complete ML pipeline works.
2. Establish a performance benchmark ("score to beat").
3. Compare different model families.
4. Select the strongest baseline model for hyperparameter tuning.

Baseline models
---------------
1. Logistic Regression
   - Simple and fast
   - Linear decision boundary
   - Easy to interpret

2. Random Forest
   - Non-linear model
   - Captures feature interactions
   - Robust for tabular data

3. XGBoost
   - Powerful gradient boosting model
   - Excellent for structured/tabular data
   - Default parameters are used here
   - Hyperparameter tuning is done later in Step 6

Important
---------
- Class weights / scale_pos_weight are calculated ONLY from y_train.
- Validation data is used only for model comparison.
- Test data is NOT used during baseline selection.
- No hyperparameter optimization is performed here.
- The best baseline model will be tuned in hyperparameter_tuning.py.

Pipeline
--------
Raw Data
   ↓
Data Cleaning
   ↓
Train / Validation / Test Split
   ↓
Feature Engineering
   ↓
Preprocessing
   ↓
Class Imbalance Handling
   ↓
Baseline Models
   ↓
Validation PR-AUC Comparison
   ↓
Best Baseline
   ↓
Hyperparameter Tuning (STEP 6)

Run
---
From the project root:

    python -m src.train_baselines
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import os
import pandas as pd
import sys
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.append(path)
from src import config
from src.data_preprocessing import get_clean_splits
from src.feature_engineering import prepare_features, build_preprocessor
from src.imbalance_cost import (
    get_class_weight_dict,
    get_scale_pos_weight,
)
from src.evaluate import full_report, print_report


# ============================================================
# 1. BUILD COMMON MODEL PIPELINE
# ============================================================

def build_pipeline(model) -> Pipeline:
    """
    Creates a common preprocessing + model pipeline.

    The same preprocessing logic is used for all baseline models
    so that the comparison is fair.
    """

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


# ============================================================
# 2. TRAIN BASELINE MODELS
# ============================================================

def run_baselines():

    print("\n" + "=" * 70)
    print("STEP 5 - BASELINE MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load clean train / validation / test splits
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = get_clean_splits()

    print("\nDataset splits:")
    print(f"Train : {X_train.shape}")
    print(f"Val   : {X_val.shape}")
    print(f"Test  : {X_test.shape}")

    # --------------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------------
    # IMPORTANT:
    # Feature engineering is applied separately to each split.
    # The model is trained only on X_train.
    # Validation is used only for model comparison.

    X_train_fe = prepare_features(X_train)
    X_val_fe = prepare_features(X_val)

    print("\nFeature engineering completed.")

    # --------------------------------------------------------
    # Handle Class Imbalance
    # --------------------------------------------------------
    # Calculate imbalance parameters ONLY from training data.
    # This prevents validation/test information from influencing
    # the model configuration.

    class_weights = get_class_weight_dict(y_train)
    scale_pos_weight = get_scale_pos_weight(y_train)

    print("\nClass imbalance parameters:")
    print(f"Class weights       : {class_weights}")
    print(f"Scale positive weight: {scale_pos_weight:.4f}")

    # ========================================================
    # 3. DEFINE BASELINE MODELS
    # ========================================================

    models = {

        # ----------------------------------------------------
        # Model 1: Logistic Regression
        # ----------------------------------------------------
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight=class_weights,
            random_state=config.RANDOM_STATE,
        ),

        # ----------------------------------------------------
        # Model 2: Random Forest
        # ----------------------------------------------------
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight=class_weights,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),

        # ----------------------------------------------------
        # Model 3: XGBoost
        # ----------------------------------------------------
        # Default-style parameters.
        # Tuning will be performed later in
        # hyperparameter_tuning.py.

        "XGBoost_default": XGBClassifier(
            n_estimators=300,
            scale_pos_weight=scale_pos_weight,
            random_state=config.RANDOM_STATE,
            eval_metric="aucpr",
            n_jobs=-1,
        ),
    }

    # ========================================================
    # 4. TRAIN AND EVALUATE BASELINES
    # ========================================================

    results = {}

    for name, model in models.items():

        print("\n" + "-" * 70)
        print(f"TRAINING: {name}")
        print("-" * 70)

        # Create preprocessing + model pipeline
        pipeline = build_pipeline(model)

        # Train ONLY on training data
        pipeline.fit(
            X_train_fe,
            y_train,
        )

        # ----------------------------------------------------
        # Validation Prediction
        # ----------------------------------------------------
        # We use probabilities because fraud detection is
        # evaluated using PR-AUC and threshold-based metrics.

        y_val_proba = pipeline.predict_proba(
            X_val_fe
        )[:, 1]

        # ----------------------------------------------------
        # Evaluate using default threshold = 0.50
        # ----------------------------------------------------

        report = full_report(
            y_val,
            y_val_proba,
            threshold=0.50,
        )

        print_report(
            report,
            title=f"{name} - Validation Performance",
        )

        # Store important metrics
        results[name] = {
            "pr_auc": report["pr_auc"],
            "recall_at_fpr": report.get("recall_at_fpr"),
            "precision": report["classification_report"]["1"]["precision"],
            "recall": report["classification_report"]["1"]["recall"],
            "f1": report["classification_report"]["1"]["f1-score"],
        }

    # ========================================================
    # 5. BASELINE MODEL COMPARISON
    # ========================================================

    print("\n" + "=" * 70)
    print("BASELINE MODEL COMPARISON")
    print("=" * 70)

    # Sort models by PR-AUC
    ranked_results = sorted(
        results.items(),
        key=lambda item: item[1]["pr_auc"],
        reverse=True,
    )

    for rank, (name, metrics) in enumerate(
        ranked_results,
        start=1,
    ):

        print(
            f"{rank}. {name:20s} "
            f"PR-AUC = {metrics['pr_auc']:.4f}"
        )

    # --------------------------------------------------------
    # Identify best baseline
    # --------------------------------------------------------

    best_model_name = ranked_results[0][0]
    best_pr_auc = ranked_results[0][1]["pr_auc"]

    print("\n" + "-" * 70)
    print("BEST BASELINE MODEL")
    print("-" * 70)

    print(f"Model   : {best_model_name}")
    print(f"PR-AUC  : {best_pr_auc:.4f}")

    print("\nNext step:")
    print(
        f"Use {best_model_name} as the candidate for "
        "hyperparameter tuning."
    )

    print("\n" + "=" * 70)
    print("STEP 5 COMPLETED")
    print("=" * 70)

    return results


# ============================================================
# 6. MAIN
# ============================================================

if __name__ == "__main__":
    run_baselines()