"""
hyperparameter_tuning.py
========================

STEP 6: XGBOOST HYPERPARAMETER TUNING WITH OPTUNA

Purpose
-------
Step 5 established baseline performance using:

    1. Logistic Regression
    2. Random Forest
    3. XGBoost with default parameters

Now we improve the strongest tabular model, XGBoost, by searching
for better hyperparameter combinations.

Why Hyperparameter Tuning?
---------------------------
XGBoost performance depends heavily on parameters such as:

    - learning_rate
    - max_depth
    - n_estimators
    - subsample
    - colsample_bytree
    - min_child_weight
    - gamma
    - reg_alpha
    - reg_lambda

Manually testing combinations is slow and difficult to reproduce.

Optuna automatically searches the parameter space.

Optimization objective
----------------------
The project uses:

    PR-AUC / Average Precision

as the primary optimization metric because this is a fraud detection
problem and the positive class is the important class.

IMPORTANT:
----------
    Training data -> used to train each trial
    Validation data -> used to calculate PR-AUC
    Test data -> NEVER used during tuning

The test set remains untouched until train_final.py.

Run:
----
    python -m src.hyperparameter_tuning
"""

import json

import optuna
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import os
import sys
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.append(path)
from src import config
from src.data_preprocessing import get_clean_splits
from src.feature_engineering import (
    prepare_features,
    build_preprocessor,
)
from src.imbalance_cost import get_scale_pos_weight
from src.evaluate import pr_auc


# ============================================================
# 1. BUILD XGBOOST PIPELINE
# ============================================================

def build_xgb_pipeline(params, scale_pos_weight):
    """
    Build the complete preprocessing + XGBoost pipeline.

    Keeping preprocessing inside the Pipeline prevents the
    preprocessing logic from being accidentally fitted outside
    the training data.
    """

    model = XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        random_state=config.RANDOM_STATE,
        eval_metric="aucpr",
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )

    return pipeline


# ============================================================
# 2. OPTUNA OBJECTIVE
# ============================================================

def objective(
    trial,
    X_train_fe,
    y_train,
    X_val_fe,
    y_val,
    scale_pos_weight,
):
    """
    One Optuna trial.

    Optuna selects one hyperparameter combination,
    trains XGBoost and evaluates validation PR-AUC.

    Higher PR-AUC is better.
    """

    params = {

        # Number of boosting trees
        "n_estimators": trial.suggest_int(
            "n_estimators",
            150,
            600,
        ),

        # Tree complexity
        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            10,
        ),

        # Learning speed
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.30,
            log=True,
        ),

        # Row sampling
        "subsample": trial.suggest_float(
            "subsample",
            0.60,
            1.00,
        ),

        # Feature sampling
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.60,
            1.00,
        ),

        # Minimum child weight
        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            1,
            10,
        ),

        # Minimum loss reduction
        "gamma": trial.suggest_float(
            "gamma",
            0.0,
            5.0,
        ),

        # L1 regularization
        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-3,
            10.0,
            log=True,
        ),

        # L2 regularization
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1e-3,
            10.0,
            log=True,
        ),
    }

    # --------------------------------------------------------
    # Build pipeline
    # --------------------------------------------------------

    pipeline = build_xgb_pipeline(
        params=params,
        scale_pos_weight=scale_pos_weight,
    )

    # --------------------------------------------------------
    # Train on TRAIN only
    # --------------------------------------------------------

    pipeline.fit(
        X_train_fe,
        y_train,
    )

    # --------------------------------------------------------
    # Evaluate on VALIDATION only
    # --------------------------------------------------------

    val_proba = pipeline.predict_proba(
        X_val_fe
    )[:, 1]

    score = pr_auc(
        y_val,
        val_proba,
    )

    # Show trial result
    print(
        f"Trial {trial.number:03d} | "
        f"Validation PR-AUC = {score:.4f}"
    )

    return score


# ============================================================
# 3. RUN OPTUNA
# ============================================================

def run_tuning(
    n_trials: int = config.N_OPTUNA_TRIALS,
):
    """
    Run the complete Optuna tuning process.
    """

    print("\n" + "=" * 70)
    print("STEP 6 - XGBOOST HYPERPARAMETER TUNING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = get_clean_splits()

    print("\nDataset:")
    print(f"Train: {X_train.shape}")
    print(f"Val  : {X_val.shape}")
    print(f"Test : {X_test.shape}")

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    X_train_fe = prepare_features(X_train)
    X_val_fe = prepare_features(X_val)

    # Test is deliberately not prepared/used here.
    # It remains untouched until final evaluation.

    # --------------------------------------------------------
    # Class imbalance
    # --------------------------------------------------------

    scale_pos_weight = get_scale_pos_weight(
        y_train
    )

    print(
        f"\nTraining scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # ========================================================
    # OPTUNA STUDY
    # ========================================================

    study = optuna.create_study(
        direction="maximize",
        study_name="fraud_xgb_tuning",
    )

    print("\nStarting Optuna search...")
    print(f"Maximum trials: {n_trials}")
    print(
        f"Maximum time: "
        f"{config.OPTUNA_TIMEOUT_SEC} seconds"
    )

    study.optimize(
        lambda trial: objective(
            trial,
            X_train_fe,
            y_train,
            X_val_fe,
            y_val,
            scale_pos_weight,
        ),
        n_trials=n_trials,
        timeout=config.OPTUNA_TIMEOUT_SEC,
        show_progress_bar=True,
    )

    # ========================================================
    # BEST RESULT
    # ========================================================

    print("\n" + "=" * 70)
    print("OPTUNA RESULTS")
    print("=" * 70)

    print(
        f"Best Validation PR-AUC: "
        f"{study.best_value:.4f}"
    )

    print("\nBest Hyperparameters:")

    for parameter, value in study.best_params.items():
        print(
            f"  {parameter}: {value}"
        )

    # ========================================================
    # SAVE BEST PARAMETERS
    # ========================================================

    with open(
        config.BEST_PARAMS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            study.best_params,
            file,
            indent=4,
        )

    print(
        f"\nBest parameters saved to:"
        f"\n{config.BEST_PARAMS_PATH}"
    )

    print("\n" + "=" * 70)
    print("STEP 6 COMPLETED")
    print("=" * 70)

    print(
        "\nNext step:"
        "\nRun:"
        "\npython -m src.train_final"
    )

    return study.best_params


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_tuning()