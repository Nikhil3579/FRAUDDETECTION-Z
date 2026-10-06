"""
imbalance_cost.py
============================================================

STEP 5 - IMBALANCE STRATEGIES + BUSINESS COST ANALYSIS

This module performs:

1. Load feature-engineered dataset
2. Check Fraud_Label distribution
3. Calculate class weights
4. Calculate XGBoost scale_pos_weight
5. Optional SMOTE demonstration
6. Calculate business cost
7. Find cost-optimal threshold
8. Compare thresholds
9. Save imbalance report
10. Save deployment threshold

IMPORTANT:
-----------
The feature-engineered CSV is used here ONLY to inspect the
target distribution and demonstrate the imbalance strategy.

SMOTE must NOT be applied directly to the raw CSV containing
categorical variables.

Correct production pipeline:

Raw CSV
   ↓
Cleaning
   ↓
Train / Validation / Test Split
   ↓
Feature Engineering
   ↓
Preprocessing / Encoding
   ↓
TRAINING MATRIX
   ↓
SMOTE or Class Weight
   ↓
Model Training

Validation/Test:
   ↓
NO SMOTE
"""

import os
import json
import numpy as np
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
import sys
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.append(path)
from src import config

# ============================================================
# IMPORT CONFIG
# ============================================================

try:
    # Recommended when running:
    # python -m src.imbalance_cost
    from . import config

except ImportError:

    # Allows direct execution:
    # python src\imbalance_cost.py
    import sys

    PROJECT_ROOT = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from src import config


# ============================================================
# DATA PATH
# ============================================================

FEATURE_ENGINEERED_PATH = os.path.join(
    config.DATA_DIR,
    "synthetic_fraud_feature_engineered.csv"
)


# ============================================================
# 1. LOAD FEATURE-ENGINEERED DATA
# ============================================================

def load_feature_engineered_data():

    print("\n" + "=" * 70)
    print("LOADING FEATURE-ENGINEERED DATA")
    print("=" * 70)

    if not os.path.exists(
        FEATURE_ENGINEERED_PATH
    ):

        print("\nERROR:")
        print(
            "Feature-engineered CSV was not found:"
        )

        print(
            FEATURE_ENGINEERED_PATH
        )

        print("\nFirst run:")

        print(
            "python -m src.feature_engineering"
        )

        raise FileNotFoundError(
            FEATURE_ENGINEERED_PATH
        )

    df = pd.read_csv(
        FEATURE_ENGINEERED_PATH
    )

    print(
        f"\nFile:\n{FEATURE_ENGINEERED_PATH}"
    )

    print(
        f"\nRows    : {df.shape[0]}"
    )

    print(
        f"Columns : {df.shape[1]}"
    )

    print("\nColumns:")

    for column in df.columns:
        print(f"  - {column}")

    print("=" * 70)

    return df


# ============================================================
# 2. CHECK FEATURE-ENGINEERED FEATURES
# ============================================================

def show_engineered_features(df):

    expected_features = [
        "txn_to_balance_ratio",
        "is_high_amount",
        "txn_day_of_week",
        "txn_month",
        "is_weekend"
    ]

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING CHECK")
    print("=" * 70)

    found_features = []

    for feature in expected_features:

        if feature in df.columns:

            found_features.append(
                feature
            )

            print(
                f"✓ {feature}"
            )

        else:

            print(
                f"✗ {feature} NOT FOUND"
            )

    print(
        f"\nCreated features found: "
        f"{len(found_features)}/{len(expected_features)}"
    )

    if found_features:

        print("\nSample engineered values:")

        print(
            df[found_features]
            .head(10)
            .to_string(index=False)
        )

    print("=" * 70)


# ============================================================
# 3. CLASS DISTRIBUTION
# ============================================================

def show_class_distribution(
    y,
    dataset_name="Dataset"
):

    y = pd.Series(y)

    total = len(y)

    non_fraud = int(
        (y == 0).sum()
    )

    fraud = int(
        (y == 1).sum()
    )

    non_fraud_pct = (
        non_fraud / total * 100
        if total > 0
        else 0
    )

    fraud_pct = (
        fraud / total * 100
        if total > 0
        else 0
    )

    print("\n" + "=" * 70)
    print(
        f"{dataset_name.upper()} - CLASS DISTRIBUTION"
    )
    print("=" * 70)

    print(
        f"Total transactions : {total}"
    )

    print(
        f"Non-Fraud (0)     : "
        f"{non_fraud:,} "
        f"({non_fraud_pct:.2f}%)"
    )

    print(
        f"Fraud (1)         : "
        f"{fraud:,} "
        f"({fraud_pct:.2f}%)"
    )

    if fraud_pct < 5:

        status = "HIGHLY IMBALANCED"

    elif fraud_pct < 20:

        status = "IMBALANCED"

    elif fraud_pct < 40:

        status = "MODERATELY IMBALANCED"

    else:

        status = "RELATIVELY BALANCED"

    print(
        f"\nDataset status : {status}"
    )

    print("=" * 70)

    return {
        "total": total,
        "non_fraud": non_fraud,
        "fraud": fraud,
        "non_fraud_percentage":
            non_fraud_pct,
        "fraud_percentage":
            fraud_pct,
        "status": status
    }


# ============================================================
# 4. CLASS WEIGHT
# ============================================================

def get_class_weight_dict(
    y_train
):

    classes = np.unique(
        y_train
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = dict(
        zip(
            classes.astype(int),
            weights.astype(float)
        )
    )

    print("\n" + "=" * 70)
    print("CLASS WEIGHT STRATEGY")
    print("=" * 70)

    for cls, weight in class_weights.items():

        label = (
            "Non-Fraud"
            if cls == 0
            else "Fraud"
        )

        print(
            f"Class {cls} ({label}) "
            f"weight = {weight:.4f}"
        )

    print(
        "\nMeaning:"
    )

    print(
        "The model gives more importance to "
        "mistakes involving the minority class."
    )

    print("=" * 70)

    return class_weights


# ============================================================
# 5. XGBOOST SCALE POS WEIGHT
# ============================================================

def get_scale_pos_weight(
    y_train
):

    y_train = np.asarray(
        y_train
    )

    negative_count = np.sum(
        y_train == 0
    )

    positive_count = np.sum(
        y_train == 1
    )

    scale_pos_weight = (
        negative_count /
        max(positive_count, 1)
    )

    print("\n" + "=" * 70)
    print("XGBOOST SCALE_POS_WEIGHT")
    print("=" * 70)

    print(
        f"Non-Fraud count : "
        f"{negative_count:,}")

    print(
        f"Fraud count     : "
        f"{positive_count:,}")

    print(
        f"\nscale_pos_weight = "
        f"{scale_pos_weight:.4f}")

    print(
        "\nFormula:")

    print(
        "Negative samples / Positive samples")

    print("=" * 70)

    return float(
        scale_pos_weight)


# ============================================================
# 6. SMOTE
# ============================================================

def apply_smote(
    X_train,
    y_train,
    random_state=None):

    if random_state is None:

        random_state = config.RANDOM_STATE

    from imblearn.over_sampling import SMOTE

    print("\n" + "=" * 70)
    print("SMOTE ANALYSIS")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "SMOTE should be applied ONLY to "
        "the training matrix after preprocessing.")

    print(
        "\nBefore SMOTE:")

    before = pd.Series(
        y_train
    ).value_counts().sort_index()

    print(
        f"Non-Fraud : "
        f"{before.get(0, 0):,}"
    )

    print(
        f"Fraud     : "
        f"{before.get(1, 0):,}"
    )

    smote = SMOTE(
        random_state=random_state
    )

    X_resampled, y_resampled = (
        smote.fit_resample(
            X_train,
            y_train
        )
    )

    print(
        "\nAfter SMOTE:"
    )

    after = pd.Series(
        y_resampled
    ).value_counts().sort_index()

    print(
        f"Non-Fraud : "
        f"{after.get(0, 0):,}"
    )

    print(
        f"Fraud     : "
        f"{after.get(1, 0):,}"
    )

    print(
        "\n✓ SMOTE completed."
    )

    print(
        "✓ Validation and test data "
        "must NOT be SMOTE-resampled."
    )

    print("=" * 70)

    return (
        X_resampled,
        y_resampled
    )


# ============================================================
# 7. BUSINESS COST
# ============================================================

def total_cost(
    y_true,
    y_pred,
    cost_fn=None,
    cost_fp=None
):

    if cost_fn is None:

        cost_fn = config.COST_FALSE_NEGATIVE

    if cost_fp is None:

        cost_fp = config.COST_FALSE_POSITIVE

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    false_negatives = np.sum(
        (y_true == 1) &
        (y_pred == 0)
    )

    false_positives = np.sum(
        (y_true == 0) &
        (y_pred == 1)
    )

    return float(
        false_negatives * cost_fn
        +
        false_positives * cost_fp
    )


# ============================================================
# 8. FIND OPTIMAL THRESHOLD
# ============================================================

def find_cost_optimal_threshold(y_true, y_probability):
    """
    Find the decision threshold that minimizes business cost.

    False Negative cost = config.COST_FALSE_NEGATIVE
    False Positive cost = config.COST_FALSE_POSITIVE

    Threshold selection is performed ONLY on validation data.
    """

    best_threshold = 0.50
    best_cost = float("inf")

    best_metrics = {}

    thresholds = np.arange(0.01, 1.00, 0.01)

    for threshold in thresholds:

        y_pred = (
            np.asarray(y_probability) >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        cost = (
            fn * config.COST_FALSE_NEGATIVE
            + fp * config.COST_FALSE_POSITIVE
        )

        if cost < best_cost:

            best_cost = float(cost)
            best_threshold = float(threshold)

            precision = (
                tp / (tp + fp)
                if (tp + fp) > 0
                else 0.0
            )

            recall = (
                tp / (tp + fn)
                if (tp + fn) > 0
                else 0.0
            )

            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            accuracy = (
                (tp + tn) / (tp + tn + fp + fn)
            )
            print(
                f"\nNew best threshold found: "
                f"{best_threshold:.2f} "
                f"with cost = {best_cost:.2f}"
            )   
            best_metrics = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            }

    return {
        "best_threshold": best_threshold,
        "best_cost": best_cost,
        "min_cost": best_cost,       # compatibility
        **best_metrics,
    }
# ============================================================
# 9. THRESHOLD COMPARISON
# ============================================================

def compare_thresholds(
    y_true,
    y_probability,
    selected_threshold=None
):

    thresholds = [
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80
    ]

    if selected_threshold is not None:

        selected_threshold = round(
            selected_threshold,
            2
        )

        if selected_threshold not in thresholds:

            thresholds.append(
                selected_threshold
            )

    thresholds = sorted(
        set(thresholds)
    )

    rows = []

    for threshold in thresholds:

        y_pred = (
            np.asarray(
                y_probability
            ) >= threshold
        ).astype(int)

        tn, fp, fn, tp = (
            confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1]
            ).ravel()
        )

        rows.append({

            "Threshold":
                threshold,

            "TP":
                tp,

            "TN":
                tn,

            "FP":
                fp,

            "FN":
                fn,

            "Precision":
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),

            "Recall":
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),

            "F1":
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),

            "Business_Cost":
                total_cost(
                    y_true,
                    y_pred
                )
        })

    comparison = pd.DataFrame(
        rows
    )

    print("\n" + "=" * 70)
    print("THRESHOLD COMPARISON")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

    print("=" * 70)

    return comparison


# ============================================================
# 10. SAVE REPORT
# ============================================================

def save_imbalance_report(
    distribution,
    class_weights,
    scale_pos_weight
):

    report_path = os.path.join(
        config.REPORTS_DIR,
        "imbalance_analysis.json"
    )

    report = {

        "class_distribution":
            distribution,

        "class_weights":
            {
                str(k): float(v)
                for k, v
                in class_weights.items()
            },

        "scale_pos_weight":
            float(scale_pos_weight),

        "cost_false_negative":
            config.COST_FALSE_NEGATIVE,

        "cost_false_positive":
            config.COST_FALSE_POSITIVE,

        "notes": [

            "SMOTE must only be applied to training data.",

            "Validation and test data must remain unchanged.",

            "Threshold should be optimized on validation data.",

            "Test data should be used only for final evaluation."
        ]
    }

    os.makedirs(
        config.REPORTS_DIR,
        exist_ok=True
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print("\n" + "=" * 70)
    print("REPORT SAVED")
    print("=" * 70)

    print(
        f"File:\n{report_path}"
    )

    print("=" * 70)

    return report_path


# ============================================================
# 11. SAVE DEPLOYMENT THRESHOLD
# ============================================================

def save_deployment_threshold(
    threshold_result
):

    threshold_path = os.path.join(
        config.MODELS_DIR,
        "decision_threshold.json"
    )

    os.makedirs(
        config.MODELS_DIR,
        exist_ok=True
    )

    data = {

        "threshold":
            threshold_result[
                "best_threshold"
            ],

        "business_cost":
            threshold_result[
                "business_cost"
            ],

        "precision":
            threshold_result[
                "precision"
            ],

        "recall":
            threshold_result[
                "recall"
            ],

        "f1":
            threshold_result[
                "f1"
            ],

        "cost_false_negative":
            config.COST_FALSE_NEGATIVE,

        "cost_false_positive":
            config.COST_FALSE_POSITIVE
    }

    with open(
        threshold_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )

    print("\n" + "=" * 70)
    print("DEPLOYMENT THRESHOLD SAVED")
    print("=" * 70)

    print(
        f"File:\n{threshold_path}"
    )

    print(
        f"\nThreshold = "
        f"{data['threshold']:.2f}"
    )

    print("=" * 70)

    return threshold_path


# ============================================================
# 12. DEPLOYMENT INSIGHTS
# ============================================================

def show_deployment_insights():

    print("\n" + "=" * 70)
    print("DEPLOYMENT INSIGHTS")
    print("=" * 70)

    print(
        "\n1. CLASS IMBALANCE"
    )

    print(
        "Use class_weight or XGBoost "
        "scale_pos_weight during training."
    )

    print(
        "\n2. SMOTE"
    )

    print(
        "Use SMOTE only on the TRAINING "
        "matrix after preprocessing."
    )

    print(
        "\n3. VALIDATION"
    )

    print(
        "Do NOT apply SMOTE to validation data."
    )

    print(
        "\n4. TEST"
    )

    print(
        "Do NOT apply SMOTE to test data."
    )

    print(
        "\n5. THRESHOLD"
    )

    print(
        "Do not automatically use 0.50."
    )

    print(
        "Choose the threshold using "
        "validation-set business cost."
    )

    print(
        "\n6. DEPLOYMENT"
    )

    print(
        "Flask should load the saved "
        "decision_threshold.json."
    )

    print(
        "\n7. FINAL TEST"
    )

    print(
        "Use the test set only once for "
        "final unbiased evaluation."
    )

    print(
        "\nRecommended architecture:"
    )


    print("=" * 70)


# ============================================================
# 13. MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("RUNNING IMBALANCE ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD NEW FEATURE-ENGINEERED CSV

    df = load_feature_engineered_data()

    # VERIFY CREATED FEATURES
    # --------------------------------------------------------

    show_engineered_features(df)

    # --------------------------------------------------------
    # CHECK TARGET
    # --------------------------------------------------------
    if config.TARGET_COL not in df.columns:

        raise ValueError(
            f"Target column "
            f"'{config.TARGET_COL}' "
            f"not found in dataset."
        )

    # -----------------------------------------
    # TARGET
    y = df[
        config.TARGET_COL
    ]
    # SHOW CLASS DISTRIBUTION
    # --------------------------------------------------------
    distribution = (
        show_class_distribution(
            y,
            "Feature-Engineered Dataset"
        )
    )
    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_weights = (
        get_class_weight_dict(
            y
        )
    )

    # --------------------------------------------------------
    # XGBOOST WEIGHT
    # --------------------------------------------------------

    scale_pos_weight = (
        get_scale_pos_weight(
            y
        )
    )

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    save_imbalance_report(
        distribution,
        class_weights,
        scale_pos_weight
    )

    # --------------------------------------------------------
    # DEPLOYMENT INFORMATION
    # --------------------------------------------------------

    show_deployment_insights()

    # --------------------------------------------------------
    # IMPORTANT MESSAGE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("IMPORTANT NEXT STEP")
    print("=" * 70)

    print(
        "\nThreshold optimization requires "
        "VALIDATION predictions."
    )

    print(
        "\nExample:"
    )

    print(
        "val_probability = model.predict_proba(X_val)[:, 1]"
    )

    print(
        "threshold = "
        "find_cost_optimal_threshold("
        "y_val, val_probability)"
    )

    print(
        "\nThe TEST dataset must NOT be used "
        "to select the threshold."
    )

    print("=" * 70)

    print("\n")
    print("=" * 70)
    print("IMBALANCE ANALYSIS COMPLETED")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()