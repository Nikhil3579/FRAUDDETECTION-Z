"""
config.py
---------
Single place for every path, constant and setting used across the project.
"""

import os

# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
DATA_DIR = os.path.join(
    PROJECT_DIR,
    "DATA"
)
RAW_DATA_PATH = os.path.join(
    DATA_DIR,
    "synthetic_fraud_dataset1.csv"
)
SRC_DIR = os.path.join(
    PROJECT_DIR,
    "src"
)
MODELS_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)
REPORTS_DIR = os.path.join(
    PROJECT_DIR,
    "reports"
)
OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "visualization"
)
# ============================================================
# CREATE DIRECTORIES
# ===========================================================

os.makedirs(MODELS_DIR, exist_ok=True)

os.makedirs(REPORTS_DIR, exist_ok=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# MODEL ARTIFACTS
# ============================================================

FINAL_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "fraud_model.joblib"
)

PREPROCESSOR_PATH = os.path.join(
    MODELS_DIR,
    "preprocessor.joblib"
)

BEST_PARAMS_PATH = os.path.join(
    MODELS_DIR,
    "best_params.json"
)

THRESHOLD_PATH = os.path.join(
    MODELS_DIR,
    "decision_threshold.json"
)


# ============================================================
# COLUMNS
# ============================================================

TARGET_COL = "Fraud_Label"

ID_COLS = [
    "Transaction_ID",
    "User_ID"
]

DATE_COL = "Date"


NUMERIC_COLS = [
    "Transaction_Amount",
    "Account_Balance",
    "Previous_Fraudulent_Activity",
    "Daily_Transaction_Count",
    "Card_Age",
]


CATEGORICAL_COLS = [
    "Transaction_Type",
    "Device_Type",
    "Location",
    "Merchant_Category",
    "Card_Type",
]


# ============================================================
# ENGINEERED FEATURES
# ============================================================

ENGINEERED_NUMERIC_COLS = [
    "txn_to_balance_ratio",
    "is_high_amount",
    "txn_day_of_week",
    "txn_month",
    "is_weekend",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

RANDOM_STATE = 42


# ============================================================
# TRAIN / VALIDATION / TEST
# ============================================================

TEST_SIZE = 0.15

VAL_SIZE = 0.15


# ============================================================
# BUSINESS COST
# ============================================================

COST_FALSE_NEGATIVE = 10

COST_FALSE_POSITIVE = 1

TARGET_FPR = 0.05


# ============================================================
# OPTUNA
# ============================================================

N_OPTUNA_TRIALS = 40

OPTUNA_TIMEOUT_SEC = 600