"""
feature_engineering.py
------------------------
Two things live here:
1. add_engineered_features()  -> pure pandas, creates new columns
2. build_preprocessor()       -> an sklearn ColumnTransformer that scales
                                  numeric columns and one-hot-encodes
                                  categorical columns.

Beginner note: an sklearn Pipeline/ColumnTransformer is important because
it guarantees the EXACT same transformation is applied at training time
and at prediction time (in the API). This is the #1 source of bugs in
real ML projects ("training-serving skew") - so we solve it once, here.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
import sys
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.append(path)
from src import config
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def load_data():
    df = pd.read_csv(config.RAW_DATA_PATH)

    print("\n" + "=" * 70)
    print("ORIGINAL DATASET")
    print("=" * 70)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df

import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import sys
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.append(path)
from src import config


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data():
    df = pd.read_csv(config.RAW_DATA_PATH)

    print("\n" + "=" * 70)
    print("ORIGINAL DATASET")
    print("=" * 70)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def add_engineered_features(df):
    df = df.copy()
    # Convert Date
    # --------------------------------------------------------
    df[config.DATE_COL] = pd.to_datetime(
        df[config.DATE_COL],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Feature 1: Transaction Amount / Account Balance
    # --------------------------------------------------------
    df["txn_to_balance_ratio"] = np.where( df["Account_Balance"] > 0,df["Transaction_Amount"] / df["Account_Balance"],
        0)

    # --------------------------------------------------------
    # Feature 2: High Transaction Amount
    # --------------------------------------------------------
    high_amount_threshold = df["Transaction_Amount"].quantile(0.95)

    df["is_high_amount"] = (df["Transaction_Amount"] >= high_amount_threshold).astype(int)

    # --------------------------------------------------------
    # Feature 3: Day of Week
    # Monday = 0
    # Sunday = 6
    # --------------------------------------------------------
    df["txn_day_of_week"] = df[config.DATE_COL].dt.dayofweek
    # --------------------------------------------------------
    # Feature 4: Month
    # --------------------------------------------------------
    df["txn_month"] = df[config.DATE_COL].dt.month# --------------------------------------------------------
    # Feature 5: Weekend
    # --------------------------------------------------------
    df["is_weekend"] = (
        df["txn_day_of_week"] >= 5
    ).astype(int)

    return df


# ============================================================
# 3. SHOW CREATED FEATURES
# ============================================================

def show_engineered_features(df):
    engineered_features = [
        "txn_to_balance_ratio",
        "is_high_amount",
        "txn_day_of_week",
        "txn_month",
        "is_weekend"
    ]

    print("\n" + "=" * 70)
    print("ENGINEERED FEATURES")
    print("=" * 70)

    print("\nCreated features:")

    for feature in engineered_features:
        print(f"  ✓ {feature}")

    print("\nFeature information:")
    print(
        df[engineered_features]
        .describe()
        .round(4)
    )

    print("\nFirst 10 rows:")
    print(
        df[
            [
                "Transaction_ID",
                "Transaction_Amount",
                "Account_Balance",
                "Date",
                "Fraud_Label"
            ] + engineered_features
        ].head(10)
    )


# ============================================================
# 4. SAVE FEATURE-ENGINEERED CSV
# ============================================================

def save_feature_engineered_data(df):

    output_path = os.path.join(
        config.DATA_DIR,
        "synthetic_fraud_feature_engineered.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("FEATURE-ENGINEERED DATA SAVED")
    print("=" * 70)

    print(f"File: {output_path}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return output_path


# ============================================================
# 5. PREPARE FEATURES FOR MODEL
# ============================================================

def prepare_features(df):

    df = add_engineered_features(df)

    keep_cols = (
        config.NUMERIC_COLS
        + config.ENGINEERED_NUMERIC_COLS
        + config.CATEGORICAL_COLS
    )

    return df[keep_cols]


# ============================================================
# 6. PREPROCESSOR
# ============================================================

def build_preprocessor():

    numeric_features = (
        config.NUMERIC_COLS
        + config.ENGINEERED_NUMERIC_COLS
    )
    categorical_features = config.CATEGORICAL_COLS

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ],
        remainder="drop"
    )

    return preprocessor


# ============================================================
# 7. MAIN
# ============================================================

def main():

    # Load original data
    df = load_data()

    # Create engineered features
    df_engineered = add_engineered_features(df)

    # Display new features
    show_engineered_features(df_engineered)

    # Save new CSV
    output_path = save_feature_engineered_data(
        df_engineered
    )

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 70)

    print("\nOriginal dataset:")
    print(config.RAW_DATA_PATH)

    print("\nNew dataset:")
    print(output_path)


if __name__ == "__main__":
    main()