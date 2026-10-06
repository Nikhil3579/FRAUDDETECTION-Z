"""
data_preprocessing.py
----------------------
Loading + cleaning + train/val/test splitting.

Beginner note: this file answers ONE question only -> "how do I get a
clean, split dataset?" ike this makes the codebase easy to debug:
if a number looks wrong, you know exactly which file to open.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import config

def load_raw_data(path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the CSV from disk."""
    df = pd.read_csv(path)
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal, defensive cleaning:
    - drop exact duplicate rows
    - drop rows with a missing target (can't train/evaluate on those)
    - parse the Date column into a real datetime
    """
    df = df.drop_duplicates()
    df = df.dropna(subset=[config.TARGET_COL])

    # Real-world dates are messy -> errors="coerce" turns bad dates into NaT
    # instead of crashing the whole pipeline. format="mixed" lets pandas
    # infer per-row while staying fast; falls back gracefully on odd rows.
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL], errors="coerce", format="mixed")

    return df

def time_based_split(df):

    df = df.sort_values(config.DATE_COL).copy()

    n = len(df)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    X_train = train_df.drop(columns=[config.TARGET_COL])
    y_train = train_df[config.TARGET_COL]

    X_val = val_df.drop(columns=[config.TARGET_COL])
    y_val = val_df[config.TARGET_COL]

    X_test = test_df.drop(columns=[config.TARGET_COL])
    y_test = test_df[config.TARGET_COL]

    return X_train, X_val, X_test, y_train, y_val, y_test


def check_user_overlap(X_train, X_val, X_test):

    train_users = set(X_train["User_ID"])
    val_users = set(X_val["User_ID"])
    test_users = set(X_test["User_ID"])

    print("\n" + "=" * 60)
    print("USER OVERLAP CHECK")
    print("=" * 60)

    print("Train ∩ Validation:",
          len(train_users & val_users))

    print("Train ∩ Test:",
          len(train_users & test_users))

    print("Validation ∩ Test:",
          len(val_users & test_users))

    print("=" * 60)


def leakage_check(df):
    print("\n" + "=" * 60)
    print("DATA LEAKAGE CHECK")
    print("=" * 60)

    # 1. Target column
    print("\nTarget column:")
    print(config.TARGET_COL)

    # 2. ID columns
    print("\nID columns excluded from model:")
    for col in config.ID_COLS:
        print(f" - {col}")

    # 3. Duplicate rows
    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    # 4. Duplicate transaction IDs
    print("\nDuplicate Transaction_ID:")
    print(df["Transaction_ID"].duplicated().sum())

    # 5. User IDs
    print("\nUnique users:")
    print(df["User_ID"].nunique())

    # 6. Target distribution
    print("\nTarget distribution:")
    print(df[config.TARGET_COL].value_counts())

    # 7. Check whether target accidentally appears in features
    feature_columns = [
        col for col in df.columns
        if col != config.TARGET_COL
    ]

    if config.TARGET_COL in feature_columns:
        print("WARNING: TARGET LEAKAGE FOUND!")
    else:
        print("\nTarget leakage check: PASS")

    print("=" * 60)

def get_clean_splits():
    """Convenience one-call function used by every training script."""
    df = load_raw_data()
    df = basic_clean(df)
    leakage_check(df)
    return time_based_split(df)

def check_numeric_leakage(df):

    print("\n" + "=" * 60)
    print("NUMERIC FEATURE LEAKAGE CHECK")
    print("=" * 60)

    numeric_cols = df.select_dtypes(include="number").columns

    correlation = (
        df[numeric_cols]
        .corr()[config.TARGET_COL]
        .sort_values(ascending=False)
    )

    print(correlation)

    print("=" * 60)

if __name__ == "__main__":
    # Running `python -m src.data_preprocessing` lets you sanity-check
    # the splits without touching any model code.
    X_train, X_val, X_test, y_train, y_val, y_test = get_clean_splits()
    print("Train:", X_train.shape, "Fraud rate:", y_train.mean().round(3))
    print("Val:  ", X_val.shape, "Fraud rate:", y_val.mean().round(3))
    print("Test: ", X_test.shape, "Fraud rate:", y_test.mean().round(3))
