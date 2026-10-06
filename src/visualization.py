"""
visualization.py
----------------

Exploratory Data Analysis for Fraud Detection.

This module:
1. Loads the raw fraud dataset.
2. Displays basic data information.
3. Analyzes fraud distribution.
4. Analyzes numerical variables.
5. Compares numerical variables by fraud status.
6. Analyzes categorical fraud rates.
7. Generates a correlation heatmap.
8. Analyzes important feature relationships.
9. Analyzes fraud rate over time.
10. Saves all plots into outputs/visualization/.

This module does NOT train any machine learning model.
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from src import config


# ============================================================
# PATHS
# ============================================================

DATA_PATH = config.RAW_DATA_PATH

OUTPUT_DIR = config.OUTPUT_DIR

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def save_plot(filename):
    """
    Save the current matplotlib figure.
    """

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {output_path}")


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("LOADING FRAUD DATASET")
    print("=" * 70)

    print(f"Dataset path: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print(f"\nRows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")

    print("\nColumns:")
    for column in df.columns:
        print(f" - {column}")

    print("\nMissing values:")

    print(
        df.isnull().sum()
    )

    return df


# ============================================================
# BASIC SUMMARY
# ============================================================

def basic_summary(df):

    print("\n" + "=" * 70)
    print("BASIC DATA SUMMARY")
    print("=" * 70)

    print("\nData types:")

    print(
        df.dtypes
    )

    print("\nNumerical statistics:")

    print(
        df.describe()
    )

    print("\nFraud distribution:")

    print(
        df[config.TARGET_COL].value_counts()
    )

    print("\nFraud percentage:")

    print(
        df[config.TARGET_COL]
    .value_counts(normalize=True)
    .mul(100)
    )


# ============================================================
# 1. FRAUD DISTRIBUTION
# ============================================================

def plot_fraud_distribution(df):

    counts = (
        df[config.TARGET_COL]
        .value_counts()
        .sort_index()
    )

    labels = [
        "Non-Fraud",
        "Fraud"
    ]

    plt.figure(
        figsize=(8, 5)
    )

    sns.barplot(
        x=labels,
        y=counts.values
    )

    plt.title(
        "Fraud vs Non-Fraud Transactions"
    )

    plt.xlabel(
        "Transaction Class"
    )

    plt.ylabel(
        "Number of Transactions"
    )

    for i, value in enumerate(
        counts.values
    ):

        plt.text(
            i,
            value,
            f"{value:,}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    save_plot(
        "01_fraud_distribution.png"
    )

    plt.show()

    plt.close()


# ============================================================
# 2. FRAUD PERCENTAGE
# ============================================================

def plot_fraud_percentage(df):

    fraud_rate = (
        df[config.TARGET_COL]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
    )

    labels = [
        "Non-Fraud",
        "Fraud"
    ]

    plt.figure(
        figsize=(8, 5)
    )

    sns.barplot(
        x=labels,
        y=fraud_rate.values
    )

    plt.title(
        "Fraud Rate"
    )

    plt.xlabel(
        "Transaction Class"
    )

    plt.ylabel(
        "Percentage (%)"
    )

    for i, value in enumerate(
        fraud_rate.values
    ):

        plt.text(
            i,
            value,
            f"{value:.2f}%",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    save_plot(
        "02_fraud_percentage.png"
    )

    plt.show()

    plt.close()


# ============================================================
# 3. NUMERICAL DISTRIBUTIONS
# ============================================================

def plot_numerical_distributions(df):

    numerical_columns = [
        "Transaction_Amount",
        "Account_Balance",
        "Previous_Fraudulent_Activity",
        "Daily_Transaction_Count",
        "Card_Age"
    ]

    for column in numerical_columns:

        plt.figure(
            figsize=(8, 5)
        )

        sns.histplot(
            data=df,
            x=column,
            bins=40,
            kde=True
        )

        plt.title(
            f"Distribution of {column}"
        )

        plt.xlabel(
            column
        )

        plt.ylabel(
            "Frequency"
        )

        plt.tight_layout()

        filename = (
            column.lower()
            .replace(" ", "_")
            + "_distribution.png"
        )

        save_plot(
            filename
        )

        plt.show()

        plt.close()


# ============================================================
# 4. NUMERICAL FEATURES BY FRAUD
# ============================================================

def plot_numerical_by_fraud(df):

    numerical_columns = [
        "Transaction_Amount",
        "Account_Balance",
        "Previous_Fraudulent_Activity",
        "Daily_Transaction_Count",
        "Card_Age"
    ]

    for column in numerical_columns:

        plt.figure(
            figsize=(8, 5)
        )

        sns.boxplot(
            data=df,
            x=config.TARGET_COL,
            y=column
        )

        plt.title(
            f"{column} by Fraud Status"
        )

        plt.xlabel(
            "Fraud Label (0 = Non-Fraud, 1 = Fraud)"
        )

        plt.ylabel(
            column
        )

        plt.tight_layout()

        filename = (
            column.lower()
            + "_by_fraud.png"
        )

        save_plot(
            filename
        )

        plt.show()

        plt.close()


# ============================================================
# 5. CATEGORICAL FRAUD ANALYSIS
# ============================================================

def plot_categorical_fraud(df):

    categorical_columns = [
        "Transaction_Type",
        "Device_Type",
        "Location",
        "Merchant_Category",
        "Card_Type"
    ]

    for column in categorical_columns:

        fraud_rate = (
            df.groupby(column)[config.TARGET_COL]
            .mean()
            .sort_values(ascending=False)
            .mul(100)
        )

        plt.figure(
            figsize=(10, 6)
        )

        sns.barplot(
            x=fraud_rate.values,
            y=fraud_rate.index
        )

        plt.title(
            f"Fraud Rate by {column}"
        )

        plt.xlabel(
            "Fraud Rate (%)"
        )

        plt.ylabel(
            column
        )

        plt.tight_layout()

        filename = (
            column.lower()
            + "_fraud_rate.png"
        )

        save_plot(
            filename
        )

        plt.show()

        plt.close()


# ============================================================
# 6. CORRELATION HEATMAP
# ============================================================

def plot_correlation_heatmap(df):

    numerical_df = df.select_dtypes(
        include=["int64", "float64"]
    )

    correlation = numerical_df.corr()

    plt.figure(
        figsize=(10, 7)
    )

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title(
        "Numerical Feature Correlation Heatmap"
    )

    plt.tight_layout()

    save_plot(
        "correlation_heatmap.png"
    )

    plt.show()

    plt.close()


# ============================================================
# 7. TRANSACTION AMOUNT VS BALANCE
# ============================================================

def plot_amount_vs_balance(df):

    plt.figure(
        figsize=(9, 6)
    )

    sns.scatterplot(
        data=df,
        x="Transaction_Amount",
        y="Account_Balance",
        hue=config.TARGET_COL,
        alpha=0.5
    )

    plt.title(
        "Transaction Amount vs Account Balance"
    )

    plt.xlabel(
        "Transaction Amount"
    )

    plt.ylabel(
        "Account Balance"
    )

    plt.tight_layout()

    save_plot(
        "transaction_amount_vs_balance.png"
    )

    plt.show()

    plt.close()


# ============================================================
# 8. DAILY COUNT VS TRANSACTION AMOUNT
# ============================================================

def plot_amount_vs_daily_count(df):

    plt.figure(
        figsize=(9, 6)
    )

    sns.scatterplot(
        data=df,
        x="Daily_Transaction_Count",
        y="Transaction_Amount",
        hue=config.TARGET_COL,
        alpha=0.5
    )

    plt.title(
        "Daily Transaction Count vs Transaction Amount"
    )

    plt.xlabel(
        "Daily Transaction Count"
    )

    plt.ylabel(
        "Transaction Amount"
    )

    plt.tight_layout()

    save_plot(
        "daily_count_vs_transaction_amount.png"
    )

    plt.show()

    plt.close()


# ============================================================
# 9. FRAUD RATE OVER TIME
# ============================================================

def plot_fraud_over_time(df):

    temp = df.copy()

    temp[config.DATE_COL] = pd.to_datetime(
        temp[config.DATE_COL],
        errors="coerce"
    )

    daily_fraud = (
        temp
        .dropna(subset=[config.DATE_COL])
        .groupby(
            temp[config.DATE_COL].dt.date
        )[config.TARGET_COL]
        .mean()
        .mul(100)
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        daily_fraud.index,
        daily_fraud.values
    )

    plt.title(
        "Fraud Rate Over Time"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Fraud Rate (%)"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    save_plot(
        "fraud_rate_over_time.png"
    )

    plt.show()

    plt.close()


# ============================================================
# 10. FRAUD RATE SUMMARY
# ============================================================

def fraud_rate_summary(df):

    categorical_columns = [
        "Transaction_Type",
        "Device_Type",
        "Location",
        "Merchant_Category",
        "Card_Type"
    ]

    print("\n" + "=" * 70)
    print("FRAUD RATE BY CATEGORY")
    print("=" * 70)

    for column in categorical_columns:

        print(
            f"\n--- {column} ---"
        )

        summary = (
            df.groupby(column)[config.TARGET_COL]
            .agg(
                transactions="count",
                fraud_count="sum",
                fraud_rate="mean"
            )
        )

        summary["fraud_rate"] *= 100

        print(
            summary.sort_values(
                "fraud_rate",
                ascending=False
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    basic_summary(df)

    print("\nGenerating visualizations...")

    plot_fraud_distribution(df)

    plot_fraud_percentage(df)

    plot_numerical_distributions(df)

    plot_numerical_by_fraud(df)

    plot_categorical_fraud(df)

    plot_correlation_heatmap(df)

    plot_amount_vs_balance(df)

    plot_amount_vs_daily_count(df)

    plot_fraud_over_time(df)

    fraud_rate_summary(df)

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)

    print(
        f"All plots saved to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()