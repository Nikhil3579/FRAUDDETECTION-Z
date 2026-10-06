import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "DATA",
    "synthetic_fraud_dataset1.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "fraud_model.joblib"
)

THRESHOLD_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "decision_threshold.json"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Fraud Detection ML Insights",
    page_icon="🔐",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data
def load_threshold():

    if not os.path.exists(THRESHOLD_PATH):
        return 0.50

    with open(THRESHOLD_PATH, "r") as file:
        data = json.load(file)

    if "best_threshold" in data:
        return float(data["best_threshold"])

    if "threshold" in data:
        return float(data["threshold"])

    return 0.50


# ============================================================
# LOAD
# ============================================================

try:
    df = load_data()
except Exception as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()


model = load_model()
threshold = load_threshold()


# ============================================================
# HEADER
# ============================================================

st.title("🔐 AI-Powered Fraud Detection — ML Insights Dashboard")

st.markdown(
    """
    **Purpose:** Understand the dataset, fraud patterns, machine
    learning performance, threshold optimization, and business
    insights behind the fraud detection system.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "EDA",
        "Numerical Analysis",
        "Categorical Analysis",
        "Model Performance",
        "Threshold Analysis",
        "Business Insights"
    ]
)


# ============================================================
# COMMON VARIABLES
# ============================================================

target = "Fraud_Label"

fraud_count = int(
    (df[target] == 1).sum()
)

non_fraud_count = int(
    (df[target] == 0).sum()
)

total_transactions = len(df)

fraud_rate = (
    fraud_count / total_transactions * 100
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "Fraud Transactions",
        f"{fraud_count:,}"
    )

    col3.metric(
        "Non-Fraud Transactions",
        f"{non_fraud_count:,}"
    )

    col4.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )

    st.subheader("Dataset Information")

    info_col1, info_col2 = st.columns(2)

    with info_col1:

        st.write("**Rows:**", df.shape[0])
        st.write("**Columns:**", df.shape[1])

        if "User_ID" in df.columns:
            st.write(
                "**Unique Users:**",
                df["User_ID"].nunique()
            )

        if "Transaction_ID" in df.columns:
            st.write(
                "**Unique Transactions:**",
                df["Transaction_ID"].nunique()
            )

    with info_col2:

        st.write(
            "**Fraud Transactions:**",
            f"{fraud_count:,}"
        )

        st.write(
            "**Non-Fraud Transactions:**",
            f"{non_fraud_count:,}"
        )

        st.write(
            "**Fraud Rate:**",
            f"{fraud_rate:.2f}%"
        )

    st.subheader("Target Distribution")

    target_counts = df[target].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.barplot(
        x=["Non-Fraud", "Fraud"],
        y=[
            target_counts.get(0, 0),
            target_counts.get(1, 0)
        ],
        ax=ax
    )

    ax.set_xlabel("Transaction Class")
    ax.set_ylabel("Number of Transactions")
    ax.set_title("Fraud vs Non-Fraud Transactions")

    st.pyplot(fig)

    st.info(
        f"""
        **Interview insight:** The dataset contains approximately
        {fraud_rate:.2f}% fraudulent transactions. Because fraud
        detection is a classification problem with an imbalanced
        target, accuracy alone should not be the primary metric.
        """
    )


# ============================================================
# EDA
# ============================================================

elif page == "EDA":

    st.header("📈 Exploratory Data Analysis")

    st.subheader("Data Types")

    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values,
        "Unique Values": [
            df[column].nunique()
            for column in df.columns
        ]
    })

    st.dataframe(
        dtype_df,
        use_container_width=True
    )

    st.subheader("Fraud Distribution")

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.countplot(
        data=df,
        x=target,
        ax=ax
    )

    ax.set_title("Fraud Label Distribution")
    ax.set_xlabel("Fraud Label")
    ax.set_ylabel("Transaction Count")

    st.pyplot(fig)

    if "Date" in df.columns:

        st.subheader("Fraud Trend by Month")

        temp = df.copy()

        temp["Date"] = pd.to_datetime(
            temp["Date"],
            errors="coerce"
        )

        temp["Month"] = temp["Date"].dt.month

        monthly = (
            temp.groupby("Month")[target]
            .agg(["count", "sum"])
            .reset_index()
        )

        monthly["fraud_rate"] = (
            monthly["sum"] /
            monthly["count"] *
            100
        )

        fig, ax = plt.subplots(figsize=(10, 4))

        sns.lineplot(
            data=monthly,
            x="Month",
            y="fraud_rate",
            marker="o",
            ax=ax
        )

        ax.set_title("Monthly Fraud Rate")
        ax.set_xlabel("Month")
        ax.set_ylabel("Fraud Rate (%)")

        st.pyplot(fig)


# ============================================================
# NUMERICAL ANALYSIS
# ============================================================

elif page == "Numerical Analysis":

    st.header("🔢 Numerical Feature Analysis")

    numeric_columns = [
        column
        for column in [
            "Transaction_Amount",
            "Account_Balance",
            "Daily_Transaction_Count",
            "Card_Age"
        ]
        if column in df.columns
    ]

    selected_column = st.selectbox(
        "Select Numerical Feature",
        numeric_columns
    )

    st.subheader(
        f"{selected_column}: Fraud vs Non-Fraud"
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.boxplot(
        data=df,
        x=target,
        y=selected_column,
        ax=ax
    )

    ax.set_title(
        f"{selected_column} Distribution by Fraud Label"
    )

    st.pyplot(fig)

    st.subheader("Summary Statistics")

    summary = (
        df.groupby(target)[selected_column]
        .agg([
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max"
        ])
    )

    st.dataframe(
        summary,
        use_container_width=True
    )

    if (
        "Transaction_Amount" in df.columns
        and "Account_Balance" in df.columns
    ):

        st.subheader(
            "Transaction Amount vs Account Balance"
        )

        plot_df = df.sample(
            min(5000, len(df)),
            random_state=42
        )

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        sns.scatterplot(
            data=plot_df,
            x="Account_Balance",
            y="Transaction_Amount",
            hue=target,
            alpha=0.5,
            ax=ax
        )

        ax.set_title(
            "Transaction Amount vs Account Balance"
        )

        st.pyplot(fig)


# ============================================================
# CATEGORICAL ANALYSIS
# ============================================================

elif page == "Categorical Analysis":

    st.header("🏷️ Categorical Feature Analysis")

    categorical_columns = [
        column
        for column in [
            "Transaction_Type",
            "Device_Type",
            "Location",
            "Merchant_Category",
            "Card_Type"
        ]
        if column in df.columns
    ]

    selected_column = st.selectbox(
        "Select Categorical Feature",
        categorical_columns
    )

    analysis = (
        df.groupby(selected_column)[target]
        .agg(
            transaction_count="count",
            fraud_count="sum"
        )
        .reset_index()
    )

    analysis["fraud_rate"] = (
        analysis["fraud_count"] /
        analysis["transaction_count"] *
        100
    )

    st.subheader(
        f"Fraud Rate by {selected_column}"
    )

    display_df = analysis.sort_values(
        "fraud_rate",
        ascending=False
    )

    st.dataframe(
        display_df,
        use_container_width=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.barplot(
        data=display_df,
        x="fraud_rate",
        y=selected_column,
        ax=ax
    )

    ax.set_title(
        f"Fraud Rate by {selected_column}"
    )

    ax.set_xlabel("Fraud Rate (%)")
    ax.set_ylabel(selected_column)

    st.pyplot(fig)

    st.info(
        """
        **Interview insight:** I use fraud rate rather than only
        fraud count because categories with more transactions will
        naturally have more fraud cases. Fraud rate provides a more
        meaningful comparison between categories.
        """
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header("🤖 Model Performance")

    if model is None:

        st.warning(
            "Saved fraud_model.joblib was not found. "
            "Train the model first."
        )

    else:

        st.success(
            "Saved fraud detection model loaded successfully."
        )

        st.write(
            f"Decision threshold: **{threshold:.2f}**"
        )

        st.info(
            """
            The saved model is used for inference. The dashboard
            does not retrain the model.
            """
        )

        st.subheader(
            "Model Architecture"
        )

        st.code(
            """
Transaction Data
       ↓
Preprocessing
       ↓
One-Hot Encoding
       ↓
Feature Engineering
       ↓
XGBoost
       ↓
Fraud Probability
       ↓
Decision Threshold
       ↓
Fraud / Non-Fraud
            """,
            language="text"
        )

        st.subheader(
            "Model Evaluation Concept"
        )

        st.markdown(
            """
            **Precision:** How many predicted fraud cases were
            actually fraudulent?

            **Recall:** How many actual fraud cases were detected?

            **F1-score:** Harmonic mean of precision and recall.

            **PR-AUC:** Measures precision-recall performance
            across classification thresholds.

            **ROC-AUC:** Measures ranking/discrimination ability
            across thresholds.
            """
        )


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

elif page == "Threshold Analysis":

    st.header("🎯 Fraud Decision Threshold")

    st.metric(
        "Current Optimized Threshold",
        f"{threshold:.2f}"
    )

    st.write(
        f"""
        The system currently uses approximately **{threshold:.0%}**
        as the fraud decision threshold.
        """
    )

    st.subheader(
        "Why not simply use 0.50?"
    )

    st.markdown(
        """
        A machine learning classifier normally uses 0.50 as a
        default probability threshold.

        In fraud detection, however, the cost of missing fraud
        can be significantly higher than the cost of investigating
        a false alarm.

        Therefore, I optimized the threshold using validation data
        rather than automatically using 0.50.
        """
    )

    threshold_values = np.arange(
        0.05,
        0.96,
        0.05
    )

    if model is not None:

        try:

            feature_df = df.drop(
                columns=[target],
                errors="ignore"
            )

            probabilities = model.predict_proba(
                feature_df
            )[:, 1]

            threshold_results = []

            for t in threshold_values:

                predictions = (
                    probabilities >= t
                ).astype(int)

                precision = precision_score(
                    df[target],
                    predictions,
                    zero_division=0
                )

                recall = recall_score(
                    df[target],
                    predictions,
                    zero_division=0
                )

                f1 = f1_score(
                    df[target],
                    predictions,
                    zero_division=0
                )

                threshold_results.append({
                    "threshold": t,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                })

            threshold_df = pd.DataFrame(
                threshold_results
            )

            st.dataframe(
                threshold_df,
                use_container_width=True
            )

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            ax.plot(
                threshold_df["threshold"],
                threshold_df["precision"],
                marker="o",
                label="Precision"
            )

            ax.plot(
                threshold_df["threshold"],
                threshold_df["recall"],
                marker="o",
                label="Recall"
            )

            ax.plot(
                threshold_df["threshold"],
                threshold_df["f1"],
                marker="o",
                label="F1-score"
            )

            ax.axvline(
                threshold,
                linestyle="--",
                label="Selected Threshold"
            )

            ax.set_xlabel(
                "Decision Threshold"
            )

            ax.set_ylabel(
                "Metric"
            )

            ax.set_title(
                "Precision, Recall and F1 vs Threshold"
            )

            ax.legend()

            st.pyplot(fig)

        except Exception as error:

            st.warning(
                f"""
                Threshold visualization could not be generated:
                {error}

                This can happen if the saved pipeline expects
                engineered features that are not present in the
                raw dataset.
                """
            )


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

elif page == "Business Insights":

    st.header("💡 Business Insights")

    st.subheader(
        "What does this system provide to a business?"
    )

    st.markdown(
        """
        ### 1. Fraud Prioritization

        The model provides a fraud probability so transactions can
        be prioritized for investigation.

        ### 2. Risk-Based Decision Making

        Transactions can be categorized into LOW, MEDIUM and HIGH
        risk levels.

        ### 3. Reduced Manual Investigation

        Instead of manually reviewing every transaction, analysts
        can focus on transactions with higher fraud probability.

        ### 4. Cost-Aware Detection

        The decision threshold can be adjusted according to the
        business cost of false negatives and false positives.

        ### 5. Automated Alerts

        When a transaction crosses the fraud threshold, the system
        can trigger an email alert.

        ### 6. Explainable Analysis

        EDA charts allow analysts to understand patterns in
        transaction amount, device, merchant category, location,
        transaction type and other variables.
        """
    )

    st.subheader(
        "End-to-End System"
    )

    st.code(
        """
Customer Transaction
        ↓
Data Processing
        ↓
Feature Engineering
        ↓
XGBoost Model
        ↓
Fraud Probability
        ↓
Decision Threshold
        ↓
Risk Level
        ↓
Dashboard
        ↓
Fraud Alert
        """,
        language="text"
    )

    st.success(
        """
        **Final takeaway:** This project combines machine learning,
        data analysis, model evaluation, API development,
        visualization and automated alerting into an end-to-end
        fraud detection solution.
        """
    )