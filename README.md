# FRAUDDETECTION-Z
# AI-Powered Fraud Detection System

## 1. Abstract

This project presents an end-to-end machine learning system for detecting potentially fraudulent financial transactions. The system performs data preprocessing, exploratory data analysis, feature engineering, class imbalance handling, baseline modeling, hyperparameter optimization, cross-validation, cost-sensitive threshold selection, and deployment through a Flask API.

## 2. Problem Statement

Financial fraud detection is a binary classification problem where the system must identify suspicious transactions while minimizing false alerts to legitimate customers.

## 3. Objectives

- Analyze transaction data
- Identify fraud-related patterns
- Build classification models
- Handle class imbalance
- Optimize model hyperparameters
- Select a business-oriented decision threshold
- Evaluate the final model
- Deploy the model through an API

## 4. Dataset

The dataset contains 50,000 transaction records and 14 columns.

The target variable is `Fraud_Label`.

## 5. Exploratory Data Analysis

EDA was performed to understand:

- Fraud distribution
- Numerical feature distributions
- Fraud rates across categorical features
- Correlations
- Transaction amount relationships
- Temporal fraud patterns

## 6. Data Preprocessing

The preprocessing pipeline performs:

- Duplicate removal
- Missing target removal
- Date conversion
- Stratified train/validation/test splitting

The resulting splits are:

| Dataset | Records | Fraud Rate |
|---|---:|---:|
| Train | 34,999 | 32.1% |
| Validation | 7,501 | 32.1% |
| Test | 7,500 | 32.1% |

## 7. Feature Engineering

Engineered features include:

- Transaction-to-balance ratio
- High transaction amount indicator
- Transaction day of week
- Transaction month
- Weekend indicator

## 8. Imbalance Handling

The project uses class weighting and XGBoost `scale_pos_weight`.

SMOTE is implemented as an optional experimental approach.

## 9. Models

The following models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

## 10. Hyperparameter Tuning

Optuna was used to search for better XGBoost hyperparameters.

The optimization objective was validation PR-AUC.

## 11. Cross Validation

Stratified 5-fold cross-validation was used to evaluate model stability.

## 12. Evaluation

The primary metrics are:

- PR-AUC
- Recall@FPR
- Precision
- Recall
- F1-score
- Confusion matrix

## 13. Cost-Based Threshold

The final decision threshold was selected using a business cost function that assigns a higher cost to false negatives.

## 14. Deployment

The trained pipeline is saved using Joblib.

The model is served through a Flask REST API and can be deployed using Render.

## 15. Future Scope

Future improvements include:

- Real-time streaming
- Explainable AI
- Data drift monitoring
- Temporal validation
- Database integration
- Authentication
- Automated fraud alerts
- Model monitoring
