
"""
api/main.py
===========

FastAPI serving layer for the Fraud Detection project.

Responsibilities
----------------
1. Load the trained fraud detection pipeline once at startup.
2. Load the optimized decision threshold.
3. Accept a transaction through POST /predict.
4. Apply the same feature engineering used during training.
5. Return:
   - fraud probability
   - fraud prediction
   - decision threshold
   - risk level

Run locally
-----------
From the project root:

    uvicorn api.main:app --reload --port 8000

Swagger UI:

    http://127.0.0.1:8000/docs
"""

import json
import os
import sys

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# ------------------------------------------------------------------
# PROJECT PATH
# ----------------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ------------------------------------------------------------------
# PROJECT IMPORTS
# ------------------------------------------------------------------
from src import config
from src.feature_engineering import prepare_features
from api.schemas import TransactionRequest, FraudPredictionResponse
# ------------------------------------------------------------------
# FASTAPI APPLICATION
# ------------------------------------------------------------------
app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Fraud detection API using a trained XGBoost pipeline "
        "with optimized decision threshold."
    ),
    version="1.0.0",
)


# ------------------------------------------------------------------
# CORS
# ------------------------------------------------------------------
# Development setting.
#
# For production, replace "*" with your actual frontend URL.
# Example:
#
# allow_origins=[
#     "https://your-frontend.onrender.com"
# ]
# ------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# GLOBAL MODEL ARTIFACTS
# ------------------------------------------------------------------

_model_pipeline = None
_threshold = 0.50


# ------------------------------------------------------------------
# LOAD MODEL + THRESHOLD
# ------------------------------------------------------------------

def load_artifacts():
    """
    Load the trained model pipeline and decision threshold.

    The model is loaded once when the application starts instead of
    loading it for every API request.
    """

    global _model_pipeline
    global _threshold

    # --------------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------------

    if not os.path.exists(config.FINAL_MODEL_PATH):
        raise RuntimeError(
            "Trained model was not found.\n"
            f"Expected location:\n{config.FINAL_MODEL_PATH}\n\n"
            "Run:\n"
            "python -m src.train_final"
        )

    # --------------------------------------------------------------
    # LOAD SAVED PIPELINE
    # --------------------------------------------------------------

    _model_pipeline = joblib.load(
        config.FINAL_MODEL_PATH
    )

    # --------------------------------------------------------------
    # LOAD OPTIMIZED THRESHOLD
    # --------------------------------------------------------------

    if os.path.exists(config.THRESHOLD_PATH):

        with open(
            config.THRESHOLD_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            threshold_data = json.load(file)

        # train_final.py should save:
        #
        # {
        #     "threshold": 0.XX
        # }
        #
        # But we also support:
        #
        # {
        #     "best_threshold": 0.XX
        # }

        if "threshold" in threshold_data:

            _threshold = float(
                threshold_data["threshold"]
            )

        elif "best_threshold" in threshold_data:

            _threshold = float(
                threshold_data["best_threshold"]
            )

        else:

            print(
                "WARNING: No threshold value found in "
                "decision_threshold.json."
            )

            _threshold = 0.50

    else:

        print(
            "WARNING: decision_threshold.json not found."
        )

        print(
            "Using default threshold = 0.50"
        )

        _threshold = 0.50

    print("=" * 60)
    print("FRAUD DETECTION API")
    print("=" * 60)
    print(f"Model loaded successfully.")
    print(f"Model path      : {config.FINAL_MODEL_PATH}")
    print(f"Threshold       : {_threshold:.4f}")
    print("=" * 60)


# ------------------------------------------------------------------
# STARTUP EVENT
# ------------------------------------------------------------------

@app.on_event("startup")
def startup_event():
    """
    Load model artifacts when FastAPI starts.
    """

    load_artifacts()


# ------------------------------------------------------------------
# RISK LEVEL
# ------------------------------------------------------------------

def _risk_level(proba: float) -> str:
    """
    Convert fraud probability into a human-readable risk level.

    LOW:
        probability < 0.30

    MEDIUM:
        0.30 <= probability < 0.70

    HIGH:
        probability >= 0.70
    """

    if proba < 0.30:
        return "LOW"

    if proba < 0.70:
        return "MEDIUM"

    return "HIGH"


# ------------------------------------------------------------------
# ROOT ENDPOINT
# ------------------------------------------------------------------

@app.get("/")
def root():

    return {
        "status": "ok",
        "message": "Fraud Detection API is running.",
        "docs": "/docs",
        "health": "/health",
        "prediction_endpoint": "/predict",
    }


# ------------------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": _model_pipeline is not None,
        "threshold": round(_threshold, 4),
    }


# ------------------------------------------------------------------
# PREDICTION ENDPOINT
# ------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=FraudPredictionResponse
)
def predict(request: TransactionRequest):

    # --------------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------------

    if _model_pipeline is None:

        raise HTTPException(
            status_code=503,
            detail="Fraud detection model is not loaded."
        )

    try:

        # ----------------------------------------------------------
        # CONVERT REQUEST TO DATAFRAME
        # ----------------------------------------------------------

        # Pydantic v2:
        #     model_dump()
        #
        # Pydantic v1:
        #     dict()
        #
        # Support both versions.

        if hasattr(request, "model_dump"):

            request_data = request.model_dump()

        else:

            request_data = request.dict()

        raw_df = pd.DataFrame(
            [request_data]
        )

        # ----------------------------------------------------------
        # FEATURE ENGINEERING
        # ----------------------------------------------------------

        # This must use the SAME feature engineering logic
        # used during training.

        features_df = prepare_features(
            raw_df
        )

        # ----------------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------------

        probabilities = (
            _model_pipeline
            .predict_proba(features_df)
        )

        proba = float(
            probabilities[0][1]
        )

        # ----------------------------------------------------------
        # DECISION THRESHOLD
        # ----------------------------------------------------------

        is_fraud = (
            proba >= _threshold
        )

        # ----------------------------------------------------------
        # RISK LEVEL
        # ----------------------------------------------------------

        risk = _risk_level(
            proba
        )

        # ----------------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------------

        return FraudPredictionResponse(

            fraud_probability=round(
                proba,
                4
            ),

            is_fraud=is_fraud,

            threshold_used=round(
                _threshold,
                4
            ),

            risk_level=risk,
        )

    except Exception as e:

        # Do not expose stack traces to API users.

        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )


# ------------------------------------------------------------------
# LOCAL ENTRY POINT
# ------------------------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )