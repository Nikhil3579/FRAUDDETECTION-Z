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
5. Generate fraud probability.
6. Apply the optimized fraud decision threshold.
7. Calculate risk level.
8. Send an SMTP email alert when fraud is detected.

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

# ------------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES FIRST
# ------------------------------------------------------------

from dotenv import load_dotenv

load_dotenv()


# ------------------------------------------------------------
# PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
DASHBOARD_DIR = os.path.join(PROJECT_ROOT, "dashboard")

# ------------------------------------------------------------
# THIRD-PARTY IMPORTS
# ------------------------------------------------------------

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src import config
from src.feature_engineering import prepare_features
from api.schemas import (
    TransactionRequest,
    FraudPredictionResponse,
)
from api.email_alert import send_fraud_alert


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Fraud detection API using a trained XGBoost pipeline "
        "with optimized decision threshold and SMTP email alerts."
    ),
    version="1.1.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GLOBAL MODEL ARTIFACTS
# ============================================================

_model_pipeline = None
_threshold = 0.50


# ============================================================
# LOAD MODEL + THRESHOLD
# ============================================================

def load_artifacts():
    """
    Load the trained model pipeline and decision threshold.
    """

    global _model_pipeline
    global _threshold

    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if not os.path.exists(config.FINAL_MODEL_PATH):

        raise RuntimeError(
            "Trained model was not found.\n"
            f"Expected location:\n"
            f"{config.FINAL_MODEL_PATH}\n\n"
            "Run:\n"
            "python -m src.train_final"
        )

    # --------------------------------------------------------
    # LOAD SAVED MODEL
    # --------------------------------------------------------

    _model_pipeline = joblib.load(
        config.FINAL_MODEL_PATH
    )

    # --------------------------------------------------------
    # LOAD OPTIMIZED THRESHOLD
    # --------------------------------------------------------

    if os.path.exists(config.THRESHOLD_PATH):

        with open(
            config.THRESHOLD_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            threshold_data = json.load(file)

        # Supported format:
        #
        # {
        #     "threshold": 0.20
        # }

        if "threshold" in threshold_data:

            _threshold = float(
                threshold_data["threshold"]
            )

        # Also support:
        #
        # {
        #     "best_threshold": 0.20
        # }

        elif "best_threshold" in threshold_data:

            _threshold = float(
                threshold_data["best_threshold"]
            )

        else:

            print(
                "WARNING: No threshold value found "
                "in decision_threshold.json."
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

    # --------------------------------------------------------
    # STARTUP INFORMATION
    # --------------------------------------------------------

    print("=" * 60)
    print("FRAUD DETECTION API")
    print("=" * 60)

    print("Model loaded successfully.")
    print(
        f"Model path : {config.FINAL_MODEL_PATH}"
    )

    print(
        f"Threshold  : {_threshold:.4f}"
    )

    print(
        f"SMTP user configured : "
        f"{bool(os.getenv('SMTP_USERNAME'))}"
    )

    print(
        f"Alert email configured : "
        f"{bool(os.getenv('ALERT_TO_EMAIL'))}"
    )

    print("=" * 60)


# ============================================================
# STARTUP EVENT
# ============================================================

@app.on_event("startup")
def startup_event():
    """
    Load model artifacts when FastAPI starts.
    """

    load_artifacts()


# ============================================================
# RISK LEVEL
# ============================================================

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


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return FileResponse(
        os.path.join(DASHBOARD_DIR, "index.html")
    )

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": _model_pipeline is not None,
        "threshold": round(
            _threshold,
            4
        ),
        "smtp_configured": bool(
            os.getenv("SMTP_USERNAME")
            and os.getenv("SMTP_PASSWORD")
            and os.getenv("ALERT_TO_EMAIL")
        ),
    }
# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post(
    "/predict",
    response_model=FraudPredictionResponse,
)
def predict(request: TransactionRequest):

    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if _model_pipeline is None:

        raise HTTPException(
            status_code=503,
            detail="Fraud detection model is not loaded.",
        )

    try:

        # ====================================================
        # 1. CONVERT REQUEST TO DATAFRAME
        # ====================================================

        if hasattr(
            request,
            "model_dump",
        ):

            # Pydantic v2
            request_data = request.model_dump()

        else:

            # Pydantic v1
            request_data = request.dict()

        raw_df = pd.DataFrame(
            [request_data]
        )

        # ====================================================
        # 2. FEATURE ENGINEERING
        # ====================================================

        features_df = prepare_features(
            raw_df
        )

        # ====================================================
        # 3. MODEL PREDICTION
        # ====================================================

        probabilities = (
            _model_pipeline
            .predict_proba(
                features_df
            )
        )

        # Probability of fraud class
        proba = float(
            probabilities[0][1]
        )

        # ====================================================
        # 4. FRAUD DECISION
        # ====================================================

        is_fraud = (
            proba >= _threshold
        )

        # ====================================================
        # 5. RISK LEVEL
        # ====================================================

        risk = _risk_level(
            proba
        )

        # ====================================================
        # 6. SMTP EMAIL ALERT
        # ====================================================

        email_sent = False

        if is_fraud:

            email_sent = send_fraud_alert(
                transaction_amount=float(
                    request.Transaction_Amount
                ),
                fraud_probability=proba,
                risk_level=risk,
                threshold=_threshold,
            )

            if email_sent:

                print(
                    "FRAUD ALERT: "
                    "Email notification sent."
                )

            else:

                print(
                    "FRAUD ALERT: "
                    "Email notification could not be sent."
                )

        # ====================================================
        # 7. TERMINAL LOG
        # ====================================================

        print("-" * 60)

        print(
            f"Transaction Amount : "
            f"₹{float(request.Transaction_Amount):.2f}"
        )

        print(
            f"Fraud Probability  : "
            f"{proba * 100:.2f}%"
        )

        print(
            f"Threshold          : "
            f"{_threshold * 100:.2f}%"
        )

        print(
            f"Fraud Decision     : "
            f"{'FRAUD' if is_fraud else 'LEGITIMATE'}"
        )

        print(
            f"Risk Level         : "
            f"{risk}"
        )

        print(
            f"Email Alert        : "
            f"{'SENT' if email_sent else 'NOT SENT'}"
        )

        print("-" * 60)

        # ====================================================
        # 8. API RESPONSE
        # ====================================================

        return FraudPredictionResponse(

            fraud_probability=round(
                proba,
                4,
            ),

            is_fraud=is_fraud,

            threshold_used=round(
                _threshold,
                4,
            ),

            risk_level=risk,
        )

    except Exception as e:

        print(
            f"Prediction error: {str(e)}"
        )

        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}",
        )


app.mount(
    "/",
    StaticFiles(directory=DASHBOARD_DIR, html=True),
    name="dashboard",
)


# ============================================================
# LOCAL ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
