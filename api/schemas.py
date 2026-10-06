"""
schemas.py
-----------
Pydantic models = strict data contracts for the API.
Beginner note: this is what gives you automatic input validation AND
automatic interactive docs at /docs - FastAPI reads these classes and
builds the docs UI for you, no extra work needed.
"""
import sys
import os
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if path not in sys.path:
    sys.path.insert(0, path)
from pydantic import BaseModel, Field
from src import config
from typing import Optional 


class TransactionRequest(BaseModel):
    Transaction_Amount: float = Field(..., ge=0, example=250.50)
    Account_Balance: float = Field(..., example=15000.0)
    Transaction_Type: str = Field(..., example="Online")
    Device_Type: str = Field(..., example="Mobile")
    Location: str = Field(..., example="New York")
    Merchant_Category: str = Field(..., example="Electronics")
    Card_Type: str = Field(..., example="Visa")
    Previous_Fraudulent_Activity: int = Field(..., ge=0, le=1, example=0)
    Daily_Transaction_Count: int = Field(..., ge=0, example=5)
    Card_Age: int = Field(..., ge=0, example=120)
    Date: str = Field(..., example="24-Sep-26")  # format matching training data, e.g. "14-Aug-23"

    class Config:
        json_schema_extra = {
            "example": {
                "Transaction_Amount": 250.50,
                "Account_Balance": 15000.0,
                "Transaction_Type": "Online",
                "Device_Type": "Mobile",
                "Location": "New York",
                "Merchant_Category": "Electronics",
                "Card_Type": "Visa",
                "Previous_Fraudulent_Activity": 0,
                "Daily_Transaction_Count": 5,
                "Card_Age": 120,
                "Date": "24-Sep-26",
            }
        }


class FraudPredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold_used: float
    risk_level: str  # "LOW" / "MEDIUM" / "HIGH" - for a friendlier UI
