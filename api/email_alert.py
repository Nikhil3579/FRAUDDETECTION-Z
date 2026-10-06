import os
import smtplib
from email.message import EmailMessage


SMTP_HOST = os.getenv("SMTP_HOST", "NB.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

ALERT_TO_EMAIL = os.getenv("ALERT_TO_EMAIL")


def send_fraud_alert(
    transaction_amount: float,
    fraud_probability: float,
    risk_level: str,
    threshold: float
):
    """
    Send an email alert when a transaction is detected as fraud.
    """

    if not SMTP_USERNAME or not SMTP_PASSWORD or not ALERT_TO_EMAIL:
        print("SMTP email settings are not configured.")
        return False

    message = EmailMessage()

    message["Subject"] = "🚨 Fraud Detection Alert"
    message["From"] = SMTP_USERNAME
    message["To"] = ALERT_TO_EMAIL

    message.set_content(
        f"""
Fraud Detection System Alert

A potentially fraudulent transaction has been detected.

Transaction Amount : ₹{transaction_amount:.2f}
Fraud Probability  : {fraud_probability * 100:.2f}%
Decision Threshold  : {threshold * 100:.2f}%
Risk Level          : {risk_level}

Action:
The transaction has been flagged for review.

This is an automated message from the Fraud Detection System.
"""
    )

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(message)

        print("Fraud alert email sent successfully.")
        return True

    except Exception as error:
        print(f"Failed to send fraud alert email: {error}")
        return False
