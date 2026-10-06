import os
import smtplib
from email.message import EmailMessage


# ============================================================
# SMTP CONFIGURATION
# ============================================================

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

ALERT_TO_EMAIL = os.getenv("ALERT_TO_EMAIL")


# ============================================================
# SEND FRAUD ALERT
# ============================================================

def send_fraud_alert(
    transaction_amount: float,
    fraud_probability: float,
    risk_level: str,
    threshold: float
) -> bool:
    """
    Send an email alert when a transaction is detected as fraud.

    Returns:
        True  -> email sent successfully
        False -> email could not be sent
    """

    print("=" * 60)
    print("EMAIL ALERT PROCESS STARTED")
    print("=" * 60)

    # --------------------------------------------------------
    # CHECK CONFIGURATION
    # --------------------------------------------------------

    print(f"SMTP Host       : {SMTP_HOST}")
    print(f"SMTP Port       : {SMTP_PORT}")
    print(
        f"SMTP Username   : "
        f"{'Configured' if SMTP_USERNAME else 'Missing'}"
    )
    print(
        f"SMTP Password   : "
        f"{'Configured' if SMTP_PASSWORD else 'Missing'}"
    )
    print(
        f"Alert Recipient : "
        f"{ALERT_TO_EMAIL if ALERT_TO_EMAIL else 'Missing'}"
    )

    if not SMTP_USERNAME:
        print("ERROR: SMTP_USERNAME is missing.")
        return False

    if not SMTP_PASSWORD:
        print("ERROR: SMTP_PASSWORD is missing.")
        return False

    if not ALERT_TO_EMAIL:
        print("ERROR: ALERT_TO_EMAIL is missing.")
        return False

    # --------------------------------------------------------
    # CREATE EMAIL
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # SMTP CONNECTION
    # --------------------------------------------------------

    try:

        print("Connecting to SMTP server...")

        with smtplib.SMTP(
            SMTP_HOST,
            SMTP_PORT,
            timeout=20
        ) as server:

            print("SMTP connection established.")

            # ------------------------------------------------
            # START TLS
            # ------------------------------------------------

            print("Starting TLS encryption...")

            server.starttls()

            print("TLS encryption established.")

            # ------------------------------------------------
            # LOGIN
            # ------------------------------------------------

            print("Logging in to SMTP server...")

            server.login(
                SMTP_USERNAME,
                SMTP_PASSWORD
            )

            print("SMTP login successful.")

            # ------------------------------------------------
            # SEND EMAIL
            # ------------------------------------------------

            print("Sending fraud alert email...")

            server.send_message(message)

            print("Fraud alert email sent successfully.")

        print("=" * 60)
        print("EMAIL ALERT PROCESS COMPLETED")
        print("=" * 60)

        return True

    # --------------------------------------------------------
    # AUTHENTICATION ERROR
    # --------------------------------------------------------

    except smtplib.SMTPAuthenticationError as error:

        print("=" * 60)
        print("SMTP AUTHENTICATION ERROR")
        print("=" * 60)

        print(
            "Gmail rejected the SMTP username/password."
        )

        print(f"Error: {error}")

        print(
            "Make sure SMTP_PASSWORD is a Gmail "
            "16-character App Password."
        )

        return False

    # --------------------------------------------------------
    # SMTP CONNECTION ERROR
    # --------------------------------------------------------

    except smtplib.SMTPConnectError as error:

        print("=" * 60)
        print("SMTP CONNECTION ERROR")
        print("=" * 60)

        print(
            "Could not connect to the Gmail SMTP server."
        )

        print(f"Error: {error}")

        return False

    # --------------------------------------------------------
    # SMTP ERROR
    # --------------------------------------------------------

    except smtplib.SMTPException as error:

        print("=" * 60)
        print("SMTP ERROR")
        print("=" * 60)

        print(f"Error: {error}")

        return False

    # --------------------------------------------------------
    # OTHER ERROR
    # --------------------------------------------------------

    except Exception as error:

        print("=" * 60)
        print("EMAIL ALERT ERROR")
        print("=" * 60)

        print(
            f"{type(error).__name__}: {error}"
        )

<<<<<<< HEAD
        return False
=======
        return False
>>>>>>> e5e94df0b7ede91b945c02911ce19b86e6ca74b1
