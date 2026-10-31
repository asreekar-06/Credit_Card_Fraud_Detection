"""Credit Card Fraud Detection API (Vercel Python function).

GET  /api/predict  -> thresholds, model name and allowed values
POST /api/predict  -> scores one transaction

The saved scikit-learn pipeline (preprocessing + XGBoost) is loaded as it is.
It is never retrained or modified here.
"""
import json
import os
import warnings
from http.server import BaseHTTPRequestHandler

import joblib
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_model_v2.joblib")

# Decision thresholds: up to LOW is Genuine, from HIGH is Fraud, in between is Review.
LOW = 0.57
HIGH = 0.91

NUMERIC = ["Transaction_Amount", "hour", "day_of_week", "other_state", "Is_International"]
CATEGORICAL = ["Merchant_Category", "Payment_Method", "Transaction_Channel", "Device_Type"]
FEATURES = NUMERIC + CATEGORICAL

DEFAULT_CATEGORIES = {
    "Merchant_Category": ["Airline", "Ecommerce", "Education", "Electronics", "Entertainment",
                          "Fashion", "Financial Services", "Food Delivery", "Fuel", "Grocery",
                          "Hospital", "Hotel", "Pharmacy", "Travel", "Utilities"],
    "Payment_Method": ["Credit Card", "Debit Card", "Net Banking", "UPI"],
    "Transaction_Channel": ["ATM", "Mobile App", "Online Web", "POS"],
    "Device_Type": ["ATM Machine", "Android Mobile", "Mac Desktop", "POS Terminal",
                    "Windows PC", "iOS Mobile"],
}

LABELS = {"genuine": "Genuine", "fraud": "Fraud", "review": "Review"}

_artifact = joblib.load(MODEL_PATH)
PIPELINE = _artifact["pipeline"] if isinstance(_artifact, dict) and "pipeline" in _artifact else _artifact

_saved = _artifact.get("categories") if isinstance(_artifact, dict) else None
_saved = _saved if isinstance(_saved, dict) else {}
CATEGORIES = {key: list(_saved.get(key) or DEFAULT_CATEGORIES[key]) for key in CATEGORICAL}


class ValidationError(ValueError):
    """Raised when the submitted transaction is incomplete or out of range."""


def _number(data, key, low, high, integer=False):
    if key not in data or data[key] in (None, ""):
        raise ValidationError(f"{key} is required.")
    try:
        value = float(data[key])
    except (TypeError, ValueError):
        raise ValidationError(f"{key} must be a number.")
    if value != value or value in (float("inf"), float("-inf")):
        raise ValidationError(f"{key} must be a valid number.")
    if not low <= value <= high:
        raise ValidationError(f"{key} must be between {low} and {high}.")
    if integer and int(value) != value:
        raise ValidationError(f"{key} must be a whole number.")
    return int(value) if integer else value


def validate(data):
    """Checks the nine allowed inputs and returns a clean row."""
    if not isinstance(data, dict):
        raise ValidationError("Send a JSON object with the transaction details.")
    row = {
        "Transaction_Amount": _number(data, "Transaction_Amount", 1, 100_000_000),
        "hour": _number(data, "hour", 0, 23, True),
        "day_of_week": _number(data, "day_of_week", 0, 6, True),
        "other_state": _number(data, "other_state", 0, 1, True),
        "Is_International": _number(data, "Is_International", 0, 1, True),
    }
    for key in CATEGORICAL:
        value = data.get(key)
        if value not in CATEGORIES[key]:
            raise ValidationError(f"{key} must be one of: " + ", ".join(CATEGORIES[key]) + ".")
        row[key] = value
    return row


def decide(probability):
    if probability >= HIGH:
        return "fraud"
    if probability <= LOW:
        return "genuine"
    return "review"


def signals(row):
    """Simple checks on the inputs. These are not the model's internal reasoning."""
    found = []
    if row["Is_International"] == 1:
        found.append("International transaction")
    if row["hour"] <= 5:
        found.append("Late-night transaction (12 am to 5 am)")
    if row["other_state"] == 1:
        found.append("Merchant is in a different state")
    return found


def score(row):
    frame = pd.DataFrame([row], columns=FEATURES)
    for column in NUMERIC:
        frame[column] = frame[column].astype(float)
    probability = float(PIPELINE.predict_proba(frame)[:, 1][0])
    decision = decide(probability)
    return {
        "probability": round(probability, 4),
        "decision": decision,
        "label": LABELS[decision],
        "signals": signals(row),
        "low": LOW,
        "high": HIGH,
    }


def config():
    return {
        "status": "ok",
        "model": "XGBoost (feature set A)",
        "low": LOW,
        "high": HIGH,
        "features": FEATURES,
        "categories": CATEGORIES,
    }


def handle_post(body):
    """Returns (http_status, payload) for a POST body (bytes)."""
    try:
        data = json.loads(body.decode("utf-8") or "{}")
        if isinstance(data, dict) and isinstance(data.get("transaction"), dict):
            data = data["transaction"]
        return 200, score(validate(data))
    except ValidationError as error:
        return 400, {"error": str(error)}
    except json.JSONDecodeError:
        return 400, {"error": "Request body must be valid JSON."}
    except Exception:
        return 500, {"error": "The model could not score this transaction."}


class handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, config())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length > 10_000:
            return self._send(413, {"error": "Request is too large."})
        code, payload = handle_post(self.rfile.read(length))
        self._send(code, payload)