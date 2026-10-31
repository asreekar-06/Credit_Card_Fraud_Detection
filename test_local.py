"""Checks the saved model and prints predictions.  Run:  python test_local.py"""
import csv
import os
from collections import Counter

import joblib

from api import predict as P

artifact = joblib.load(P.MODEL_PATH)
if isinstance(artifact, dict):
    print("Saved model -> feature set:", artifact.get("feature_set"),
          "| model:", artifact.get("model"),
          "| thresholds stored in file:", artifact.get("low"), artifact.get("high"))
print("Thresholds used by the app -> low:", P.LOW, "| high:", P.HIGH)

examples = [
    {"Transaction_Amount": 2500, "hour": 14, "day_of_week": 2, "other_state": 0,
     "Is_International": 0, "Merchant_Category": "Grocery", "Payment_Method": "UPI",
     "Transaction_Channel": "Mobile App", "Device_Type": "Android Mobile"},
    {"Transaction_Amount": 95000, "hour": 2, "day_of_week": 5, "other_state": 1,
     "Is_International": 1, "Merchant_Category": "Electronics", "Payment_Method": "Credit Card",
     "Transaction_Channel": "Online Web", "Device_Type": "Windows PC"},
]
for row in examples:
    print(P.score(P.validate(row)))

path = os.path.join("public", "sample_transactions_v2.csv")
if os.path.exists(path):
    counts = Counter()
    with open(path, newline="") as handle:
        for record in csv.DictReader(handle):
            result = P.score(P.validate({key: record[key] for key in P.FEATURES}))
            counts[(record["Fraud_Flag"], result["decision"])] += 1
    print("Sample file: (actual label 1=fraud, decision) -> count")
    for key, value in sorted(counts.items()):
        print(key, value)
else:
    print("public/sample_transactions_v2.csv not found (add it before Part 2).")