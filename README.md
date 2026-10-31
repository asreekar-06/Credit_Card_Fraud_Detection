# Credit Card Fraud Detection

A web application that helps a cardholder check whether a card transaction looks like fraud. It never asks for a card number, CVV, PIN or any personal details. The result is one of three categories: **Genuine**, **Review** or **Fraud**, with an estimated fraud probability.

**Live demo:** https://creditcardfrauddetectionfinal-92arn1yaa-cc-fraud-detection.vercel.app

## How it works

The user enters nine transaction details. A saved scikit-learn pipeline (preprocessing + XGBoost) estimates the fraud probability, and two thresholds turn it into a decision:

| Probability | Decision |
|---|---|
| 0.57 or below | Genuine |
| Between 0.57 and 0.91 | Review |
| 0.91 or above | Fraud |

Inputs (feature set A): transaction amount, hour, day of week, merchant in a different state, international transaction, merchant category, payment method, transaction channel, and device type.

## Dataset

[Indian Financial Fraud Dataset](https://www.kaggle.com/datasets/jatinkhandelwal112/indian-financial-fraud-dataset) (Kaggle, jatinkhandelwal112): **synthetic** data with 250,000 transactions, of which 13,473 (5.39%) are fraud. The dataset is not included in this repository.

Setup: chronological split (70% train, 15% validation, 15% test = 175,000 / 37,500 / 37,500 rows), numeric features standardised, categories one-hot encoded, class weights for imbalance. `Fraud_Reason` and `Transaction_Status` were excluded because they leak the label.

## Results (test period, threshold 0.5)

| Features | Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| Set A (used in the app) | Logistic Regression | 0.140 | 0.397 | 0.207 | 0.308 |
| Set A | Linear SVM | 0.960 | 0.141 | 0.247 | 0.310 |
| Set A | Random Forest | 0.470 | 0.314 | 0.376 | 0.376 |
| Set A | **XGBoost** | 0.142 | 0.659 | 0.234 | **0.429** |
| Set B (bank-side, experimental) | Logistic Regression | 0.394 | 0.724 | 0.510 | 0.716 |
| Set B | Linear SVM | 0.923 | 0.561 | 0.698 | 0.710 |
| Set B | Random Forest | 0.896 | 0.625 | 0.737 | 0.765 |
| Set B | XGBoost | 0.528 | 0.765 | 0.625 | 0.777 |

Set B adds card and customer information that only a bank has (card status, credit limit, and so on) and is not used in the app. Because a cardholder cannot supply that information, the deployed Set A model is much weaker, and this is a deliberate trade-off for privacy.

## Project structure

```
api/predict.py                    Python API (Vercel function)
model/fraud_model_v2.joblib       saved preprocessing + XGBoost pipeline
public/index.html                 web page
public/sample_transactions_v2.csv sample transactions for the demo buttons
run_local.py                      local server
test_local.py                     checks the saved model
requirements.txt, pyproject.toml, vercel.json   dependencies and deployment settings
```

## Run locally

Requires Python 3.10 or newer.

```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python test_local.py
python run_local.py
```

Open http://localhost:8000.

## Deploy

Push to GitHub, then import the repository at vercel.com/new (framework preset: Other) and click Deploy.

## Limitations

- The dataset is synthetic, so results do not show how the model would perform on real bank data.
- The Set A model has low precision at the default threshold and may produce false alarms.
- Predictions are estimates, not proof of fraud. Always confirm with your bank.
- Nothing entered is stored by this application.

## Team

Addepalli Sreekar, Adepoju Sandeep Chary, Bushipaka Vamshi, Nara Abhirama
Guide: Ms. V. Veda Sri, Assistant Professor, CSE (AI & ML), Samskruti College of Engineering & Technology, Hyderabad
