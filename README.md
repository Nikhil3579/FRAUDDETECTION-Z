# Fraud Detection ML System — End-to-End Project

## 1. Task Definition

**Problem type:** Binary classification — for each transaction, predict
`Fraud_Label` (0 = genuine, 1 = fraud), output as a **probability**, not
just a hard label, because in production you want a tunable business
threshold, not a fixed one baked into the model.

**Dataset (as uploaded):** 50,000 transactions, 14 columns, ~32% fraud
rate (`synthetic_fraud_dataset1.csv`), no missing values, no duplicate
transaction IDs.

**Chosen model: XGBoost (Gradient Boosted Trees)**
Why this over alternatives:
| Model | Verdict |
|---|---|
| Logistic Regression | Good baseline, but can't capture non-linear interactions (e.g. "high amount AND new device AND weekend" combos) |
| Random Forest | Solid, but usually slightly under XGBoost on tabular data and slower to tune |
| **XGBoost** | **Best fit**: handles mixed numeric/categorical (after encoding) data, non-linear patterns, built-in imbalance handling (`scale_pos_weight`), fast inference (important for real-time fraud checks), tunable with Optuna, tiny serialized size, huge industry track record for fraud/credit risk |
| Deep Learning (MLP/TabNet) | Needs much more data & tuning to beat GBTs on tabular data this size; not worth the complexity here |

## 2. Metrics Used (and why)

- **PR-AUC (Precision-Recall AUC)**: the primary metric. Unlike accuracy
  or ROC-AUC, it isn't inflated by the large number of easy "genuine"
  transactions — it focuses on how well we rank actual fraud cases high.
- **Recall@FPR**: "If we accept flagging at most `TARGET_FPR` (default
  5%) of genuine customers, what % of real fraud do we catch?" This is
  the number you'd actually put in a slide for a business stakeholder.
- **Cost-matrix-optimal threshold**: converts probabilities into a
  yes/no decision by minimizing `FN_count * cost_FN + FP_count * cost_FP`
  instead of the default (usually wrong) 0.5 cutoff.

## 3. Project / Serving Architecture

```
Client (app/website)
      |
      v
FastAPI  /predict endpoint  --------->  loads models/fraud_model.joblib
      |                                 (preprocessing + XGBoost bundled
      |                                  together via sklearn Pipeline)
      v
JSON response: {fraud_probability, is_fraud, threshold_used, risk_level}
```

**Optional Kafka layer** (for high-throughput / async / real-time
streaming instead of simple request-response):

```
Transaction stream --> Kafka topic "transactions" --> Consumer service
                                                         (loads same
                                                          fraud_model.joblib)
                                                              |
                                                              v
                                                   Kafka topic "fraud_scores"
                                                              |
                                                              v
                                                  Downstream systems / DB / alerts
```
You do NOT need Kafka to launch — start with the plain FastAPI endpoint.
Add Kafka later only if you need to process a continuous stream of
transactions asynchronously at very high volume (thousands/sec). A
minimal consumer would just call the same `_model_pipeline.predict_proba()`
logic already in `api/main.py`, wrapped in a Kafka consumer loop instead
of a FastAPI route.

## 4. Folder Structure

```
fraud_detection_project/
├── data/
│   └── synthetic_fraud_dataset1.csv
├── src/
│   ├── __init__.py
│   ├── config.py                  # every path/constant, change things here
│   ├── data_preprocessing.py      # load, clean, split (train/val/test)
│   ├── feature_engineering.py     # new features + ColumnTransformer
│   ├── imbalance_cost.py          # Step 5: class weights, SMOTE, cost matrix
│   ├── evaluate.py                # PR-AUC, Recall@FPR, reports
│   ├── train_baselines.py         # Step 5: Logistic/RandomForest/XGBoost baselines
│   ├── hyperparameter_tuning.py   # Step 6: Optuna tuning
│   └── train_final.py             # cross-validation + final fit + save
├── api/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app (/predict, /health)
│   └── schemas.py                 # request/response validation models
├── models/                        # created automatically: saved .joblib + params
├── reports/                       # created automatically: any exported charts/reports
├── requirements.txt
├── Dockerfile
└── README.md   (this file)
```

## 5. Setup — Step by Step

```bash
# 1) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Data already placed for you in data/synthetic_fraud_dataset1.csv)
#    If using your own file, replace it and keep the same column names,
#    or update src/config.py's column lists.

# 4) Sanity-check the data splitting works
python -m src.data_preprocessing

# 5) Run baseline models (Logistic Regression / Random Forest / XGBoost defaults)
python -m src.train_baselines

# 6) Run Optuna hyperparameter tuning (this is the slow step, ~5-10 min)
python -m src.hyperparameter_tuning

# 7) Train the FINAL model (uses tuned params, runs cross-validation,
#    picks the cost-optimal threshold, evaluates on the untouched test set,
#    and saves everything to models/)
python -m src.train_final

# 8) Launch the API
uvicorn api.main:app --reload --port 8000

# 9) Open the interactive docs in your browser:
#    http://127.0.0.1:8000/docs
#    Click POST /predict -> "Try it out" -> use the example -> Execute
```

### Example request (curl)
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
    "Date": "24-Sep-26"
  }'
```

### Docker deploy
```bash
python -m src.train_final          # produces models/fraud_model.joblib FIRST
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```

## 6. Validation Strategy (why + what's included)

- **Stratified train/val/test split (70/15/15)**: fraud ratio preserved
  in every split (see `data_preprocessing.py`). Test set is touched only
  ONCE, at the very end, in `train_final.py`.
- **5-fold Stratified Cross-Validation** on the training set (inside
  `train_final.py`'s `cross_validate()`): reports mean ± std of PR-AUC
  across 5 different train/validate slices. A small std (e.g. ±0.01)
  means the model is stable; a large std is a red flag worth
  investigating (get more data, simplify the model, check for leakage).
- **Optuna tuning is done only against the validation set** — never the
  test set — so the final test score is an honest, unbiased estimate of
  real-world performance.

## 7. Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: data/synthetic_fraud_dataset1.csv` | Running scripts from the wrong directory | Always run from the project ROOT folder using `python -m src.xxx` (the `-m` matters — it makes relative imports work) |
| `ModuleNotFoundError: No module named 'src'` | Ran `python src/train_final.py` directly instead of as a module | Use `python -m src.train_final` (dot notation, no `.py`) |
| `RuntimeError: Model not found at models/fraud_model.joblib` (from the API) | Tried to launch the API before training | Run `python -m src.train_final` first — this is the file that CREATES the model artifact the API loads |
| `ValueError: Input contains NaN` | New/unexpected raw data has missing values feature_engineering didn't expect | Add `.fillna()` handling for that specific column in `feature_engineering.py`, or filter bad rows before scoring |
| `KeyError` on a column name during `/predict` | The JSON body sent to the API is missing a required field, or field name typo | Check the field names EXACTLY match `api/schemas.py`'s `TransactionRequest` (case-sensitive) |
| Very high train score, much lower validation/test score | Overfitting | Lower `max_depth`, raise `reg_alpha`/`reg_lambda`, increase `min_child_weight`, or add more training data — Optuna's search range in `hyperparameter_tuning.py` already discourages this but check anyway |
| `imblearn` import error when calling `apply_smote()` | `imbalanced-learn` not installed | `pip install imbalanced-learn` (already in requirements.txt) — note SMOTE is OPTIONAL, the default pipeline uses class-weighting instead, which needs no extra package |
| Optuna trial errors / all trials "pruned"/failed | Usually invalid hyperparameter combos or missing package | Make sure `xgboost` version supports the params used; re-check `requirements.txt` versions match |
| Docker container exits immediately | `models/fraud_model.joblib` wasn't created before `docker build` | Run `python -m src.train_final` locally BEFORE building the image — the model file must exist and gets copied in with `COPY . .` |
| CORS error calling the API from a web frontend | Browser blocking cross-origin request | Already handled via `CORSMiddleware` in `api/main.py` (open for dev — restrict `allow_origins` for real production) |

## 8. Next Steps / Production Hardening (optional, later)

- Add authentication (API key / OAuth) to the FastAPI app before exposing publicly.
- Add request/response logging + a monitoring dashboard (data drift, PR-AUC over time).
- Retrain on a schedule as new labeled fraud data comes in (fraud patterns shift over time).
- Add the optional Kafka consumer for streaming volume if/when you need it.
