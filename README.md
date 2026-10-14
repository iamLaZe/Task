# Kestrel warranty-claim fraud flag

Ranks warranty claims by fraud risk so the 40-claims-a-month investigation desk can check the riskiest first.

## Run (clean machine, no API key, no internet needed after install)
```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 app.py          # open http://localhost:8000 (screen); POST http://localhost:8000/score (endpoint)
# Screen: ranked review queue (top 40 per month, click a row for reasons), pick-a-claim checker, CSV upload.
```
`artifacts/model.joblib` is committed, so the app starts immediately. To rebuild everything from the raw data:
`python3 train.py && python3 predict.py && python3 evaluate_final.py && python3 evaluate.py`

Example:
```
curl -X POST localhost:8000/score -H 'Content-Type: application/json' -d '{"claim_id":"X","submitted_at":"2026-09-15 11:30","partner_id":"SP3160","sku":"KH-AF-02","product_serial":"kh123","days_since_purchase":200,"claim_amount_inr":1500,"photo_attached":"N","partner_inspected":"N","customer_prior_claims":2}'
```
Returns `fraud_score`, `recommendation` (HIGH / MEDIUM / LOW) and plain-English `reasons`. Bad input returns a clear 400 error, never a crash. No external model API is used.

## Files
`predictions.csv` (scores for all 2,252 test claims) · `MEMO.md` (to Ritu) · `EXPECTED_SCORE.md` (written before submitting) · `DECISIONS.md` (assumptions, AI use) · `DISCARDED.md` (what I tried and dropped, with numbers) · `experiments.py` (re-runs every comparison) · `artifacts/eval*.json` (evidence) · `features.py train.py predict.py scoring.py app.py evaluate*.py`

## Model in one paragraph
Logistic regression on six features: small (<Rs 2,000) and uninspected claim; same plus partner onboarded <1 year ago; log claim amount; customer prior claims; and each outlet's fraud history (overall and since 1 May 2026). Fitted on May-Jun 2026 claims only, because the 1 May auto-approval change moved where fraud happens (see DECISIONS.md).

**Data handling (policy section 10):** data/ holds Kestrel operational data. Keep this repository private; do not publish it.
