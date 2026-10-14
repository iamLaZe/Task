# What I tried and dropped

All numbers are from `experiments.py` (saved in `artifacts/experiments.txt`) and `evaluate.py` (`artifacts/eval.json`). Test setup: fit on May 2026 claims, score June 2026 claims the model had never seen (720 claims, 22 frauds). Because that is a small sample, differences of a point or two are noise.

| Tried | Result | Decision |
|---|---|---|
| Train on all 15 months, test on later months (logistic regression and gradient boosting) | AUC 0.12-0.42, worse than guessing, and almost no fraud caught | Dropped. The 1 May auto-approve rule changed where fraud happens, so old patterns mislead. |
| Random train/test split | AUC 0.83 (looks fine) | Dropped. It hides the problem above, because it mixes months. I test only on later months. |
| Gradient boosting on May data | AUC 0.93, 15 of top 40 | Dropped. Only 14 frauds in May, too few for it to beat a simple model. |
| Logistic regression, all 13 features | AUC 0.94, 16 of top 40 | Dropped for the 6-feature version, which does slightly better. |
| Rule: "new outlet and small and uninspected" | AUC 0.92, 11 of top 40 | Dropped. Good start, but it can't tell outlets apart. |
| Rule: "small and uninspected" only | AUC 0.65, 2 of top 40 | Dropped. Too broad, since most small claims are genuine. |
| "New partner" alone (the hunch) | AUC 0.90, 4 of top 40 | Dropped. Most new partners are fine; the problem is a few outlets. |
| Add claim description text | AUC 0.96, no more frauds caught (18 of 40) | Dropped. See below. |
| Serial number reused by another claim | Fraud 1.3% vs 1.2% | Dropped. No signal. |
| **Kept:** logistic regression, 6 features, fitted on May-June | AUC 0.955, 18 of top 40, about Rs 620 per claim checked | **This is the model in the app.** |

**Why no language model call is in the product.** The claim description has only 12 stock phrases ("remote not working", "motor not running" and so on), and adding it did not catch any more frauds. The inspector note is blank for 81% of claims. The signal is in the numbers (amount, inspection, outlet, history), so a simple model is enough and a model API would add cost and a failure point with no gain.
