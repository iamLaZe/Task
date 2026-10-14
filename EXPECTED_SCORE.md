# What I expect predictions.csv to score (written before submission)

You score on outcomes I don't have, and I don't know the metric, so these are for the quantities that matter.

- **Ranking quality (AUC): about 0.90, plausible range 0.80-0.95.** Same model fitted on May and scored on June (data it never saw) gave 0.955; I shade it down because that holdout had only 22 frauds, and the test period (Jul-Sep) is three months further on, so fraudsters/outlets may shift.
- **Precision in the top 40 per month: about 40% (range 25-55%)**, i.e. roughly 15 frauds per 40 claims checked. June holdout: 18 of 40 (bootstrap 90% range 12-24).
- **Accuracy: about 97%, and that tells you nothing.** Fraud in the recent period is about 3% of claims, so "approve everything" is already ~97% accurate. My model's accuracy at a sensible cut-off is 96-97% too, because it flags few claims. If accuracy at 0.5 is the metric, expect something lower (~90%) because the scores are tuned to rank, not to hit the 97% mark.
- Biggest risk to these numbers: fraud moving to outlets or claim types not seen in May-Jun, and 7 outlets dominating the top of my list.
