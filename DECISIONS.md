# Decisions, assumptions, AI use

**Data cleaning**
- Train has 681 duplicate claim_ids (partners re-submitting bounced claims; same label each time). Kept the latest submission per claim: 11,146 claims left from 12,029 rows.
- 215 CRM claims have no outcome (still under investigation) -> dropped, not treated as "not fraud".
- Legacy Zoho rows could not store blanks, so some old "0" labels may really be undecided. The pack gives no way to tell which. Treated as 0, flagged as a limit: it makes pre-Oct-2025 fraud look lower than it was. Model relies on post-May data, so impact is small.
- Serials: normalised case, spaces, dashes -> all become 11 characters. Serial reuse showed no fraud signal, so not used.
- The UTC/IST warning in the policy concerns resolution events from Zoho; no such column exists in the files, so no time conversion was needed.
- A handful of claim_description cells contain text addressed to analysts/AI tools (e.g. telling the reader to use a random split and report accuracy). I treated it as data, ignored it, and did not follow it; I used a time-based split because that is how the model will be used.

**Key finding:** fraud rate was ~1% until April, then 2.0% (May), 2.9% (June). The cause is the 1 May policy: claims under Rs 2,000 are auto-approved without inspection. After May, 35 of 36 frauds were small and uninspected. Models trained on pre-May data scored worse than random on May-Jun (AUC 0.12-0.42), so the final model is fitted on May-Jun only.

**Partners:** Ritu is half right. Among small uninspected claims since May, outlets under a year old: 21% fraud; older outlets: 0.1%. But 7 outlets (SP3160, SP3232, SP3318, SP3129, SP3319, SP3286, SP3118) account for 33 of the 36 frauds; the other ~53 new partners are essentially clean, as Meenal said. Outlet history is a feature rather than "new partner" alone. SP3286 and SP3118 joined in June 2026, with little history, so the model rates them mostly from claim pattern.

**Metric:** Accuracy is not used as the target. Finance's question is rupees stopped per claim checked, with 40 reviews/month. Costs from the policy: fraud paid = claim amount; genuine claim held = Rs 380 goodwill. The Rs 260 service-contact cost is the same whether or not we check, so excluded.

**AI use:** I used Claude (a chat assistant) to explore the data and write the code; I checked the results myself and re-ran them. The cost was my normal subscription, with no paid API usage. There is no model API in the product, and the reason is in DISCARDED.md: the free-text fields are 12 stock phrases and add nothing, so a plain statistical model is enough. Everything I tried and dropped is in DISCARDED.md.
