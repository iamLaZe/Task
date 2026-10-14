# Submission Form: Kestrel Warranty Fraud Review Tool

## 1. What did you build, and what business outcome does it move? State the number and the money.

I built a tool that helps Kestrel's investigation team decide which warranty claims to check first. Every claim gets a risk score, and the team starts from the top of the list. They can only check about 40 claims a month, so the point is to make sure those 40 are the ones most likely to be fraud.

I used June claims as the testing dataset. I picked the top 40 most likely frauds, and 18 of them were real frauds, worth Rs 24,700. That amount was saved, since it would otherwise have been paid out. The other 22 were honest customers who would have had a delay, and that goodwill costs Kestrel about Rs 8,400. So the tool saves roughly Rs 16,000 in a month, or about Rs 600 for every claim the team checks.

## 2. What does one prediction cost, and what would a month cost at Kestrel's volume (about 750 warranty claims a month)? Show the arithmetic. If you used no paid calls, say so.

I did not use any paid service. The model is small enough to run on an ordinary laptop, just like using an extension, so scoring one claim costs no money.

- One prediction: Rs 0
- One month: 750 claims × Rs 0 = Rs 0

The only real cost in this project is the goodwill when an honest customer is held by mistake, which Kestrel puts at about Rs 380 each time.

## 3. How do you know it works? Sample size, how you checked, error rate, and the kind of case it gets wrong.

I trained the model on May's claims and then tested it on June's. On June, 18 of the tool's top 40 were fraud, so a bit more than half of the claims it flagged were honest customers, and it missed 4 of the 22 frauds that month. June is only one month with 22 frauds, so these numbers could easily move up or down.

I went through the mistakes by hand. The false alarms were all small claims with no inspection, from newer outlets, and some of those outlets are actually fine. Two of the missed frauds were from brand-new outlets and only just missed the top 40. The other two came from old, established outlets, one of them a large inspected claim, and the tool did not see either as risky at all.

## 4. Did you change, narrow, or push back on the client's ask? What, when, and why. *(can only raise your score)*

Yes.

- **The 97% accuracy target.** Only about 3 in 100 claims are fraud, so a tool that approves everything would already score 97% and catch no fraud at all. I suggested measuring what the business really cares about: how much fraud we stop for each claim the team checks.
- **"New partners are the problem."** I did not go along with this. A handful of outlets cause almost all of the fraud, and most new partners are fine, as the service desk manager said. So the tool looks at each outlet's own history and not just how new it is.
- **Training window.** I trained it only on May and June. Fraud jumped after 1 May, when claims under Rs 2,000 started to be approved without inspection, and the older months gave misleading patterns.

## 5. What is wrong with what you are handing us? Be specific: bugs, shortcuts, things you know are off. *(can only raise your score)*

_[Not provided: add your answer here]_

## 6. What did you deliberately leave out, and why that rather than something else?

- **Language-model and text features:** the claim description is 12 stock phrases and added no extra frauds.
- **Photo analysis:** no photos were provided.
- **Automated retraining and a login:** a working ranked list the desk can use mattered more in the time I had.
- **Time-zone conversion:** the UTC issue concerns resolution events that aren't in the files.

## 7. Anything you built or found that nobody asked for?

I found that fraud jumped from about 1% to about 3% after the 1 May auto-approve rule, and that it sits in a handful of outlets. I built a ranked review queue and a CSV upload, and I checked that serial-number reuse carries no signal (1.3% vs 1.2%).

## 8. What did you use AI for? Which tools and models, where they helped, where they wasted your time, what you threw away. Link your three-minute screen recording here.

I used Claude (the claude.ai chat) to explore the data, write the code and draft documents. It helped me move fast, including testing many model variants quickly. It wasted time on my first version of the app, which gave no reply because of a form bug, and I rebuilt it as a ranked queue. I threw away models trained on all history, random-split testing, gradient boosting, the "new partner" rule on its own, serial duplicates and text features; `DISCARDED.md` has the numbers.

Screen recording / materials: https://drive.google.com/drive/folders/1_wnVdcxJ5hJkLQl03wOwOwTcEaABMKE_?usp=drive_link

## 9. Your Public Google Drive Link

https://drive.google.com/drive/folders/1_wnVdcxJ5hJkLQl03wOwOwTcEaABMKE_?usp=drive_link

## 10. Someone picks this up on Monday and you are unreachable. The three things they need to know.

_[Not provided: add your answer here]_

## 11. Honest hours spent. One number.

4

## GitHub Repo Link (Public)

https://github.com/iamLaZe/Task
