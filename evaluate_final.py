"""Honest check of the FINAL model config: fit on May 2026, score June 2026 (unseen, later in time). Also bootstrap."""
import pandas as pd,numpy as np,json,warnings;warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score,average_precision_score
from features import *
FEATS=['small_uninsp','new_small_uninsp','log_amt','prior','p_rate','pr_post']
p,pr=load_ref(); d=build(clean_train(pd.read_csv('data/train.csv')),p,pr); d['ts']=pd.to_datetime(d.submitted_at); d=d.sort_values('ts')
post=d[d.ts>='2026-05-01'].copy(); g=post.groupby('partner_id').is_fraud
post['pf']=g.cumsum()-post.is_fraud; post['pn']=g.cumcount(); post['pr_post']=(post.pf+.3)/(post.pn+20)
a=post[post.ts<'2026-06-01']; b=post[post.ts>='2026-06-01'].copy()
# for June rows, partner history should only use labels known at the time (expanding, from May + earlier June) - already so
m=make_pipeline(StandardScaler(),LogisticRegression(C=.3,class_weight='balanced',max_iter=3000)).fit(a[FEATS],a.is_fraud)
b['s']=m.predict_proba(b[FEATS])[:,1]; y=b.is_fraud.values
def at(k,idx):
    bb=b.iloc[idx]; o=np.argsort(-bb.s.values)[:k]; v=bb.iloc[o]; f=v.is_fraud==1
    return f.sum(), v.loc[f,'claim_amount_inr'].sum(), (k-f.sum())*380
c,amt,gw=at(40,np.arange(len(b)))
out=dict(holdout='fit May-2026, score Jun-2026',n=len(b),frauds=int(y.sum()),always_legit_accuracy=round(1-y.mean(),4),
 auc=round(roc_auc_score(y,b.s),3),pr_auc=round(average_precision_score(y,b.s),3),
 top40=dict(frauds_caught=int(c),precision=round(c/40,3),fraud_inr_stopped=float(amt),goodwill_inr_lost=int(gw),inr_stopped_per_claim_checked=round(amt/40,1),net_inr=float(amt-gw)),
 acc_at_0_5=round(float(((b.s>.5)==y).mean()),3),acc_at_top40_flag=round(float(1-((np.argsort(np.argsort(-b.s.values))<40)!=y).mean()),3))
rng=np.random.default_rng(0); cs=[];ia=[]
for _ in range(500):
    idx=rng.integers(0,len(b),len(b)); k=40; c_,a_,_=at(k,idx); cs.append(c_); ia.append(a_/k)
out['bootstrap_top40_caught_90ci']=[int(np.percentile(cs,5)),int(np.percentile(cs,95))]
out['bootstrap_inr_per_claim_90ci']=[round(float(np.percentile(ia,5))),round(float(np.percentile(ia,95)))]
out['earlier_check_model_trained_only_before_May']='AUC 0.35-0.42 on Apr-Jun/May-Jun: pre-regime patterns do not transfer (see artifacts/eval.json)'
json.dump(out,open('artifacts/eval_final.json','w'),indent=1); print(json.dumps(out,indent=1))
