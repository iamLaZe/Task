"""Fit final model on the post-1-May-2026 regime, write artifacts + predictions.csv"""
import pandas as pd,numpy as np,joblib,json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from features import *
FEATS=['small_uninsp','new_small_uninsp','log_amt','prior','p_rate','pr_post']
p,pr=load_ref(); tr=clean_train(pd.read_csv('data/train.csv'))
d=build(tr,p,pr); d['ts']=pd.to_datetime(d.submitted_at); d=d.sort_values('ts')
post=d[d.ts>='2026-05-01'].copy(); g=post.groupby('partner_id').is_fraud
post['pf']=g.cumsum()-post.is_fraud; post['pn']=g.cumcount(); post['pr_post']=(post.pf+0.3)/(post.pn+20)
m=make_pipeline(StandardScaler(),LogisticRegression(C=0.3,class_weight='balanced',max_iter=3000)).fit(post[FEATS],post.is_fraud)
# serve-time partner history (all labelled data)
hist_all={k:(float(v.is_fraud.sum()),int(len(v))) for k,v in d.groupby('partner_id')}
hist_post={k:(float(v.is_fraud.sum()),int(len(v))) for k,v in post.groupby('partner_id')}
joblib.dump(dict(model=m,feats=FEATS,hist_all=hist_all,hist_post=hist_post),'artifacts/model.joblib')
print('trained on',len(post),'rows,',int(post.is_fraud.sum()),'frauds; coefs',dict(zip(FEATS,m[-1].coef_[0].round(2))))
