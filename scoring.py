import pandas as pd,numpy as np,joblib,os
from features import *
P,PR=load_ref(os.path.join(os.path.dirname(__file__),'data')) if False else load_ref(os.path.join(os.path.dirname(os.path.abspath(__file__)),'data'))
A=joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'artifacts','model.joblib'))
M,FEATS=A['model'],A['feats']
def _hist(k,h): return h.get(k,(0,0))
def score_df(df):
    d=build(df,P,PR,hist=A['hist_all'])
    d['pr_post']=[(_hist(k,A['hist_post'])[0]+0.3)/(_hist(k,A['hist_post'])[1]+20) for k in d.partner_id]
    d['score']=M.predict_proba(d[FEATS])[:,1]; return d
def explain(r):
    why=[]
    if r.small_uninsp: why.append("Claim is under Rs 2,000 and was not inspected - since 1 May these are auto-approved, which is where almost all recent fraud sits.")
    if r.new_small_uninsp: why.append(f"The partner outlet is under a year old ({int(r.partner_age)} days) and this is a small, uninspected claim - the highest-risk combination (about 1 in 4 such claims proved fraudulent in May-Jun).")
    if r.prior>=2: why.append(f"Customer has {int(r.prior)} earlier warranty claims (fraud rate rises sharply from 2 prior claims).")
    ph=_hist(r.partner_id,A['hist_post'])
    if ph[0]>=3: why.append(f"Outlet {r.partner_id} has {int(ph[0])} confirmed fraud cases out of {ph[1]} claims since 1 May.")
    if not why: why.append("No strong risk signal: inspected or larger claim, established outlet, few prior claims.")
    return why
def band(s): return 'HIGH - hold for investigation' if s>=.8 else 'MEDIUM - check if desk capacity allows' if s>=.5 else 'LOW - pay normally'
