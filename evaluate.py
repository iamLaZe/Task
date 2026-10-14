import pandas as pd,numpy as np,json,warnings;warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score,average_precision_score
from features import *
p,pr=load_ref(); tr=clean_train(pd.read_csv('data/train.csv'))
d=build(tr,p,pr); d['ts']=pd.to_datetime(d.submitted_at); y=d.is_fraud.values
cols=X(d).columns
def models():
    return {'baseline_always_legit':None,
     'logreg':make_pipeline(StandardScaler(),LogisticRegression(C=0.3,class_weight='balanced',max_iter=2000)),
     'hgb':H(max_depth=3,learning_rate=0.05,max_iter=150,min_samples_leaf=30,l2_regularization=1.0,class_weight='balanced',random_state=0)}
def review(dv,s,k):
    """review top-k by score. rupees: fraud amount stopped minus 380 goodwill per genuine held"""
    o=np.argsort(-s)[:k]; v=dv.iloc[o]
    tp=v.is_fraud.sum(); fp=k-tp
    return dict(k=k,frauds_caught=int(tp),precision=round(tp/k,3),fraud_inr_stopped=float(v.loc[v.is_fraud==1,'claim_amount_inr'].sum()),
                goodwill_inr=380*int(fp),net_inr=float(v.loc[v.is_fraud==1,'claim_amount_inr'].sum()-380*fp),
                inr_stopped_per_claim_checked=round(float(v.loc[v.is_fraud==1,'claim_amount_inr'].sum())/k,1))
res={}
splits={'A_train<=Mar_test_Apr-Jun':('2026-04-01','2026-04-01','2026-07-01'),
        'B_train<=Apr_test_May-Jun':('2026-05-01','2026-05-01','2026-07-01'),
        'C_train<=May_test_Jun':('2026-06-01','2026-06-01','2026-07-01')}
for sn,(cut,a,b) in splits.items():
    tri=d.ts<cut; tei=(d.ts>=a)&(d.ts<b); dv=d[tei]; months=max(1,round((pd.Timestamp(b)-pd.Timestamp(a)).days/30.4))
    k=40*months; r={'n_test':int(tei.sum()),'frauds_test':int(dv.is_fraud.sum()),'always_legit_accuracy':round(1-dv.is_fraud.mean(),4)}
    for mn,m in models().items():
        if m is None: continue
        m.fit(X(d[tri])[cols],y[tri]); s=m.predict_proba(X(dv)[cols])[:,1]
        r[mn]=dict(auc=round(roc_auc_score(dv.is_fraud,s),3),pr_auc=round(average_precision_score(dv.is_fraud,s),3),
                   acc_at_half=round(float(((s>.5)==dv.is_fraud).mean()),3),review=review(dv,s,k))
    res[sn]=r; print(sn,json.dumps(r,indent=1))
json.dump(res,open('artifacts/eval.json','w'),indent=1)
