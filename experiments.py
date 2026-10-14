"""Everything I compared, in one re-runnable script. Output is saved to artifacts/experiments.txt.
Holdout for the post-May comparisons: fit on May 2026, score June 2026."""
import pandas as pd,numpy as np,warnings,re;warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack,csr_matrix
from features import *
p,pr=load_ref(); raw=pd.read_csv('data/train.csv'); tr=clean_train(raw)
d=build(tr,p,pr); d['ts']=pd.to_datetime(d.submitted_at); d=d.sort_values('ts')
post=d[d.ts>='2026-05-01'].copy(); g=post.groupby('partner_id').is_fraud
post['pf']=g.cumsum()-post.is_fraud; post['pn']=g.cumcount(); post['pr_post']=(post.pf+.3)/(post.pn+20)
a=post[post.ts<'2026-06-01']; b=post[post.ts>='2026-06-01']
def top40(y,s,amt):
    o=np.argsort(-s)[:40]; f=y[o]==1; return int(f.sum()), round(float(amt[o][f].sum())/40)
def rep(name,s):
    y=b.is_fraud.values; c,i=top40(y,s,b.claim_amount_inr.values)
    print(f'{name:55s} AUC {roc_auc_score(y,s):.3f}  PR-AUC {average_precision_score(y,s):.3f}  caught {c}/40  Rs/claim {i}')
LR=lambda C=.3:make_pipeline(StandardScaler(),LogisticRegression(C=C,class_weight='balanced',max_iter=3000))
print('Holdout: fit May 2026 (%d rows, %d frauds), score June 2026 (%d rows, %d frauds)\n'%(len(a),a.is_fraud.sum(),len(b),b.is_fraud.sum()))
ALL=['log_amt','amt_ratio','days_since_purchase','prior','small','uninsp','nophoto','partner_age','new_partner','small_uninsp','new_small_uninsp','p_rate','pr_post']
FINAL=['small_uninsp','new_small_uninsp','log_amt','prior','p_rate','pr_post']
for nm,f in [('logreg, all 13 features',ALL),('logreg, FINAL 6 features (kept)',FINAL),('logreg, tiny 4 features',['new_small_uninsp','log_amt','prior','pr_post']),('logreg, FINAL minus prior-claims & outlet history',[x for x in FINAL if 'p' not in x[:2] and x!='pr_post'])]:
    rep(nm,LR().fit(a[f],a.is_fraud).predict_proba(b[f])[:,1])
rep('gradient boosting, all features (dropped)',H(max_depth=2,learning_rate=.05,max_iter=100,min_samples_leaf=20,l2_regularization=2,random_state=0).fit(a[ALL],a.is_fraud).predict_proba(b[ALL])[:,1])
rep('rule: new outlet & small & uninspected (dropped)',(b.new_small_uninsp+0.01*b.log_amt).values)
rep('rule: small & uninspected only (dropped)',(b.small_uninsp+0.01*b.log_amt).values)
rep('"new partner" alone (the hunch, dropped)',(b.new_partner+0.001*b.log_amt).values)
# free text
print('\n-- free-text check: does claim_description / inspector_note add anything? --')
print('distinct claim_description values:',post.claim_description.nunique(),' | inspector_note blank share:',round(post.inspector_note.isna().mean(),2))
print(post.groupby('claim_description').is_fraud.agg(['mean','count']).sort_values('count',ascending=False).head(8).round(3).to_string())
tv=TfidfVectorizer(min_df=3,ngram_range=(1,2)); Ta=tv.fit_transform(a.claim_description.fillna('')); Tb=tv.transform(b.claim_description.fillna(''))
sc=StandardScaler().fit(a[FINAL]); Xa=hstack([csr_matrix(sc.transform(a[FINAL])),Ta]).tocsr(); Xb=hstack([csr_matrix(sc.transform(b[FINAL])),Tb]).tocsr()
rep('FINAL + description text (dropped)',LogisticRegression(C=.3,class_weight='balanced',max_iter=3000).fit(Xa,a.is_fraud).predict_proba(Xb)[:,1])
# serials
al=pd.concat([tr,pd.read_csv('data/test_unlabelled.csv')]); ser=al.product_serial.astype(str).str.upper().str.replace(r'[^A-Z0-9]','',regex=True)
dup=ser.duplicated(keep=False).values[:len(tr)]; print('\nserial reused by another claim: fraud rate',round(tr.is_fraud[dup].mean(),4),'vs not reused',round(tr.is_fraud[~dup].mean(),4),'(no signal, dropped)')
# random split vs time split
X=build(tr,p,pr); ALL2=[c for c in ALL if c!='pr_post']
Xa,Xb2=train_test_split(X,test_size=.3,random_state=0,stratify=X.is_fraud)
m=H(max_depth=3,max_iter=150,random_state=0).fit(Xa[ALL2],Xa.is_fraud); s=m.predict_proba(Xb2[ALL2])[:,1]
print('\nrandom split (dropped, flatters the model): AUC',round(roc_auc_score(Xb2.is_fraud,s),3),'- vs 0.12-0.42 when tested on later months, see artifacts/eval.json')
