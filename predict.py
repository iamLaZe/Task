import pandas as pd
from scoring import *
te=pd.read_csv('data/test_unlabelled.csv'); d=score_df(te)
sub=pd.read_csv('data/sample_submission.csv')[['claim_id']].merge(d[['claim_id','score']].drop_duplicates('claim_id'),on='claim_id',how='left')
assert sub.score.notna().all() and len(sub)==2252
sub['score']=sub.score.round(6); sub.to_csv('predictions.csv',index=False)
d['month']=d.submitted_at.str[:7]
print(sub.score.describe()); 
print('top-120 by score: small_uninsp share',d.sort_values('score',ascending=False).head(120).small_uninsp.mean(), 'claims>=0.5:',(d.score>=.5).sum(),' monthly top40 INR claimed:')
for m,g in d.groupby('month'): print(m,g.nlargest(40,'score').claim_amount_inr.sum())
