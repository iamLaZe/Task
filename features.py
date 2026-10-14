import pandas as pd, numpy as np
CAT=['partner_type']
NUM=['log_amt','amt_ratio','days_since_purchase','prior','small','uninsp','nophoto','partner_age','new_partner',
     'small_uninsp','new_small_uninsp','p_rate','p_n','post_may','near_exp','note_blank']
def load_ref(d='data'):
    return pd.read_csv(f'{d}/partners.csv'),pd.read_csv(f'{d}/products.csv')
def clean_train(tr):
    tr=tr.sort_values('submitted_at').drop_duplicates('claim_id',keep='last')  # resubmissions -> keep latest
    return tr[tr.is_fraud.notna()].copy()                                     # drop undecided (blank) cases
def build(df,p,pr,hist=None):
    """hist: dict partner_id->(frauds,n) known BEFORE this row (used at serve time); else computed expanding."""
    d=df.copy().merge(p,on='partner_id',how='left').merge(pr,on='sku',how='left')
    ts=pd.to_datetime(d.submitted_at)
    d['log_amt']=np.log1p(d.claim_amount_inr); d['amt_ratio']=d.claim_amount_inr/d.list_price_inr
    d['prior']=d.customer_prior_claims; d['small']=(d.claim_amount_inr<2000).astype(int)
    d['uninsp']=(d.partner_inspected=='N').astype(int); d['nophoto']=(d.photo_attached=='N').astype(int)
    d['partner_age']=(ts-pd.to_datetime(d.onboarded_date)).dt.days.fillna(1000)
    d['new_partner']=(d.partner_age<365).astype(int)
    d['small_uninsp']=d.small*d.uninsp; d['new_small_uninsp']=d.small_uninsp*d.new_partner
    d['post_may']=(ts>='2026-05-01').astype(int)
    d['near_exp']=(d.days_since_purchase/30.5/d.warranty_months)
    d['note_blank']=d.inspector_note.isna().astype(int)
    d['partner_type']=d.partner_type.fillna('unknown')
    if hist is None:   # leak-free expanding partner fraud history (excludes current row)
        d['_t']=ts; d=d.sort_values('_t')
        g=d.groupby('partner_id').is_fraud
        d['_f']=g.cumsum()-d.is_fraud; d['_n']=g.cumcount()
        d=d.drop(columns='_t')
    else:
        d['_f']=d.partner_id.map(lambda k:hist.get(k,(0,0))[0]); d['_n']=d.partner_id.map(lambda k:hist.get(k,(0,0))[1])
    d['p_rate']=(d._f+0.5)/(d._n+40); d['p_n']=d._n
    return d
def X(d): return pd.get_dummies(d[NUM+CAT],columns=CAT).astype(float)
