import io,os
from flask import Flask,request,jsonify,render_template_string
import pandas as pd,numpy as np
from scoring import score_df,explain,band
app=Flask(__name__)
HERE=os.path.dirname(os.path.abspath(__file__))
REQ=['partner_id','sku','days_since_purchase','claim_amount_inr','photo_attached','partner_inspected','customer_prior_claims']
COLS=['claim_id','submitted_at','partner_id','sku','product_serial','days_since_purchase','claim_amount_inr','photo_attached','partner_inspected','claim_description','inspector_note','customer_prior_claims','source']
def prep(df):
    for c in COLS:
        if c not in df: df[c]=None
    df['claim_id']=df.claim_id.fillna('NEW').astype(str)
    df['submitted_at']=df.submitted_at.fillna(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'))
    df['product_serial']=df.product_serial.fillna('')
    return df
def run(df):
    d=score_df(prep(df)).copy(); d['rank_all']=d.score.rank(ascending=False,method='first').astype(int); return d
def rec(d):
    out=[]
    for _,r in d.iterrows():
        out.append(dict(claim_id=r.claim_id,submitted_at=r.submitted_at,partner_id=r.partner_id,city=r.city if isinstance(r.city,str) else '',
          sku=r.sku,family=r.family if isinstance(r.family,str) else '',amount=float(r.claim_amount_inr),inspected=r.partner_inspected,
          photo=r.photo_attached,prior=int(r.customer_prior_claims),days=int(r.days_since_purchase),score=round(float(r.score),4),
          recommendation=band(float(r.score)),reasons=explain(r)))
    return out
TEST=run(pd.read_csv(os.path.join(HERE,'data','test_unlabelled.csv')))
@app.get('/api/queue')
def queue():
    m=request.args.get('month',''); n=int(request.args.get('limit',40))
    d=TEST[TEST.submitted_at.str.startswith(m)] if m else TEST
    return jsonify(months=sorted(TEST.submitted_at.str[:7].unique()),total=len(d),rows=rec(d.nlargest(n,'score')))
@app.get('/api/claim/<cid>')
def claim(cid):
    d=TEST[TEST.claim_id==cid]
    return (jsonify(rec(d)[0]|{'raw':d.iloc[0][COLS].fillna('').to_dict()}) if len(d) else (jsonify(error='claim not found'),404))
@app.get('/api/options')
def options():
    return jsonify(partners=sorted(TEST.partner_id.unique().tolist()),skus=sorted(TEST.sku.unique().tolist()),claims=TEST.claim_id.tolist())
@app.post('/api/upload')
def upload():
    try:
        f=request.files['file']; df=pd.read_csv(io.BytesIO(f.read()))
        miss=[k for k in REQ if k not in df.columns]
        if miss: return jsonify(error=f"CSV is missing columns: {', '.join(miss)}"),400
        d=run(df); return jsonify(total=len(d),rows=rec(d.nlargest(int(request.form.get('limit',40)),'score')))
    except Exception as e: return jsonify(error=f"could not read file: {e}"),400
@app.post('/score')
def score():
    try:
        r=request.get_json(force=True); miss=[k for k in REQ if k not in r]
        if miss: return jsonify(error=f"missing fields: {miss}"),400
        d=run(pd.DataFrame([r])).iloc[0]
        if pd.isna(d.partner_age) or pd.isna(d.list_price_inr) or pd.isna(d.city): return jsonify(error="unknown partner_id or sku"),400
        x=rec(pd.DataFrame([d]))[0]
        return jsonify(claim_id=x['claim_id'],fraud_score=x['score'],recommendation=x['recommendation'],reasons=x['reasons'],
            note="Score ranks risk; the investigation desk can review ~40 claims/month, so work down the list from the highest score.")
    except Exception as e: return jsonify(error=f"could not score record: {e}"),400
PAGE=open(os.path.join(HERE,'page.html')).read()
@app.get('/')
def home(): return render_template_string(PAGE)
if __name__=='__main__': app.run(host='0.0.0.0',port=8000)
