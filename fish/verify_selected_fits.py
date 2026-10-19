"""Check recorded metrics, selection isolation and selected fits independently."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from .paths import data_root, fish_input, fish_output
BASE=data_root()
TASK=fish_input()
OUT=fish_output()
OUT.mkdir(parents=True,exist_ok=True)
checks=[]
def check(name,condition,detail=None):
    checks.append(dict(name=name,passed=bool(condition),detail=detail))
def near(name,a,b,tol=1e-8):
    a,b=np.asarray(a),np.asarray(b)
    error=float(np.max(np.abs(a-b))) if a.shape==b.shape and a.size else np.inf
    check(name,np.isfinite(error) and error<tol,error)

keys=['variant','definition','trial_id','horizon_frames','model']
actual=pd.read_csv(TASK/'recording_metrics.csv').set_index(keys).sort_index()
own=pd.read_csv(OUT/'independent_recording_metrics.csv').set_index(keys).sort_index()
check('same_continuous_metric_keys',actual.index.equals(own.index))
for c in ['mse','rmse','r2']:near('reported_'+c,actual[c],own[c],1e-10)
keys_d=['variant','definition','trial_id','order']
actual_d=pd.read_csv(TASK/'discrete_metrics.csv').set_index(keys_d).sort_index()
own_d=pd.read_csv(OUT/'independent_discrete_metrics.csv').set_index(keys_d).sort_index()
check('same_discrete_metric_keys',actual_d.index.equals(own_d.index))
for c in ['nll','brier','rps']:near('reported_'+c,actual_d[c],own_d[c],1e-10)

selection=pd.read_csv(TASK/'training_selection.csv')
inventory=pd.read_csv(TASK/'model_inventory.csv')
for group,rows in selection.groupby(keys):
    prefix='/'.join(map(str,group))
    held=group[2]
    for col in ['inner_validation_1','inner_validation_2']:
        check(prefix+'/'+col+'_excludes_evaluation',not rows[col].eq(held).any())
    check(prefix+'/distinct_inner_recordings',not (rows.inner_validation_1==rows.inner_validation_2).any())
    best=rows.sort_values(['validation_score','complexity','ridge'],ascending=[True,True,False]).iloc[0]
    chosen=rows.loc[rows.selected]
    check(prefix+'/one_selected',len(chosen)==1)
    check(prefix+'/minimum_development_loss',len(chosen)==1 and chosen.iloc[0].complexity==best.complexity and chosen.iloc[0].ridge==best.ridge)

cache={}
for path in (OUT/'reconstructed').glob('*.parquet'):
    tid=path.stem.replace('_','');source=pd.read_parquet(path)
    for definition,field in [('heading','heading_common'),('displacement','displacement_common')]:
        v=source[field].to_numpy();valid=np.isfinite(v)
        bad=np.r_[0,np.cumsum(~valid)]
        transform=np.linalg.pinv(np.c_[np.ones(6),np.arange(-5,1)/30])
        for h in [3,10,30]:
            ix=np.arange(30,len(v)-h)
            ix=ix[(bad[ix+h+1]-bad[ix-30])==0]
            a,b=(transform@v[ix[:,None]+np.arange(-5,1)].T)
            cache[tid,definition,h]=pd.DataFrame(dict(frame=ix,current=v[ix],delayed=v[ix-5],state=a,trend=b,target=v[ix+h]))

def xy(rows,family):
    if family=='state_trend':
        return rows[['state','trend']].to_numpy(),rows.target.to_numpy()-rows.state.to_numpy()
    cols=['current','delayed'] if family=='history' else ['current']
    return rows[cols].to_numpy(),rows.target.to_numpy()-rows.current.to_numpy()

refit_count=0
for r in inventory.itertuples(index=False):
    tids=r.training_trials.split('|')
    check(f'{r.variant}/{r.definition}/{r.trial_id}/{r.horizon_frames}/{r.model}/train_ids',len(tids)==2 and len(set(tids))==2 and r.trial_id not in tids)
    if r.variant!='primary' or (r.trial_id,r.horizon_frames) not in [('N100066',3),('N700124',30)]:continue
    model=json.loads((TASK/r.model_file).read_text())['parameters']
    rows=pd.concat([cache[t,r.definition,r.horizon_frames] for t in tids],ignore_index=True)
    x,y=xy(rows,r.model)
    prefix=f'{r.definition}/{r.trial_id}/{r.horizon_frames}/{r.model}'
    check(prefix+'/full_training_rows',len(rows)==r.training_rows)
    if r.model=='current_spline':
        knots=np.asarray(model['knots'])
        d=(BSpline.design_matrix(x[:,0],knots,3).toarray()-BSpline.design_matrix(-x[:,0],knots,3).toarray())/2
    else:
        scale=np.sqrt((x*x).mean(axis=0));scale=np.where(scale>=1e-8,scale,1)
        near(prefix+'/training_state_scale',scale,model['state_scale'])
        exponents=np.asarray(model['exponents'])
        d=np.prod((x/scale)[:,None,:]**exponents[None,:,:],axis=2)
    ds=np.sqrt((d*d).mean(axis=0));ds=np.where(ds>=1e-8,ds,1)
    near(prefix+'/training_design_scale',ds,model['design_scale'])
    d=d/ds
    coeff=np.linalg.solve(d.T@d+r.ridge*np.eye(d.shape[1]),d.T@y)
    near(prefix+'/coefficients',coeff,model['coefficients'],1e-7)
    refit_count+=1

result={'checks':len(checks),'passed':sum(c['passed'] for c in checks),'independent_refits':refit_count,'details':checks}
(OUT/'refit_and_score_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='details'}))
assert all(c['passed'] for c in checks),[c for c in checks if not c['passed']]
