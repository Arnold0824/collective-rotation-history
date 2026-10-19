"""Recompute paired fish scores and forecast alignment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .paths import data_root, fish_input, fish_output
ROOT = data_root()
TASK = fish_input()
OUT = fish_output()
OUT.mkdir(parents=True,exist_ok=True)


def check(label, condition, detail=None):
    checks.append(dict(check=label, passed=bool(condition), detail=detail))


def near(label, x, y, tolerance=1e-10):
    x, y = np.asarray(x), np.asarray(y)
    error = float(np.max(np.abs(x-y))) if x.shape == y.shape and x.size else (0. if x.shape == y.shape else np.inf)
    check(label, np.isfinite(error) and error < tolerance, error)


def output_file(name):
    candidates = [TASK/name] if (TASK/name).is_file() else []
    if len(candidates) != 1:
        raise ValueError(f'Expected one {name}, found {candidates}')
    return candidates[0]


checks, metrics = [], []
pred = pd.read_parquet(output_file('predictions.parquet'))
keys = ['variant','definition','trial_id','horizon_frames']
grouped = pred.groupby(keys, sort=False)
sources = {}
for path in (OUT/'reconstructed').glob('*.parquet'):
    parts = path.stem.split('_')
    sources[parts[0]+parts[1]] = pd.read_parquet(path).set_index('frame')
for group, records in grouped:
    prefix = '/'.join(map(str,group))
    supports = []
    for model, rows in records.groupby('model',sort=False):
        rows = rows.sort_values('frame')
        check(prefix+'/'+model+'/unique_frames', not rows.frame.duplicated().any())
        check(prefix+'/'+model+'/finite_values', np.isfinite(rows[['target','prediction']].to_numpy()).all())
        supports.append((model,rows.frame.to_numpy(),rows.target.to_numpy()))
        target = rows.target.to_numpy(); estimate = rows.prediction.to_numpy()
        mse = float(np.mean((estimate-target)**2)); variance = float(np.var(target))
        metrics.append(dict(zip(keys,group),model=model,rows=len(rows),mse=mse,
                            rmse=float(np.sqrt(mse)),r2=1-mse/variance if variance else None))
        if group[0] == 'primary':
            source = sources[group[2]]
            field = {'heading':'heading_common','displacement':'displacement_common',
                     'paired_heading':'heading_common','paired_displacement':'displacement_common',
                     'body_heading':'heading_common','backward_displacement':'displacement_common'}[group[1]]
            frames = rows.frame.to_numpy(dtype=int); horizon=int(group[3])
            near(prefix+'/'+model+'/target_from_positions', target,source.loc[frames+horizon,field].to_numpy())
            if model=='persistence':
                near(prefix+'/persistence_from_positions',estimate,source.loc[frames,field].to_numpy())
            valid=np.isfinite(source[field].to_numpy())
            invalid_prefix=np.r_[0,np.cumsum(~valid)]
            check(prefix+'/'+model+'/whole_interval_valid',np.all(invalid_prefix[frames+horizon+1]-invalid_prefix[frames-30]==0))
    for model,frames,target in supports[1:]:
        near(prefix+'/'+model+'/same_frames',frames,supports[0][1],.5)
        near(prefix+'/'+model+'/same_target',target,supports[0][2])

metric=pd.DataFrame(metrics)
metric.to_csv(OUT/'independent_recording_metrics.csv',index=False)
for key,rows in metric.groupby(['variant','trial_id','horizon_frames','model']):
    check('/'.join(map(str,key))+'/paired_definition_counts',rows.rows.nunique()==1)

discrete=pd.read_parquet(output_file('discrete_predictions.parquet'))
discrete_metrics=[]
for group,rows in discrete.groupby(['variant','definition','trial_id','order'],sort=False):
    probs=rows[['p0','p1','p2']].to_numpy(); target=rows.target_state.to_numpy(dtype=int)
    near('/'.join(map(str,group))+'/probability_sum',probs.sum(axis=1),np.ones(len(rows)))
    check('/'.join(map(str,group))+'/positive_probabilities',np.all(probs>0))
    truth=np.eye(3)[target]
    nll=float(-np.log(probs[np.arange(len(rows)),target]).mean())
    brier=float(np.mean(np.sum((probs-truth)**2,axis=1)))
    rps=float(np.mean(np.sum((np.cumsum(probs,axis=1)[:,:-1]-np.cumsum(truth,axis=1)[:,:-1])**2,axis=1)))
    discrete_metrics.append(dict(zip(['variant','definition','trial_id','order'],group),rows=len(rows),nll=nll,brier=brier,rps=rps))
pd.DataFrame(discrete_metrics).to_csv(OUT/'independent_discrete_metrics.csv',index=False)
(OUT/'saved_prediction_checks.json').write_text(json.dumps({'passed':all(c['passed'] for c in checks),'checks':checks},indent=2))
print(json.dumps({'checks':len(checks),'passed':sum(c['passed'] for c in checks),'continuous_metric_rows':len(metric),'discrete_metric_rows':len(discrete_metrics)}))
assert all(c['passed'] for c in checks)
