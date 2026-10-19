"""Paired fish observables and regression calculations."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from methods.models import ParityRidgePredictor, _parity_exponents, OddSplinePredictor, ContinuousRows, SparseBackoffMarkov, build_order_run_ids
from .paths import data_root, fish_input, fish_output
ROOT = data_root()
INPUT = fish_input()
OUT = fish_output()

def read_protocol():
    return json.loads((ROOT/"analyses/fish_directions/protocol.json").read_text(encoding="utf-8"))

def sha(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):
            value.update(chunk)
    return value.hexdigest()

def jsonable(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,Path): return str(value)
    raise TypeError(type(value))

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,default=jsonable,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def states(values):return np.where(values<=-.3,0,np.where(values>=.3,2,1)).astype(np.int16)

def rotation(points,directions,support,minimum_fraction=.8,epsilon=1e-12):
    count=support.sum(axis=1)
    center=np.divide(np.where(support[...,None],points,0).sum(axis=1),count[:,None],
                     out=np.full((len(points),2),np.nan),where=count[:,None]>0)
    radial=points-center[:,None,:];rn=np.linalg.norm(radial,axis=-1)
    dn=np.linalg.norm(directions,axis=-1)
    good=support&(rn>epsilon)&(dn>epsilon)&np.isfinite(rn)&np.isfinite(dn)
    num=radial[...,0]*directions[...,1]-radial[...,1]*directions[...,0]
    term=np.divide(num,rn*dn,out=np.zeros_like(rn),where=good)
    used=good.sum(axis=1)
    m=np.divide(term.sum(axis=1),used,out=np.full(len(points),np.nan),where=used>0)
    valid=(used>=np.ceil(minimum_fraction*points.shape[1]))&np.isfinite(m)
    m=np.clip(m,-1,1);m[~valid]=np.nan
    return m,valid,used,center,good

def paired_observables(points,headings,frames,threshold,*,max_jump=50,jump_fraction=.5,minimum_fraction=.8):
    delta=np.full_like(points,np.nan)
    adjacent=np.r_[False,np.diff(frames)==1]
    delta[1:]=points[1:]-points[:-1]
    delta[~adjacent]=np.nan
    step=np.linalg.norm(delta,axis=-1)
    boundaries=(np.mean(step>max_jump,axis=1)>jump_fraction)&adjacent
    base=np.isfinite(points).all(axis=2)&np.isfinite(headings)&np.isfinite(delta).all(axis=2)&(step>threshold)
    base[boundaries]=False
    heading=np.stack((np.cos(headings),np.sin(headings)),axis=-1)
    mh,vh,nh,center,support_h=rotation(points,heading,base,minimum_fraction)
    md,vd,nd,_,support_d=rotation(points,delta,base,minimum_fraction)
    assert np.array_equal(support_h,support_d)
    valid=vh&vd;mh[~valid]=np.nan;md[~valid]=np.nan
    ids=build_order_run_ids(frames,valid,minimum_run_length=2)
    valid=ids>=0;mh[~valid]=np.nan;md[~valid]=np.nan
    unit=np.divide(delta,step[...,None],out=np.full_like(delta,np.nan),where=(step[...,None]>0))
    cosine=np.sum(heading*unit,axis=-1)
    angles=np.degrees(np.arccos(np.clip(cosine,-1,1)))
    def avg(v):return np.divide(np.where(support_h,v,0).sum(axis=1),nh,out=np.full(len(frames),np.nan),where=nh>0)
    return dict(heading=mh,displacement=md,valid=valid,run_id=ids,count=nh,boundary=boundaries,
                step=step,base_count=base.sum(axis=1),angle_mean=avg(angles),angle_gt90_fraction=avg((angles>90).astype(float)),
                center=center,individual_support=support_h)

def make_rows(frame_table,definition,horizon,reserve=30,delay=5):
    result=[]
    for rid,g in frame_table.loc[frame_table.valid].groupby('run_id',sort=True):
        frames=g.frame.to_numpy(np.int64);v=g[definition].to_numpy(float)
        assert np.all(np.diff(frames)==1)
        ix=np.arange(reserve,len(g)-horizon)
        if not len(ix):continue
        w=v[ix[:,None]+np.arange(-5,1)[None,:]]
        tau=np.arange(-5,1)/float(g.fps.iloc[0]);tc=tau-tau.mean()
        b=(w@tc)/(tc@tc);a=w.mean(axis=1)-b*tau.mean()
        result.append(pd.DataFrame(dict(frame=frames[ix],run_id=int(rid),current=v[ix],delayed=v[ix-delay],
                    target=v[ix+horizon],target_frame=frames[ix+horizon],state=a,trend=b)))
    cols=['frame','run_id','current','delayed','target','target_frame','state','trend']
    return pd.concat(result,ignore_index=True) if result else pd.DataFrame(columns=cols)

def discrete_rows(frame_table,definition,step=5,reserve=25,order=6):
    contexts=[];targets=[];frames=[];runs=[]
    for rid,g in frame_table.loc[frame_table.valid].groupby('run_id',sort=True):
        f=g.frame.to_numpy(np.int64);s=states(g[definition].to_numpy(float))
        assert np.all(np.diff(f)==1)
        ix=np.arange(reserve,len(g)-step)
        if len(ix):
            contexts.append(s[ix[:,None]-np.arange(order)[None,:]*step]);targets.append(s[ix+step]);frames.append(f[ix]);runs.append(np.full(len(ix),rid))
    if not contexts:return dict(context=np.empty((0,order),np.int16),target=np.empty(0,np.int16),frame=np.empty(0,np.int64),run_id=np.empty(0,np.int64))
    return dict(context=np.concatenate(contexts),target=np.concatenate(targets),frame=np.concatenate(frames),run_id=np.concatenate(runs))

def xy(rows,family):
    if family=='state_trend':return rows[['state','trend']].to_numpy(),rows.target.to_numpy()-rows.state.to_numpy(),rows.state.to_numpy()
    x=rows[['current','delayed']].to_numpy() if family=='history' else rows[['current']].to_numpy()
    return x,rows.target.to_numpy()-rows.current.to_numpy(),rows.current.to_numpy()

def prepared(train,family,complexity):
    x,y,_=xy(train,family)
    if family=='current_spline':
        knots=np.r_[np.repeat(-1.,3),np.linspace(-1,1,complexity),np.repeat(1.,3)]
        design=OddSplinePredictor.design(x[:,0],knots)
        ds=np.sqrt(np.mean(design**2,axis=0));ds=np.where(ds>1e-8,ds,1)
        info=dict(knots=knots,design_scale=ds)
    else:
        scale=np.sqrt(np.mean(x**2,axis=0));scale=np.where(scale>=1e-8,scale,1.)
        exponents=_parity_exponents(np.full(x.shape[1],-1,dtype=np.int8),complexity)
        design=np.prod((x/scale)[:,None,:]**exponents[None,:,:],axis=-1)
        ds=np.sqrt(np.mean(design**2,axis=0));ds=np.where(ds>=1e-8,ds,1.)
        info=dict(state_scale=scale,exponents=exponents,design_scale=ds)
    normalized=design/ds
    return normalized.T@normalized,normalized.T@y,info

def design_for(rows,family,info):
    x,_,base=xy(rows,family)
    if family=='current_spline':xdesign=OddSplinePredictor.design(x[:,0],info['knots'])
    else:xdesign=np.prod((x/info['state_scale'])[:,None,:]**info['exponents'][None,:,:],axis=-1)
    return xdesign/info['design_scale'],base

def choose_and_fit(training,family,complexities,ridges):
    candidates={};trials=list(training)
    for complexity in complexities:
        for vtid in trials:
            train=pd.concat([r for tid,r in training.items() if tid!=vtid],ignore_index=True);val=training[vtid]
            gram,rhs,info=prepared(train,family,complexity);d,base=design_for(val,family,info)
            denominator=max(float(np.mean((val.target-val.current)**2)),1e-12)
            for ridge in ridges:
                coefficients=np.linalg.solve(gram+ridge*np.eye(len(rhs)),rhs)
                loss=float(np.mean((val.target.to_numpy()-(base+d@coefficients))**2))/denominator
                candidates.setdefault((complexity,ridge),[]).append((vtid,loss))
    records=[]
    for (complexity,ridge),folds in candidates.items():
        records.append(dict(complexity=complexity,ridge=ridge,validation_score=float(np.mean([x[1] for x in folds])),
                            inner_validation_1=folds[0][0],inner_score_1=folds[0][1],inner_validation_2=folds[1][0],inner_score_2=folds[1][1]))
    chosen=min(records,key=lambda r:(r['validation_score'],r['complexity'],-r['ridge']))
    all_train=pd.concat(list(training.values()),ignore_index=True)
    gram,rhs,info=prepared(all_train,family,chosen['complexity'])
    coefficients=np.linalg.solve(gram+chosen['ridge']*np.eye(len(rhs)),rhs)
    info.update(coefficients=coefficients,ridge=chosen['ridge'],family=family,complexity=chosen['complexity'],
                coefficient_count=len(coefficients),effective_df=float(np.trace(np.linalg.solve(gram+chosen['ridge']*np.eye(len(rhs)),gram))),
                training_trials=trials,training_rows=len(all_train))
    return info,chosen,records

def predict(model,rows):
    d,base=design_for(rows,model['family'],model)
    return base+d@model['coefficients']

def metrics(target,pred):
    target=np.asarray(target);pred=np.asarray(pred);err=target-pred
    mse=float(np.mean(err**2));var=float(np.mean((target-target.mean())**2))
    return dict(n=len(target),mse=mse,rmse=float(np.sqrt(mse)),mae=float(np.mean(np.abs(err))),r2=1-mse/var if var>0 else None,
                target_variance=var,outside_count=int((np.abs(pred)>1).sum()),clipped_mse=float(np.mean((target-np.clip(pred,-1,1))**2)))

def probability_metrics(p,target):
    one=np.eye(3)[target]
    return dict(n=len(target),nll=float(-np.log(p[np.arange(len(target)),target]).mean()),
                brier=float(((p-one)**2).sum(axis=1).mean()),
                rps=float(((np.cumsum(p,axis=1)[:,:-1]-np.cumsum(one,axis=1)[:,:-1])**2).sum(axis=1).mean()),
                accuracy=float((p.argmax(axis=1)==target).mean()))
