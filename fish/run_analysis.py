from .core import *
import argparse,itertools,time,sys,os
import pyarrow as pa
import pyarrow.parquet as pq

def merge_parquets(paths,target):
    writer=None
    schema=None
    try:
        for path in paths:
            t=pq.read_table(path)
            if writer is None:
                schema=t.schema;writer=pq.ParquetWriter(target,schema,compression='zstd')
            elif t.schema!=schema:
                t=t.cast(schema)
            writer.write_table(t)
    finally:
        if writer is not None:writer.close()

def run():
    protocol=read_protocol();manifest=json.loads((OUT/'input_manifest.json').read_text())
    if sha(OUT/'protocol.json')!=manifest['protocol_sha256']:raise ValueError('protocol changed after observation construction')
    obs=pd.read_parquet(OUT/'observation_frames.parquet')
    settings=protocol['continuous'];parts=OUT/'prediction_parts';models=OUT/'models';dp=OUT/'discrete_parts'
    for p in (parts,models,dp):p.mkdir(exist_ok=True)
    metrics_rows=[];choices=[];inventory=[];parts_list=[];discrete_parts=[];dm_rows=[]
    families={'current_matched':[settings['current_matched_degree']], 'current_flexible':settings['current_flexible_degrees'],
              'current_spline':settings['spline_breakpoints'],'history':[3],'state_trend':[3]}
    start=time.time()
    for variant in [v['name'] for v in protocol['variants']]:
        tables={tid:g.copy() for tid,g in obs.loc[obs.variant==variant].groupby('trial_id',sort=True)}
        for definition in ('heading','displacement'):
            for horizon in settings['horizons']:
                rows={tid:make_rows(t,definition,horizon) for tid,t in tables.items()}
                for held,eval_rows in rows.items():
                    g=tables[held];n=int(g.group_size.iloc[0]);trial=str(g.trial.iloc[0])
                    training={tid:r for tid,r in rows.items() if tid!=held and int(tables[tid].group_size.iloc[0])==n}
                    assert len(training)==2 and held not in training
                    if not len(eval_rows) or any(not len(r) for r in training.values()):
                        raise ValueError(f'insufficient rows {variant} {definition} {held} {horizon}')
                    common=dict(variant=variant,definition=definition,group_size=n,trial_id=held,trial=trial,horizon_frames=horizon)
                    predictions={'persistence':eval_rows.current.to_numpy()}
                    for family,complexities in families.items():
                        model,chosen,candidates=choose_and_fit(training,family,complexities,settings['ridge_grid'])
                        name=f'{variant}_{definition}_{held}_h{horizon}_{family}'
                        write_json(models/(name+'.json'),dict(**common,model=family,selection=chosen,parameters=model,protocol_sha256=manifest['protocol_sha256']))
                        for candidate in candidates:
                            choices.append(dict(**common,model=family,**candidate,selected=bool(candidate==chosen),training_trials='|'.join(training)))
                        inventory.append(dict(**common,model=family,training_trials='|'.join(training),
                            training_rows=sum(len(r) for r in training.values()),evaluation_rows=len(eval_rows),
                            coefficient_count=model['coefficient_count'],effective_df=model['effective_df'],complexity=chosen['complexity'],
                            ridge=chosen['ridge'],model_file=str((models/(name+'.json')).relative_to(OUT))))
                        predictions[family]=predict(model,eval_rows)
                    frames=[]
                    for family,pred in predictions.items():
                        assert np.isfinite(pred).all()
                        metrics_rows.append(dict(**common,model=family,**metrics(eval_rows.target,pred)))
                        frames.append(pd.DataFrame(dict(**common,model=family,frame=eval_rows.frame.to_numpy(np.int64),
                            target_frame=eval_rows.target_frame.to_numpy(np.int64),run_id=eval_rows.run_id.to_numpy(np.int64),
                            current=eval_rows.current.to_numpy(),target=eval_rows.target.to_numpy(),prediction=pred,clipped_prediction=np.clip(pred,-1,1))))
                    target=parts/f'{variant}_{definition}_{held}_h{horizon}.parquet'
                    pd.concat(frames,ignore_index=True).to_parquet(target,index=False,compression='zstd');parts_list.append(target)
                    print('continuous',variant,definition,held,'h',horizon,'n',len(eval_rows),'elapsed',round(time.time()-start,1),flush=True)
                pd.DataFrame(metrics_rows).to_csv(OUT/'recording_metrics.csv',index=False)
                pd.DataFrame(choices).to_csv(OUT/'training_selection.csv',index=False)
                pd.DataFrame(inventory).to_csv(OUT/'model_inventory.csv',index=False)
            drows={tid:discrete_rows(t,definition) for tid,t in tables.items()}
            for held,ev in drows.items():
                n=int(tables[held].group_size.iloc[0]);training=[tid for tid in tables if tid!=held and int(tables[tid].group_size.iloc[0])==n]
                cx=np.concatenate([drows[tid]['context'] for tid in training]);y=np.concatenate([drows[tid]['target'] for tid in training])
                for order in protocol['discrete']['orders']:
                    model=SparseBackoffMarkov(order,state_count=3,alpha=.5,reflection_pool=True).fit(cx[:,:order],y)
                    probability=model.predict_proba(ev['context'][:,:order]);np.testing.assert_allclose(probability.sum(axis=1),1,atol=1e-14)
                    common=dict(variant=variant,definition=definition,group_size=n,trial_id=held,order=order)
                    dm_rows.append(dict(**common,**probability_metrics(probability,ev['target']),training_trials='|'.join(training)))
                    grid=np.array(list(itertools.product(range(3),repeat=order)),dtype=np.int16)
                    values=model.predict_proba(grid)
                    levels=model._require_fitted()
                    write_json(models/f'{variant}_{definition}_{held}_K{order}.json',dict(**common,training_trials=training,alpha=.5,
                        grid=grid,probabilities=values,observed_contexts=[len(x.codes) for x in levels],stored_count_entries=sum(x.counts.size for x in levels),
                        protocol_sha256=manifest['protocol_sha256']))
                    part=pd.DataFrame(dict(**common,frame=ev['frame'],target_frame=ev['frame']+5,run_id=ev['run_id'],
                        target_state=ev['target'],p0=probability[:,0],p1=probability[:,1],p2=probability[:,2]))
                    for i in range(order):part[f'context_{i}']=ev['context'][:,i]
                    # Fixed schema across K1 and K4 for a single readable output.
                    for i in range(order,4):part[f'context_{i}']=-1
                    target=dp/f'{variant}_{definition}_{held}_K{order}.parquet'
                    part.to_parquet(target,index=False,compression='zstd');discrete_parts.append(target)
            pd.DataFrame(dm_rows).to_csv(OUT/'discrete_metrics.csv',index=False)
            print('discrete',variant,definition,'completed',flush=True)
    finalize_outputs(start)

def finalize_outputs(start=None):
    manifest=json.loads((OUT/'input_manifest.json').read_text())
    parts_list=sorted((OUT/'prediction_parts').glob('*.parquet'))
    discrete_parts=sorted((OUT/'discrete_parts').glob('*.parquet'))
    if len(parts_list)!=162 or len(discrete_parts)!=108:raise ValueError('incomplete result parts')
    merge_parquets(parts_list,OUT/'predictions.parquet')
    merge_parquets(discrete_parts,OUT/'discrete_predictions.parquet')
    write_json(OUT/'run_manifest.json',dict(protocol_sha256=manifest['protocol_sha256'],input_manifest_sha256=sha(OUT/'input_manifest.json'),
        predictions_rows=sum(pq.ParquetFile(p).metadata.num_rows for p in parts_list),continuous_cases=len(parts_list),
        discrete_rows=sum(pq.ParquetFile(p).metadata.num_rows for p in discrete_parts),discrete_cases=len(discrete_parts),
        elapsed_seconds=time.time()-start if start is not None else None,script_sha256={p.name:sha(p) for p in [Path(__file__).with_name(n) for n in ['core.py','run_analysis.py','build_observations.py']]},
        imported_source_sha256={p.name:sha(p) for p in [Path(__file__).resolve().parents[1]/'methods/models.py']},
        threads={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},processes=1))

if __name__=='__main__':
    if '--finalize-only' in sys.argv:finalize_outputs()
    else:run()
