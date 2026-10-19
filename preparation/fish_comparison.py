"""Numerical comparisons that distinguish choices, predictions and diagnostics.

Reference files are never changed. Every candidate loss is retained, including
large rejected losses, and all fitted models are replayed on the observed data.
"""
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd

GROUP = ['variant', 'definition', 'trial_id', 'horizon_frames', 'model']
SCORES = ['validation_score', 'inner_score_1', 'inner_score_2']


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False,
                                   default=lambda x: x.item())+'\n', encoding='utf-8')


def compare_candidates(reference, generated, diagnostics):
    keys = GROUP + ['complexity', 'ridge']
    old = pd.read_csv(reference).set_index(keys).sort_index()
    new = pd.read_csv(generated).set_index(keys).sort_index()
    if not old.index.equals(new.index) or set(old.columns) != set(new.columns):
        raise AssertionError('Candidate identities or columns differ')
    if old.index.has_duplicates:
        raise AssertionError('Duplicate candidate identities')
    for name in old.columns:
        if name not in SCORES and not old[name].equals(new[name]):
            raise AssertionError(f'Candidate metadata or selected flags differ: {name}')
    a, b = old.reset_index(), new.reset_index()
    detail = a[keys+['selected']].copy()
    result = {'table': 'training_selection.csv', 'rows': len(a), 'score_differences': {},
              'selected_score_tolerance': {'rtol': 1e-8, 'atol': 1e-10},
              'unselected_rule': 'unchanged complete ordering and winner; average-loss gap exceeds 100 times the observed candidate and winner score shifts plus 1e-10'}
    failures, groups = [], []
    for name in SCORES:
        av, bv = a[name].to_numpy(float), b[name].to_numpy(float)
        if not np.isfinite(av).all() or not np.isfinite(bv).all():
            raise AssertionError('Non-finite candidate score')
        delta = np.abs(av-bv)
        detail[name+'_reference'] = av
        detail[name+'_generated'] = bv
        detail[name+'_abs_difference'] = delta
        selected = a.selected.to_numpy(bool)
        within = np.isclose(av, bv, rtol=1e-8, atol=1e-10)
        if not within[selected].all():
            failures.append(f'Selected candidate scores exceed tight relative tolerance: {name}')
        result['score_differences'][name] = {
            'max_abs': float(delta.max()), 'over_absolute_1e10': int((delta>1e-10).sum()),
            'selected_max_abs': float(delta[selected].max()),
            'selected_max_relative': float((delta[selected]/np.maximum(np.abs(av[selected]),1e-300)).max()),
            'selected_over_absolute_1e10': int((delta[selected]>1e-10).sum())}
    for label, og in a.groupby(GROUP, sort=True):
        ng = b.loc[og.index]
        os = og.sort_values(['validation_score','complexity','ridge'], ascending=[True,True,False])
        ns = ng.sort_values(['validation_score','complexity','ridge'], ascending=[True,True,False])
        winner_a, winner_b = os.iloc[0], ns.iloc[0]
        full_order_equal = list(os.index)==list(ns.index)
        selected_matches = (len(og.loc[og.selected])==1 and len(ng.loc[ng.selected])==1
                            and bool(winner_a.selected) and bool(winner_b.selected))
        if not full_order_equal or not selected_matches:
            failures.append(f'Candidate ordering, argmin or recorded choice differs: {label}')
        oldrank = {ix:i+1 for i,ix in enumerate(os.index)}
        newrank = {ix:i+1 for i,ix in enumerate(ns.index)}
        for ix in og.index:
            detail.loc[ix,'rank_reference'] = oldrank[ix]
            detail.loc[ix,'rank_generated'] = newrank[ix]
        gap = np.minimum(og.validation_score,ng.validation_score)-max(winner_a.validation_score,winner_b.validation_score)
        shift = np.abs(og.validation_score-ng.validation_score)+abs(winner_a.validation_score-winner_b.validation_score)+1e-10
        detail.loc[og.index,'gap_to_winner'] = gap
        detail.loc[og.index,'score_shift_bound'] = shift
        detail.loc[og.index,'gap_over_shift'] = gap/shift
        changed = (detail.loc[og.index,[n+'_abs_difference' for n in SCORES]].max(axis=1)>1e-10)&~og.selected
        if not (gap[changed]>100*shift[changed]).all():
            failures.append(f'Unselected numerical change is too close to the winner: {label}')
        groups.append(dict(zip(GROUP,label), full_order_equal=full_order_equal,
                           selected_matches_argmin=selected_matches,
                           winner_margin_reference=float(os.iloc[1].validation_score-winner_a.validation_score),
                           winner_margin_generated=float(ns.iloc[1].validation_score-winner_b.validation_score)))
    detail.to_csv(diagnostics/'candidate_score_differences.csv',index=False)
    pd.DataFrame(groups).to_csv(diagnostics/'candidate_order_checks.csv',index=False)
    changed = (detail[[n+'_abs_difference' for n in SCORES]].max(axis=1)>1e-10)&~detail.selected
    result.update(groups=len(groups), complete_order_unchanged=all(x['full_order_equal'] for x in groups),
                  selected_matches_argmin=all(x['selected_matches_argmin'] for x in groups),
                  unselected_min_gap_over_shift=float(detail.loc[changed,'gap_over_shift'].min()) if changed.any() else None,
                  failures=failures, passed=not failures)
    write_json(diagnostics/'candidate_comparison.json',result)
    if failures:
        raise AssertionError('; '.join(failures))
    return result


def compare_models(reference, generated, diagnostics):
    # core exposes pure numerical helpers alongside environment-based CLI paths.
    old_environment = {name:os.environ.get(name) for name in ('ROTATION_DATA_DIR','ROTATION_FISH_OUTPUT_DIR')}
    try:
        os.environ['ROTATION_DATA_DIR']=str(generated.parent)
        os.environ['ROTATION_FISH_OUTPUT_DIR']=str(generated)
        from fish.core import make_rows, discrete_rows, predict, prepared, design_for
    finally:
        for name,value in old_environment.items():
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value
    oldfiles = {p.name:p for p in (reference/'models').glob('*.json')}
    newfiles = {p.name:p for p in (generated/'models').glob('*.json')}
    if not oldfiles or oldfiles.keys()!=newfiles.keys():
        raise AssertionError('Fitted-model identities differ')
    observations = pd.read_parquet(generated/'observation_frames.parquet')
    tables = {(str(v),str(t)):g for (v,t),g in observations.groupby(['variant','trial_id'],sort=False)}
    rows_cache, differences, replays, edf_checks, residual_checks, failures = {}, [], [], [], [], []
    models = {}
    coefficient_exceptions = set()

    def compare(a,b,path,filename):
        if isinstance(a,dict):
            if set(a)!=set(b):
                raise AssertionError(f'Model keys differ: {filename}/{path}')
            for k in a:
                if k!='protocol_sha256': compare(a[k],b[k],path+'/'+k,filename)
        elif isinstance(a,list):
            if not isinstance(b,list) or len(a)!=len(b):
                raise AssertionError(f'Model array dimensions differ: {filename}/{path}')
            for i,(av,bv) in enumerate(zip(a,b)):compare(av,bv,path+'/'+str(i),filename)
        elif isinstance(a,float):
            if not isinstance(b,(float,int)) or not np.isfinite(a) or not np.isfinite(b):
                raise AssertionError(f'Non-finite model value: {filename}/{path}')
            delta=abs(a-b)
            if delta: differences.append({'model':filename,'field':path,'reference':a,'generated':b,'abs_difference':delta})
            if path.startswith('/parameters/coefficients/'):
                if delta>1e-7:coefficient_exceptions.add(filename)
            elif path=='/parameters/effective_df':
                pass  # Compared with an independent SVD trace below when needed.
            elif path.startswith('/selection/') and path.split('/')[-1] in SCORES:
                if not np.isclose(a,b,rtol=1e-8,atol=1e-10):failures.append(filename+path)
            elif path.startswith(('/parameters/design_scale/','/parameters/state_scale/')):
                if not np.isclose(a,b,rtol=1e-13,atol=1e-10):failures.append(filename+path)
            elif delta>1e-10:failures.append(filename+path)
        elif a!=b:
            raise AssertionError(f'Model metadata, integer or choice differs: {filename}/{path}')

    def rows_for(variant,definition,horizon,trial):
        key=(variant,definition,horizon,trial)
        if key not in rows_cache:rows_cache[key]=make_rows(tables[(variant,trial)],definition,horizon)
        return rows_cache[key]

    def numerical_model(record):
        return {k:np.asarray(v) if isinstance(v,list) and k!='training_trials' else v
                for k,v in record['parameters'].items()}

    try:
        for filename in sorted(oldfiles):
            a=json.loads(oldfiles[filename].read_text());b=json.loads(newfiles[filename].read_text())
            compare(a,b,'',filename)
            models[filename]=(a,b)
            if 'parameters' not in a:
                if a['probabilities']!=b['probabilities']:
                    failures.append(filename+'/discrete_probabilities_not_exact')
                continue
            oldmodel,newmodel=numerical_model(a),numerical_model(b)
            variant,definition,horizon=a['variant'],a['definition'],int(a['horizon_frames'])
            evaluation=rows_for(variant,definition,horizon,a['trial_id'])
            op,npred=predict(oldmodel,evaluation),predict(newmodel,evaluation)
            delta=float(np.max(np.abs(op-npred)))
            replays.append({'model':filename,'support':'evaluation','rows':len(evaluation),'max_abs_prediction_difference':delta})
            if delta>1e-7:failures.append(filename+'/evaluation_prediction')
            edf_difference=abs(oldmodel['effective_df']-newmodel['effective_df'])
            if filename in coefficient_exceptions or edf_difference>1e-10:
                train=pd.concat([rows_for(variant,definition,horizon,t) for t in oldmodel['training_trials']],ignore_index=True)
                op,npred=predict(oldmodel,train),predict(newmodel,train)
                delta=float(np.max(np.abs(op-npred)))
                replays.append({'model':filename,'support':'training','rows':len(train),'max_abs_prediction_difference':delta})
                if delta>1e-7:failures.append(filename+'/training_prediction')
                coef_shift=float(np.max(np.abs(oldmodel['coefficients']-newmodel['coefficients'])))
                coef_scale=max(1.,float(np.max(np.abs(oldmodel['coefficients']))))
                if coef_shift>1e-5*coef_scale:failures.append(filename+'/coefficient_relative_shift')
                if filename in coefficient_exceptions:
                    gram,rhs,_=prepared(train,newmodel['family'],newmodel['complexity'])
                    matrix=gram+newmodel['ridge']*np.eye(len(rhs))
                    matrix_norm=float(np.linalg.norm(matrix,ord=2));rhs_norm=float(np.linalg.norm(rhs))
                    for label,model in [('reference',oldmodel),('generated',newmodel)]:
                        coef=model['coefficients']
                        residual=float(np.linalg.norm(matrix@coef-rhs)/(matrix_norm*np.linalg.norm(coef)+rhs_norm))
                        residual_checks.append({'model':filename,'source':label,'normalized_equation_residual':residual})
                        if residual>1e-12:failures.append(filename+'/'+label+'/normal_equation_residual')
                if edf_difference>1e-10:
                    # SVD evaluation avoids solving the nearly singular Gram system.
                    design,_=design_for(train,newmodel['family'],newmodel)
                    singular=np.linalg.svd(design,compute_uv=False)
                    squares=singular**2;ridge=float(newmodel['ridge'])
                    stable_df=float(np.sum(squares/(squares+ridge)))
                    condition=float((squares.max()+ridge)/(squares.min()+ridge))
                    bound=max(1e-10,8*np.finfo(float).eps*len(squares)*condition)
                    ok=abs(oldmodel['effective_df']-stable_df)<=bound and abs(newmodel['effective_df']-stable_df)<=bound
                    edf_checks.append({'model':filename,'reference':oldmodel['effective_df'],'generated':newmodel['effective_df'],
                        'svd_effective_df':stable_df,'regularized_design_condition':condition,
                        'roundoff_bound':bound,'passed':bool(ok)})
                    if not ok:failures.append(filename+'/effective_df_svd_check')
        # Verify the files to be installed are predictions of the regenerated models.
        for part in sorted((generated/'prediction_parts').glob('*.parquet')):
            block=pd.read_parquet(part)
            for family,g in block.groupby('model',sort=False):
                r=g.iloc[0]
                rows=rows_for(r.variant,r.definition,int(r.horizon_frames),r.trial_id)
                for field in ['frame','target_frame','run_id','current','target']:
                    if not np.array_equal(g[field].to_numpy(),rows[field].to_numpy()):failures.append(part.name+'/'+family+'/'+field)
                if family=='persistence':estimate=rows.current.to_numpy()
                else:
                    filename=f'{r.variant}_{r.definition}_{r.trial_id}_h{int(r.horizon_frames)}_{family}.json'
                    estimate=predict(numerical_model(models[filename][0]),rows)
                delta=float(np.max(np.abs(estimate-g.prediction.to_numpy())))
                replays.append({'model':part.stem+'_'+family,'support':'saved_prediction','rows':len(g),'max_abs_prediction_difference':delta})
                if delta>1e-7:failures.append(part.name+'/'+family+'/saved_prediction')
        for part in sorted((generated/'discrete_parts').glob('*.parquet')):
            block=pd.read_parquet(part);r=block.iloc[0];order=int(r.order)
            rows=discrete_rows(tables[(r.variant,r.trial_id)],r.definition)
            for field,expected in [('frame',rows['frame']),('target_frame',rows['frame']+5),
                                   ('run_id',rows['run_id']),('target_state',rows['target'])]:
                if not np.array_equal(block[field].to_numpy(),expected):failures.append(part.name+'/'+field)
            context=block[[f'context_{i}' for i in range(order)]].to_numpy(dtype=int)
            if not np.array_equal(context,rows['context'][:,:order]):failures.append(part.name+'/context')
            filename=f'{r.variant}_{r.definition}_{r.trial_id}_K{order}.json'
            model=models[filename][0]
            index=context@(3**np.arange(order-1,-1,-1))
            expected=np.asarray(model['probabilities'])[index]
            actual=block[['p0','p1','p2']].to_numpy()
            delta=float(np.max(np.abs(expected-actual)))
            replays.append({'model':filename,'support':'saved_discrete_prediction','rows':len(block),'max_abs_prediction_difference':delta})
            if not np.array_equal(expected,actual):failures.append(filename+'/saved_discrete_probability')
    finally:
        pd.DataFrame(differences).to_csv(diagnostics/'model_value_differences.csv',index=False)
        pd.DataFrame(replays).to_csv(diagnostics/'model_prediction_replay.csv',index=False)
        pd.DataFrame(edf_checks).to_csv(diagnostics/'effective_df_svd_checks.csv',index=False)
        pd.DataFrame(residual_checks).to_csv(diagnostics/'coefficient_equation_residuals.csv',index=False)
    result={'passed':not failures,'model_files_compared':len(oldfiles),
            'coefficient_comparison': 'Absolute 1e-7; larger changes require <=1e-5 relative to max(1, coefficient norm), and every observed training/evaluation prediction within 1e-7.',
            'coefficient_exceptions_replayed':len(coefficient_exceptions),
            'coefficient_equation_residual_limit':1e-12,
            'max_coefficient_equation_residual':max((x['normalized_equation_residual'] for x in residual_checks),default=0),
            'effective_df_comparison':'Original and regenerated trace each checked against a direct SVD value with 8*machine_epsilon*coefficient_count*regularized_condition roundoff bound; all differences retained.',
            'effective_df_svd_checks':len(edf_checks),
            'prediction_atol':1e-7,'prediction_rtol':0,
            'max_prediction_difference':max(x['max_abs_prediction_difference'] for x in replays),
            'metadata_exclusion':'protocol_sha256 only', 'failures':failures}
    write_json(diagnostics/'model_comparison.json',result)
    if failures:raise AssertionError('Model comparison failed: '+str(failures[:10]))
    return result
