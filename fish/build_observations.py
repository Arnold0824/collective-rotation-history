from .core import *
from scipy.optimize import linear_sum_assignment

def main():
    protocol=read_protocol();src=ROOT/Path(protocol['input_root'])
    tables=[];quality=[];boundary_rows=[];inputs=[];support_rows=[]
    for path in sorted(src.glob('*.npz')):
        if path.name.startswith('N151'):continue
        with np.load(path,allow_pickle=False) as z:
            pos=z['positions'];head=z['heading_angles'];frames=z['frames'];original=z['milling'];oldvalid=z['frame_valid'];oldrun=z['order_run_id']
            n=int(z['group_size']);trial=str(z['trial']);tid=f'N{n}{trial}';fps=float(z['frame_rate_hz'])
        assert fps==30 and np.all(np.diff(frames)==1)
        hd=np.stack((np.cos(head),np.sin(head)),axis=-1)
        finite=np.isfinite(pos).all(axis=-1)&np.isfinite(head)
        recomputed,_,_,_,_=rotation(pos,hd,finite)
        np.testing.assert_allclose(original,recomputed,atol=1e-14,rtol=0,equal_nan=True)
        inputs.append(dict(trial_id=tid,group_size=n,trial=trial,path=str(path),sha256=sha(path),frames=len(frames),fps=fps,
                           original_heading_reconstruction_max_error=float(np.nanmax(np.abs(original-recomputed)))))
        for spec in protocol['variants']:
            obs=paired_observables(pos,head,frames,spec['minimum_step_pixels'])
            if spec['name']=='primary':
                for k in np.flatnonzero(obs['boundary']):
                    dist=np.linalg.norm(pos[k,:,None,:]-pos[k-1,None,:,:],axis=-1)
                    ii,jj=linear_sum_assignment(dist)
                    delta=pos[k]-pos[k-1];length=np.linalg.norm(delta,axis=1)
                    boundary_rows.append(dict(trial_id=tid,frame=int(frames[k]),previous_frame=int(frames[k-1]),
                        same_index_median_pixels=float(np.median(length)),same_index_max_pixels=float(length.max()),
                        fraction_over_50_pixels=float((length>50).mean()),nearest_median_pixels=float(np.median(dist.min(axis=1))),
                        optimal_assignment_median_pixels=float(np.median(dist[ii,jj])),optimal_assignment_mean_pixels=float(dist[ii,jj].mean()),
                        optimal_identity_mismatch_count=int((ii!=jj).sum()),centroid_step_pixels=float(np.linalg.norm(pos[k].mean(axis=0)-pos[k-1].mean(axis=0))),
                        original_heading_previous=float(original[k-1]),original_heading_current=float(original[k]),
                        mean_heading_x_previous=float(np.cos(head[k-1]).mean()),mean_heading_y_previous=float(np.sin(head[k-1]).mean()),
                        mean_heading_x_current=float(np.cos(head[k]).mean()),mean_heading_y_current=float(np.sin(head[k]).mean())))
            good=obs['valid'];a=obs['heading'][good];b=obs['displacement'][good]
            q=dict(trial_id=tid,group_size=n,trial=trial,variant=spec['name'],minimum_step_pixels=spec['minimum_step_pixels'],
                   original_frames=len(frames),original_heading_valid_frames=int(oldvalid.sum()),paired_valid_frames=int(good.sum()),
                   paired_valid_fraction=float(good.mean()),paired_runs=int(len(np.unique(obs['run_id'][good]))),
                   inferred_boundary_count=int(obs['boundary'].sum()),first_or_gap_frames=int(np.sum(np.r_[True,np.diff(frames)!=1])),
                   non_boundary_insufficient_fraction_frames=int((~good&~obs['boundary']&(frames>frames[0])).sum()),
                   common_direction_count_mean=float(obs['count'][good].mean()) if good.any() else None,
                   heading_vs_displacement_correlation=float(np.corrcoef(a,b)[0,1]) if good.sum()>1 else None,
                   heading_vs_displacement_mae=float(np.abs(a-b).mean()) if good.any() else None,
                   paired_heading_vs_original_mae=float(np.abs(a-original[good]).mean()) if good.any() else None,
                   state_agreement=float((states(a)==states(b)).mean()) if good.any() else None,
                   heading_exact_minus_threshold=int((a==-.3).sum()),heading_exact_plus_threshold=int((a==.3).sum()),
                   displacement_exact_minus_threshold=int((b==-.3).sum()),displacement_exact_plus_threshold=int((b==.3).sum()))
            quality.append(q)
            usable=obs['individual_support'];angles=obs['angle_mean'][good]
            finite_step=np.isfinite(obs['step']);all_positive=obs['step'][finite_step&(obs['step']>0)]
            support_rows.append(dict(trial_id=tid,variant=spec['name'],individual_intervals=int(finite_step.sum()),
                zero_step_count=int((obs['step']==0).sum()),near_zero_le_1e12_count=int((finite_step&(obs['step']<=1e-12)).sum()),
                step_le_0p5_count=int((finite_step&(obs['step']<=.5)).sum()),step_le_1_count=int((finite_step&(obs['step']<=1)).sum()),
                step_median_pixels=float(np.median(all_positive)),step_99pct_pixels=float(np.quantile(all_positive,.99)),
                paired_angle_mean_degrees=float(angles.mean()) if len(angles) else None,
                paired_angle_gt90_fraction=float(obs['angle_gt90_fraction'][good].mean()) if good.any() else None))
            tables.append(pd.DataFrame(dict(trial_id=tid,group_size=n,trial=trial,variant=spec['name'],frame=frames,fps=fps,
                original_heading=original,original_valid=oldvalid,original_run_id=oldrun,
                heading=obs['heading'],displacement=obs['displacement'],valid=good,run_id=obs['run_id'],
                common_individual_count=obs['count'],base_individual_count=obs['base_count'],identity_boundary=obs['boundary'],
                mean_heading_displacement_angle_deg=obs['angle_mean'],fraction_angle_gt90=obs['angle_gt90_fraction'])))
        print(tid,'observations constructed',flush=True)
    all_obs=pd.concat(tables,ignore_index=True)
    all_obs.to_parquet(OUT/'observation_frames.parquet',index=False,compression='zstd')
    pd.DataFrame(quality).to_csv(OUT/'observation_quality.csv',index=False)
    pd.DataFrame(support_rows).to_csv(OUT/'individual_support_and_angle_summary.csv',index=False)
    pd.DataFrame(boundary_rows).to_csv(OUT/'identity_boundary_evidence.csv',index=False)
    boundaries={tid:[int(r['frame']) for r in boundary_rows if r['trial_id']==tid] for tid in all_obs.trial_id.unique()}
    write_json(OUT/'input_manifest.json',dict(protocol_sha256=sha(OUT/'protocol.json'),inputs=inputs,identity_boundaries=boundaries,
        boundary_interpretation='Segment boundaries inferred from indexed jumps and optimal position matching; original video-file boundary metadata are unavailable.',
        source_steps_per_array_row=1,frame_origin='Zero-based rows of the published processed arrays, not absolute source-video frame numbers.',
        no_added_smoothing_or_interpolation=True,upstream_tracking_preprocessing='The public processing notebooks add no smoothing or interpolation; upstream tracking and manual correction are not fully documented.'))
    coverage=[]
    for (variant,tid),g in all_obs.groupby(['variant','trial_id']):
        for h in protocol['continuous']['horizons']:
            a=make_rows(g,'heading',h);b=make_rows(g,'displacement',h)
            assert np.array_equal(a.frame,b.frame)
            assert np.array_equal(a.target_frame,b.target_frame)
            coverage.append(dict(variant=variant,trial_id=tid,group_size=int(g.group_size.iloc[0]),horizon_frames=h,paired_prediction_rows=len(a)))
    pd.DataFrame(coverage).to_csv(OUT/'sample_coverage.csv',index=False)
    print(pd.DataFrame(quality)[['trial_id','variant','paired_valid_frames','paired_runs','heading_vs_displacement_correlation']].to_string(index=False))
    print(pd.DataFrame(coverage).groupby(['variant','horizon_frames']).paired_prediction_rows.agg(['min','sum']).to_string())

if __name__=='__main__':main()
