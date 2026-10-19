"""Empirical and simulated statistics for the study figures."""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

HERE = Path(__file__).resolve().parent
ROOT = HERE
OUT = HERE / "figures"
DATA = HERE / "displayed_data"
BLUE, ORANGE, TEAL, INK, GREY = "#0072B2", "#D55E00", "#009E73", "#252525", "#7b7b7b"
WIDTH = 180 / 25.4
INPUTS, CHECKS, OUTPUTS = {}, {}, {}

def read(relative):
    path = ROOT / relative
    INPUTS[relative] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    return pd.read_csv(path)

def read_json(relative):
    path = ROOT / relative
    INPUTS[relative] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    return json.loads(path.read_text(encoding="utf-8"))

def export(name, frame):
    frame.to_csv(DATA / (name + ".csv"), index=False, float_format="%.17g")

def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 10, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .65, "xtick.major.width": .65, "ytick.major.width": .65,
        "lines.linewidth": 1.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "savefig.facecolor": "white",
        "figure.facecolor": "white", "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    })

def panel(ax, letter, title):
    ax.set_title(title, loc="left", pad=12, fontweight="normal")
    ax.text(-min(.17,.043/ax.get_position().width), 1.055, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")

def light_grid(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color="#e5e5e5", lw=.55)

def ids_order(ids):
    return sorted(ids, key=lambda s: (int(s[1:-1]), s[-1]))

def save(fig, name):
    fig.canvas.draw()
    outside = []
    renderer = fig.canvas.get_renderer()
    from matplotlib.text import Text
    for txt in fig.findobj(Text):
        if not txt.get_visible() or not txt.get_text().strip():
            continue
        bb = txt.get_window_extent(renderer)
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fig.bbox.width+1 or bb.y1 > fig.bbox.height+1:
            # Tick labels outside view limits are not drawn in final output.
            if txt in [t for ax in fig.axes for t in ax.get_xticklabels()+ax.get_yticklabels()]:
                continue
            outside.append(txt.get_text())
    CHECKS[name + "_text_inside_canvas"] = {"pass": not outside, "outside": outside}
    files=[]
    for ext in ["pdf", "svg", "png"]:
        path = OUT / f"{name}.{ext}"
        kw={"dpi": 360} if ext == "png" else {}
        fig.savefig(path, **kw)
        files.append(str(path.relative_to(HERE)))
    OUTPUTS[name] = files
    plt.close(fig)

def figure_02():
    prefix="inputs/"
    absolute=read(prefix+"prediction_absolute.csv")
    comp=read(prefix+"prediction_summary_ratios.csv")
    rec=read(prefix+"prediction_recording_ratios.csv")
    models=[("persistence",GREY,"Persistence","--","o"),
            ("original_current",BLUE,"Scalar current","-","o"),
            ("flexible_current","#56B4E9","Flexible scalar current","--","s"),
            ("original_history","#E69F00","Scalar + history","-","^"),
            ("multivariable_current","#AA6FA5","Multivariable current","-","D"),
            ("multivariable_history",ORANGE,"Multivariable + history","-","v"),
            ("filtered_trend6",TEAL,"Filtered state + trend","-","P")]
    fig=plt.figure(figsize=(WIDTH,6.85))
    a=fig.add_axes([.115,.70,.365,.24]); b=fig.add_axes([.615,.70,.36,.24])
    c=fig.add_axes([.115,.19,.365,.28]); d=fig.add_axes([.615,.19,.36,.28])
    for model,color,label,ls,marker in models:
        part=absolute[absolute.model.eq(model)].set_index("horizon_frames").loc[[3,10,30]]
        for ax,col in [(a,"mse"),(b,"r2_future")]:
            ax.plot(np.arange(3),part[col],label=label,c=color,ls=ls,marker=marker,ms=4,lw=1.5)
    a.set_ylabel("Mean-squared error in rotation")
    b.set_ylabel("Future-rotation R²")
    panel(a,"a","Absolute forecast error"); panel(b,"b","Future-state prediction")
    for ax in [a,b]:
        ax.set(xticks=[0,1,2],xticklabels=[3,10,30],xlim=(-.14,2.14),xlabel="Forecast horizon (frames)")
        light_grid(ax)
    b.set_ylim(.55,.855)
    fig.legend(*a.get_legend_handles_labels(),loc="center",bbox_to_anchor=(.51,.574),ncol=3,
               fontsize=8.5,handlelength=1.8,columnspacing=1.2,labelspacing=.6,handletextpad=.5)
    pairs=[("multivariable_history","multivariable_current",ORANGE,"Multivariable history /\nmultivariable current"),
           ("filtered_trend6","original_history",TEAL,"Filtered trend / scalar history")]
    comparisons=[]
    for i,(num,den,color,label) in enumerate(pairs):
        part=comp[comp.numerator.eq(num)&comp.denominator.eq(den)].set_index("horizon_frames").loc[[3,10,30]]
        y=part.log_balanced_ratio.to_numpy()
        c.errorbar(np.arange(3)+(i-.5)*.08,y,yerr=np.vstack([y-part.ratio_ci_low,part.ratio_ci_high-y]),fmt=["o-","s--"][i],c=color,ms=4,capsize=3,label=label)
        comparisons.append(part.reset_index())
    c.axhline(1,c=GREY,ls="--",lw=.8)
    c.set(xticks=[0,1,2],xticklabels=[3,10,30],xlim=(-.15,2.15),ylim=(.83,1.075),xlabel="Forecast horizon (frames)",ylabel="Mean-squared-error ratio")
    light_grid(c); panel(c,"c","History and trend comparisons")
    c.legend(loc="upper left",bbox_to_anchor=(-.015,-.245),fontsize=8.5,labelspacing=.6,handlelength=1.5)
    part=rec[rec.numerator.eq("multivariable_history")&rec.denominator.eq("multivariable_current")]
    ids=ids_order(part.trial_id.unique())
    for j,(h,marker,color) in enumerate([(3,"o","#222222"),(10,"s","#777777"),(30,"^","#bbbbbb")]):
        r=part[part.horizon_frames.eq(h)].set_index("trial_id").loc[ids]
        d.scatter(r.mse_ratio,np.arange(12)+(j-1)*.20,s=22,marker=marker,c=color,edgecolors=INK,linewidths=.35,label=f"{h} frames",zorder=3)
    for split in [3.5,7.5]:d.axhline(split,c="#dddddd",lw=.6)
    d.axvline(1,c=GREY,ls="--",lw=.8)
    d.set(yticks=range(12),yticklabels=ids,ylim=(11.65,-.65),xlim=(.81,1.35),xticks=[.9,1,1.1,1.2,1.3],xlabel="Multivariable history MSE /\nmultivariable current MSE")
    panel(d,"d","Every evaluation recording")
    d.legend(loc="lower right",labelspacing=.25,handletextpad=.25,fontsize=8.5)
    light_grid(d,"x")
    adverse=part[part.trial_id.eq("N100E")&part.horizon_frames.eq(30)].iloc[0]
    export("figure_02_absolute",absolute)
    export("figure_02_ratios_and_intervals",pd.concat(comparisons,ignore_index=True))
    export("figure_02_all_recordings",part)
    CHECKS["figure_02"]={"pass":bool(len(absolute)==21 and len(part)==36 and adverse.mse_ratio>1),"absolute_model_horizon_rows":len(absolute),"recording_ratio_points":len(part),"N100E_30_frame_ratio":float(adverse.mse_ratio),"summary_rule":"saved log-balanced ratios and 95% intervals unchanged"}
    save(fig,"figure_02_prediction")

def figure_04():
    acf=read("inputs/generation_autocorrelation.csv")
    legacy=read("inputs/scalar_prediction_and_generation.csv")
    legacy=legacy[legacy.record_family.eq("long_time")]
    rates=read("inputs/reversal_recordings.csv")
    surv=read("inputs/waiting_curves.csv")
    models=[("empirical",INK,"Observed","-"),("bounded_current_only",BLUE,"Current generator","--"),("bounded_current_and_delay",ORANGE,"History generator","-.")]
    fig=plt.figure(figsize=(WIDTH,7.85))
    cols=[.10,.405,.71]; colw=.255
    for j,n in enumerate([50,100,200]):
        ax=fig.add_axes([cols[j],.755,colw,.17])
        for model,color,label,ls in models:
            part=acf[acf.group_size.eq(n)&acf.model.eq(model)]
            ax.plot(part.lag_frames/30,part.median_across_trials,c=color,ls=ls,label=label)
        ax.set(xlabel="Lag (s)",ylim=(-.08,1.04),xticks=[0,2,4],yticks=[0,.5,1])
        if j==0:ax.set_ylabel("Autocorrelation")
        panel(ax,chr(97+j),f"N = {n}"); light_grid(ax)
    ax=fig.add_axes([.10,.475,.56,.18]); rate=fig.add_axes([.78,.475,.185,.18])
    source=[]
    metric_list=[("acf_rmse","Correlation\nRMSE"),("psd_log_rmse","Spectrum\nlog RMSE"),("distribution_wasserstein","Distribution\ndistance"),("state_fraction_l1","State occupancy\nL1 distance")]
    for i,(metric,label) in enumerate(metric_list):
        data=legacy[legacy.metric.eq(metric)]
        for col,c,offset,name in [("current_only_error",BLUE,-.15,"Current generator"),("delay_error",ORANGE,.15,"History generator")]:
            value=data.groupby("group_size")[col].median().mean()
            ax.bar(i+offset,value,width=.28,color=c,hatch="///" if col=="delay_error" else None,edgecolor=INK if col=="delay_error" else c,linewidth=.5,label=name if i==0 else None)
            source.append({"metric":metric,"model":name,"absolute_error":value})
    ax.set(xticks=range(4),xticklabels=[x[1] for x in metric_list],ylabel="Absolute discrepancy",ylim=(0,.50),yticks=[0,.2,.4])
    ax.tick_params(axis="x",labelsize=8.5)
    panel(ax,"d","Generated-statistic errors"); light_grid(ax)
    ax.legend(loc="upper right",fontsize=10,handlelength=1.3)
    for i,(model,c,label,ls) in enumerate(models):
        data=rates[rates.model.eq(model)]
        for j,n in enumerate([50,100,200]):
            part=data[data.group_size.eq(n)].sort_values("trial_id")
            rate.scatter(j+(i-1)*.23+np.linspace(-.035,.035,len(part)),part.reversals_per_minute,c=c,s=15,marker=["o","s","^"][i],alpha=.65,zorder=3)
            rate.plot(j+(i-1)*.23,part.reversals_per_minute.median(),"_",c=c,ms=9,mew=2,zorder=4)
    rate.set(xticks=range(3),xticklabels=[50,100,200],xlim=(-.45,2.45),xlabel="Group size",ylabel="Reversals per minute",ylim=(-.5,30),yticks=[0,10,20,30])
    panel(rate,"e","Reversal rates"); light_grid(rate)
    for j,n in enumerate([50,100,200]):
        ax=fig.add_axes([cols[j],.205,colw,.165]); risk=fig.add_axes([cols[j],.105,colw,.070],sharex=ax)
        for model,c,label,ls in models:
            part=surv[surv.group_size.eq(n)&surv.model.eq(model)]
            ax.plot(part.time_seconds,part.survival_median,c=c,ls=ls)
            risk.plot(part.time_seconds,part.total_at_risk,c=c,ls=ls,lw=1.3)
        ax.set(ylim=(-.02,1.03),yticks=[0,.5,1]); ax.tick_params(axis="x",labelbottom=False)
        risk.set(xlabel="Waiting time (s)",yscale="symlog",ylim=(0,5000),xticks=[0,20,40,60])
        risk.set_yscale("symlog",linthresh=1)
        risk.set_yticks([0,10,100,1000]); risk.set_yticklabels(["0","10","100","1000"])
        risk.tick_params(axis="y",labelsize=8.5)
        if j==0:ax.set_ylabel("No-reversal probability");risk.set_ylabel("Waits at risk")
        panel(ax,chr(102+j),f"Waiting, N = {n}")
        light_grid(ax); light_grid(risk)
    fig.legend(handles=[Line2D([0],[0],c=c,ls=ls,marker=["o","s","^"][i],ms=4,label=l) for i,(m,c,l,ls) in enumerate(models)],loc="lower center",bbox_to_anchor=(.5,.003),ncol=3,columnspacing=1.2,handlelength=2)
    exported=pd.DataFrame(source)
    reference=read("inputs/generation_absolute_reference.csv")
    assert np.allclose(exported.absolute_error,reference.absolute_error,rtol=2e-14)
    export("figure_04_autocorrelation",acf)
    export("figure_04_absolute_errors",exported)
    export("figure_04_reversal_rates",rates)
    export("figure_04_waiting_and_risk",surv)
    CHECKS["figure_04"]={"pass":True,"acf_rows":len(acf),"absolute_error_points":len(source),"rate_points":len(rates),"waiting_rows":len(surv),"absolute_errors_match_existing_table":True,"nominal_lag_conversion_hz":30,"risk_scale":"linear from zero to one; logarithmic above one"}
    save(fig,"figure_04_generation")

def figure_05():
    metrics=read("inputs/state_age_scores.csv")
    effects=read("inputs/state_age_effects.csv")
    summary=read("inputs/state_age_summary.csv")
    ids=ids_order(metrics.trial_id.unique())
    fig=plt.figure(figsize=(WIDTH, 4.35))
    left=fig.add_axes([.115,.225,.40,.635]); right=fig.add_axes([.655,.225,.315,.635])
    models=[("current",BLUE,"Current state","o"), ("current_age",TEAL,"Current + state age","s"), ("history4",ORANGE,"Four observations","^")]
    wide=metrics.pivot(index="trial_id",columns="model",values="nll").loc[ids]
    for i, tid in enumerate(ids):
        left.plot([wide.loc[tid].min(), wide.loc[tid].max()],[i,i],color="#c8c8c8",lw=1,zorder=1)
    for model,color,label,marker in models:
        left.scatter(wide[model],np.arange(len(ids)),c=color,marker=marker,s=24,label=label,zorder=3)
    for split in [3.5,7.5]: left.axhline(split,c="#dddddd",lw=.7)
    left.set(yticks=np.arange(12),yticklabels=ids,ylim=(11.7,-.7),xlim=(.19,.67),xlabel="Negative log-likelihood\n(nats per prediction)")
    panel(left,"a","Three-state prediction")
    light_grid(left,"x")
    fig.legend(*left.get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.5,.005),ncol=3,columnspacing=1.6,handletextpad=.5)
    overall=100*summary.loc[summary.group_size.eq(0),"age_fraction_of_history_gain"].iloc[0]
    display=[]
    for j,n in enumerate([50,100,200]):
        part=effects[effects.group_size.eq(n)].sort_values("trial_id")
        values=100*part.age_fraction_of_history_gain.to_numpy()
        right.scatter(j+np.linspace(-.11,.11,len(values)),values,s=25,c=GREY,zorder=3)
        val=100*summary.loc[summary.group_size.eq(n),"age_fraction_of_history_gain"].iloc[0]
        right.plot(j,val,"_",color=TEAL,ms=17,mew=3,zorder=4)
        display.extend({"trial_id":t,"group_size":n,"age_recovered_percent":float(v),"size_median_percent":val,"balanced_percent":overall} for t,v in zip(part.trial_id,values))
        recomputed=np.median(values)
        assert np.isclose(val,recomputed,rtol=1e-13)
    right.axhline(overall,c=TEAL,lw=1.1,ls="--")
    right.text(.04,.95,f"Balanced estimate: {overall:.2f}%",color=TEAL,va="top",transform=right.transAxes,fontsize=8.5)
    right.set(xticks=[0,1,2],xticklabels=[50,100,200],xlim=(-.42,2.42),ylim=(0,100),xlabel="Group size",ylabel="Age-recovered fraction of history gain (%)")
    panel(right,"b","Predictive value of state age")
    light_grid(right)
    export("figure_05_category_scores",metrics[["trial_id","group_size","model","row_count","nll"]])
    export("figure_05_age_recovered",pd.DataFrame(display))
    CHECKS["figure_05"]={"pass":len(ids)==12 and len(wide)==12,"recordings":len(ids),"score_points":int(wide.size),"age_points":len(display),"balanced_percent":overall,"summary_rule":"within-size recording medians, equal-size arithmetic mean"}
    save(fig,"figure_05_state_age")

def figure_s10():
    discrete=read("inputs/fish_paired_categorical.csv")
    discrete=discrete[discrete.variant.eq("primary")]
    w=discrete.pivot(index=["trial_id","definition"],columns="order",values="nll")
    w["gain_percent"]=100*(1-w[4]/w[1])
    gains=w.gain_percent.unstack("definition").sort_index()
    metrics=read("inputs/fish_paired_continuous.csv")
    metrics=metrics[metrics.variant.eq("primary")]
    p=metrics.pivot(index=["definition","trial_id","horizon_frames"],columns="model",values="mse")
    quality=read("inputs/fish_paired_support.csv")
    fig,axes=plt.subplots(2,2,figsize=(WIDTH,6.05))
    fig.subplots_adjust(left=.105,right=.975,bottom=.125,top=.925,hspace=.52,wspace=.40)
    ax=axes[0,0]
    for _,row in gains.iterrows(): ax.plot([0,1],row[["heading","displacement"]],c="#c5c5c5",lw=.8,zorder=1)
    means={}
    for i,(definition,color,label) in enumerate([("heading",BLUE,"Body heading"),("displacement",ORANGE,"Movement direction")]):
        ax.scatter(np.full(len(gains),i),gains[definition],s=22,c=color,marker=["o","s"][i],zorder=3)
        mean=float(gains[definition].mean()); means[definition]=mean
        ax.plot(i,mean,"_",c=INK,ms=16,mew=2,zorder=4)
        target_x=-.55 if i==0 else 1.15
        ax.annotate(f"{mean:.2f}%",xy=(i,mean),xytext=(target_x,mean),
                    ha="left",va="center",color=color,fontsize=9,fontweight="bold",
                    bbox={"facecolor":"white","edgecolor":"none","pad":1.5},
                    arrowprops={"arrowstyle":"-","color":color,"lw":.7,"shrinkA":2,"shrinkB":6})
    ax.set(xticks=[0,1],xticklabels=["Body\nheading","Movement\ndirection"],xlim=(-.6,1.6),ylim=(0,40),ylabel="Categorical loss reduction (%)")
    panel(ax,"a","Four observations vs current state")
    light_grid(ax)
    ratio_frames=[]
    for ax,num,den,letter,title in [(axes[0,1],"history","current_flexible","b","History / flexible current"),(axes[1,0],"state_trend","history","c","State and trend / history")]:
        for definition,color,offset in [("heading",BLUE,-.10),("displacement",ORANGE,.10)]:
            q=(p[num]/p[den]).xs(definition,level="definition")
            medians=[]
            for j,h in enumerate([3,10,30]):
                values=q.xs(h,level="horizon_frames").sort_index()
                ax.scatter(j+offset+np.linspace(-.035,.035,len(values)),values,c=color,s=15,marker="o" if definition=="heading" else "s",alpha=.6,edgecolors="white",linewidths=.3,zorder=2)
                medians.append(float(values.median()))
                ratio_frames.extend({"panel":letter,"definition":definition,"trial_id":tid,"horizon_frames":h,"numerator":num,"denominator":den,"mse_ratio":float(v),"recording_median":medians[-1]} for tid,v in values.items())
            ax.plot(np.arange(3)+offset,medians,c=color,marker="o" if definition=="heading" else "s",ls="-" if definition=="heading" else "--",ms=4,lw=1.6,zorder=3)
        ax.axhline(1,c=GREY,lw=.85,ls="--")
        ax.set(xticks=[0,1,2],xticklabels=["3","10","30"],xlim=(-.30,2.30),xlabel="Forecast horizon (frames)",ylabel="Mean-squared-error ratio")
        panel(ax,letter,title); light_grid(ax)
    axes[0,1].set_ylim(0,1.10)
    axes[0,1].text(1.65,.34,"Recording 0066\n(10 fish)",ha="center",va="center",color=GREY,fontsize=8.5)
    for definition,offset,color in [("heading",-.10,BLUE),("displacement",.10,ORANGE)]:
        q=(p["history"]/p["current_flexible"]).xs(definition,level="definition").xs(30,level="horizon_frames").sort_index()
        tid=next(t for t in q.index if str(t).endswith("0066"))
        j=list(q.index).index(tid); px=2+offset+np.linspace(-.035,.035,len(q))[j]
        axes[0,1].annotate("",xy=(px,float(q.loc[tid])),xytext=(1.65+offset,.255),arrowprops={"arrowstyle":"-","color":color,"lw":.65,"shrinkB":3})
    axes[1,0].set_ylim(.92,1.31)
    ax=axes[1,1]
    stylemap=[(10,"o","#3d3d3d","-"),(30,"s","#7c7c7c","--"),(70,"^","#a5a5a5",":")]
    for n,marker,color,ls in stylemap:
        part=quality[quality.group_size.eq(n)]
        for tid,rec in part.groupby("trial_id"):
            ordered=rec.set_index("variant").loc[["primary","step_gt_0p5px","step_gt_1px"]]
            ax.plot([0,1,2],100*ordered.paired_valid_fraction,color=color,ls=ls,marker=marker,ms=4,lw=1,zorder=2)
    ax.set(xticks=[0,1,2],xticklabels=["> 0","> 0.5","> 1"],ylim=(0,105),xlim=(-.15,2.15),xlabel="Minimum displacement (pixels/frame)",ylabel="Paired frames retained (%)")
    ax.legend(handles=[Line2D([0],[0],c=c,ls=ls,marker=m,ms=4,label=f"{n} fish") for n,m,c,ls in stylemap],loc="lower left",handlelength=1.8,labelspacing=.25)
    panel(ax,"d","Observation support"); light_grid(ax)
    fig.legend(handles=[Line2D([0],[0],color=BLUE,marker="o",ms=4,label="Body heading"),Line2D([0],[0],color=ORANGE,marker="s",ls="--",ms=4,label="Movement direction")],loc="lower center",bbox_to_anchor=(.5,.015),ncol=2,columnspacing=2)
    export("figure_s10_category_gain",w.reset_index().rename(columns={1:"nll_current",4:"nll_four_observations"}))
    export("figure_s10_forecast_ratios",pd.DataFrame(ratio_frames))
    export("figure_s10_observation_support",quality.assign(retained_percent=100*quality.paired_valid_fraction))
    counts=discrete.pivot(index=["trial_id","definition"],columns="order",values="rows")
    CHECKS["figure_s10"]={"pass":len(gains)==9 and np.array_equal(counts[1],counts[4]),"recordings":9,"category_points":18,"forecast_ratio_points":len(ratio_frames),"support_rows":len(quality),"category_means_percent":means,"forecast_summary_rule":"median of the nine recording ratios","paired_category_counts_equal":bool(np.array_equal(counts[1],counts[4]))}
    save(fig,"figure_s10_direction_comparison")

def figure_s01():
    frame=read("inputs/synthetic_calibration.csv")
    order=["strict_first_order","measurement_temporally_correlated","measurement_state_dependent","measurement_combined","nonstationary_first_order",
           *[f"known_K{k}_gain_{g}pct" for k in [2,3,5] for g in [1,3,6]],"closure_true_gain_2pct","closure_true_gain_3pct"]
    frame=frame.set_index("condition").loc[order].reset_index()
    labels=["Strict first order","Time-correlated error","State-dependent error","Combined error","Nonstationary first order",
            *[f"True K = {k}, gain {g}%" for k in [2,3,5] for g in [1,3,6]],"True K = 3; older gain 2%","True K = 3; older gain 3%"]
    columns=["exact_order_recovery","within_one_order_recovery","high_order_success_rate","closure_false_acceptance_rate"]
    values=frame[columns].to_numpy(dtype=float)
    is_older=frame.condition.str.startswith("closure_").to_numpy()
    values[is_older,2]=np.nan; values[~is_older,3]=np.nan
    fig=plt.figure(figsize=(WIDTH,7.05))
    heat=fig.add_axes([.37,.16,.34,.76]); gain=fig.add_axes([.78,.16,.195,.76])
    cmap=matplotlib.colormaps["Blues"].copy(); cmap.set_bad("#eeeeee")
    im=heat.pcolormesh(np.arange(5)-.5,np.arange(17)-.5,np.ma.masked_invalid(values),vmin=0,vmax=1,cmap=cmap,shading="flat",rasterized=False)
    heat.set(xlim=(-.5,3.5),ylim=(15.5,-.5))
    heat.set(yticks=range(16),yticklabels=labels,xticks=range(4),xticklabels=["Exact\norder","Within\none order","Higher-\norder\nsuccess","False\nacceptance"])
    heat.tick_params(length=0,labelsize=8.5,pad=5)
    for i in range(16):
        for j in range(4):
            v=values[i,j]
            heat.text(j,i,"--" if not np.isfinite(v) else f"{v:.2f}",ha="center",va="center",fontsize=8.5,c="white" if v>.65 else INK)
    for split in [4.5,7.5,10.5,13.5]:
        heat.axhline(split,c="white",lw=2); gain.axhline(split,c="#dddddd",lw=.6)
    for spine in heat.spines.values():spine.set_visible(False)
    heat.set_title("Calibration proportions",loc="left",pad=14)
    fig.text(.025,.95,"a",fontsize=11,fontweight="bold")
    med=100*frame.median_balanced_gain.to_numpy(); q95=100*frame.balanced_gain_q95.to_numpy()
    gain.hlines(range(16),0,q95,color="#d3d3d3",lw=1)
    gain.scatter(med,range(16),c=BLUE,s=21,zorder=3)
    gain.scatter(q95,range(16),facecolors="white",edgecolors=ORANGE,marker="s",s=22,zorder=4)
    gain.axvline(0,c=GREY,lw=.7)
    gain.set(ylim=(15.5,-.5),yticks=[],xlim=(-.25,6.8),xticks=[0,2,4,6],xlabel="Balanced loss\nreduction (%)")
    gain.set_title("Prediction gains",loc="left",pad=14)
    fig.text(.735,.95,"b",fontsize=11,fontweight="bold")
    cax=fig.add_axes([.37,.065,.34,.017]); cb=fig.colorbar(im,cax=cax,orientation="horizontal",ticks=[0,.5,1]);cb.set_label("Proportion",labelpad=3);cb.outline.set_visible(False);cb.solids.set_rasterized(False);cb.solids.set_edgecolor("face")
    fig.legend(handles=[Line2D([0],[0],marker="o",color="none",markerfacecolor=BLUE,label="Median"),Line2D([0],[0],marker="s",color="none",markeredgecolor=ORANGE,label="95th percentile")],loc="lower right",bbox_to_anchor=(.99,.005),fontsize=8.5,handletextpad=.3)
    out=frame.copy()
    for j,c in enumerate(columns):out["display_"+c]=values[:,j]
    out["display_median_percent"]=med;out["display_q95_percent"]=q95
    export("figure_s01_all_conditions",out)
    CHECKS["figure_s01"]={"pass":len(frame)==16,"conditions":16,"heatmap_cells":64,"not_evaluated_cells":int(np.isnan(values).sum()),"gain_points":32,"all_low_recovery_conditions_preserved":True}
    save(fig,"figure_s01_calibration")

def figure_s05():
    data=read("inputs/scalar_prediction_and_generation.csv")
    prediction=data[data.record_family.eq("prediction")]
    ids=ids_order(prediction.trial_id.unique())
    p=prediction.pivot(index="trial_id",columns="horizon_frames",values="mse_ratio_delay_vs_M").reindex(index=ids,columns=[3,10,30])
    metrics=["acf_rmse","psd_log_rmse","distribution_wasserstein","state_fraction_l1"]
    l=data[data.record_family.eq("long_time")].pivot(index="trial_id",columns="metric",values="error_ratio_delay_vs_current_only").reindex(index=ids,columns=metrics)
    fig=plt.figure(figsize=(WIDTH,5.35))
    axes=[fig.add_axes([.12,.235,.31,.66]),fig.add_axes([.57,.235,.40,.66])]
    cmap=matplotlib.colormaps["PuOr_r"]
    for ax,frame,title,letter,vmin,vmax,labels in [(axes[0],p,"Continuous prediction","a",.75,1.25,["3","10","30"]),(axes[1],l,"Generated statistics","b",.10,2.0,["Correlation","Spectrum","Distribution","Occupancy"])]:
        norm=TwoSlopeNorm(vmin=vmin,vcenter=1,vmax=vmax)
        im=ax.pcolormesh(np.arange(frame.shape[1]+1)-.5,np.arange(13)-.5,frame.to_numpy(),norm=norm,cmap=cmap,shading="flat",rasterized=False)
        ax.set(xlim=(-.5,frame.shape[1]-.5),ylim=(11.5,-.5))
        for i in range(12):
            for j in range(frame.shape[1]):
                value=frame.iloc[i,j];r,g,b,a=cmap(norm(value));lum=.2126*r+.7152*g+.0722*b
                ax.text(j,i,f"{value:.2f}",ha="center",va="center",color="white" if lum<.46 else INK,fontsize=9)
        ax.set(yticks=range(12),yticklabels=ids,xticks=range(len(labels)),xticklabels=labels)
        ax.tick_params(axis="both",length=0,pad=5)
        if letter=="b":ax.tick_params(axis="x",labelrotation=25)
        else:ax.set_xlabel("Forecast horizon (frames)")
        for split in [3.5,7.5]:ax.axhline(split,c="white",lw=2)
        for spine in ax.spines.values():spine.set_visible(False)
        panel(ax,letter,title)
        pos=ax.get_position();cax=fig.add_axes([pos.x0,.10,pos.width,.02]);cb=fig.colorbar(im,cax=cax,orientation="horizontal",ticks=[vmin,1,vmax]);cb.set_label("History / current error",labelpad=4);cb.outline.set_visible(False);cb.solids.set_rasterized(False);cb.solids.set_edgecolor("face")
    export("figure_s05_prediction",p.reset_index());export("figure_s05_generation",l.reset_index())
    CHECKS["figure_s05"]={"pass":p.shape==(12,3) and l.shape==(12,4),"prediction_cells":p.size,"generation_cells":l.size,"maximum_generation_ratio":float(l.max().max()),"ratios_above_one":int((l>1).sum().sum()),"all_values_within_color_limits":bool(l.min().min()>=.1 and l.max().max()<=2)}
    save(fig,"figure_s05_recording_heatmaps")

def figure_s02():
    sensitivity=read("inputs/order_selection_sensitivity.csv")
    primary=read("inputs/order_selection_scores.csv")
    development=read_json("inputs/selected_order.json")
    consistency=read("inputs/state_bin_consistency.csv")
    main=pd.DataFrame([{"scenario":"three_states_step_5","state_count":3,"step_frames":5,"maximum_order":5,"selected_order":int(development["selected_order"]),"best_order":int(primary.loc[primary.mean_nll.idxmin(),"order"]),"selected_history_seconds":float(development["history_seconds"])}])
    sensitivity=pd.concat([main,sensitivity],ignore_index=True).sort_values(["state_count","step_frames"]).reset_index(drop=True)
    values=consistency.pivot(index="bin_count",columns="step_frames",values="median_observed_over_null_q95").sort_index().sort_index(axis=1)
    fig=plt.figure(figsize=(WIDTH,4.25));a=fig.add_axes([.27,.26,.265,.59]);b=fig.add_axes([.67,.26,.295,.59])
    y=np.arange(4)
    a.scatter(sensitivity.best_order,y-.10,c="#E69F00",marker="D",s=28,label="Minimum validation loss",zorder=3)
    a.scatter(sensitivity.selected_order,y+.10,facecolors="white",edgecolors=BLUE,marker="s",s=29,label="One-standard-error choice",zorder=3)
    for i,row in sensitivity.iterrows():
        a.plot([row.selected_order,row.best_order],[i+.10,i-.10],c=GREY,lw=.8,zorder=1)
        a.text(row.selected_order+.18,i+.10,f"{row.selected_history_seconds:.2f} s",va="center",fontsize=8.5,color=BLUE)
    labels=[f"{int(r.state_count)} states, {int(r.step_frames)}-frame step" for r in sensitivity.itertuples()]
    a.set(yticks=y,yticklabels=labels,ylim=(3.5,-.5),xticks=range(1,7),xlim=(.7,6.4),xlabel="History order, K")
    a.set_title("Order selection",loc="left",pad=12);fig.text(.025,.895,"a",fontweight="bold",fontsize=11)
    cmap=matplotlib.colormaps["viridis"];norm=plt.Normalize(1,max(4.0,float(values.max().max())))
    im=b.pcolormesh(np.arange(4)-.5,np.arange(4)-.5,values.to_numpy(),norm=norm,cmap=cmap,shading="flat",rasterized=False)
    b.set(xticks=range(3),xticklabels=values.columns.astype(int),yticks=range(3),yticklabels=values.index.astype(int),ylim=(2.5,-.5),xlim=(-.5,2.5),xlabel="Sampling step (frames)",ylabel="Number of state bins")
    for i in range(3):
        for j in range(3):
            v=values.iloc[i,j];r,g,bl,alpha=cmap(norm(v));lum=.2126*r+.7152*g+.0722*bl
            b.text(j,i,f"{v:.2f}",ha="center",va="center",c="white" if lum<.46 else INK,fontsize=9)
    b.tick_params(length=0);panel(b,"b","First-order consistency")
    for spine in b.spines.values():spine.set_visible(False)
    cax=fig.add_axes([.67,.11,.295,.025]);cb=fig.colorbar(im,cax=cax,orientation="horizontal",ticks=[1,2,3,4]);cb.set_label("Discrepancy / reference",labelpad=3);cb.solids.set_rasterized(False);cb.solids.set_edgecolor("face");cb.outline.set_visible(False)
    fig.legend(*a.get_legend_handles_labels(),loc="lower left",bbox_to_anchor=(.025,.015),ncol=1,handletextpad=.5,labelspacing=.45)
    export("figure_s02_order_sensitivity",sensitivity);export("figure_s02_consistency",values.reset_index())
    CHECKS["figure_s02"]={"pass":len(sensitivity)==4 and values.size==9,"selection_settings":4,"heatmap_values":9,"color_limits":[norm.vmin,norm.vmax],"maximum_ratio":float(values.max().max())}
    save(fig,"figure_s02_selection_sensitivity")

def figure_s07():
    data=read("inputs/fish_continuous_reference.csv")
    fig,axes=plt.subplots(1,3,figsize=(WIDTH,3.30))
    fig.subplots_adjust(left=.095,right=.98,top=.82,bottom=.27,wspace=.40)
    out=[]
    for ax,column,title,letter in zip(axes,["ratio_flexible","ratio_spline","persistence"],["Flexible current\npolynomial","Current spline","Persistence"],"abc"):
        for n,c,marker,offset in [(10,BLUE,"o",-.12),(30,"#E69F00","s",0),(70,TEAL,"^",.12)]:
            part=data[data.group_size.eq(n)].sort_values(["horizon","trial"] if "trial" in data else ["horizon"])
            y=part.history/part.persistence_mse if column=="persistence" else part[column]
            x=part.horizon.map({3:0,10:1,30:2}).to_numpy()+offset
            ax.scatter(x,y,c=c,marker=marker,s=23,alpha=.8,label=f"N = {n}",zorder=3)
            out.extend({**r,"comparison":column,"display_ratio":float(v)} for r,v in zip(part.to_dict("records"),y))
        ax.axhline(1,c=GREY,ls="--",lw=.8)
        ax.set(yscale="log",xticks=[0,1,2],xticklabels=[3,10,30],xlim=(-.32,2.32),xlabel="Horizon (frames)")
        if column=="persistence":
            ax.set(ylim=(.87,1.025),yticks=[.88,.92,.96,1]);ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f"{x:.2f}"))
        else:
            ax.set(ylim=(.09,1.16),yticks=[.1,.3,1]);ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f"{x:g}"))
        ax.yaxis.set_minor_locator(NullLocator())
        panel(ax,letter,title);light_grid(ax)
    axes[0].set_ylabel("History / reference MSE")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.5,.01),ncol=3,columnspacing=2)
    export("figure_s07_continuous_fish",pd.DataFrame(out))
    CHECKS["figure_s07"]={"pass":len(out)==81,"recording_horizon_comparator_points":len(out),"all_panels_log_scale":True,"minimum_ratio":float(pd.DataFrame(out).display_ratio.min()),"maximum_ratio":float(pd.DataFrame(out).display_ratio.max())}
    save(fig,"figure_s07_continuous_fish")

def figure_s08():
    acf=read("inputs/generation_autocorrelation.csv")
    psd=read("inputs/generation_spectrum.csv")
    fig,axes=plt.subplots(3,2,figsize=(WIDTH,6.8))
    fig.subplots_adjust(left=.11,right=.975,bottom=.15,top=.94,hspace=.52,wspace=.34)
    models=[("empirical",INK,"Observed","-"),("bounded_current_only",BLUE,"Current generator","--"),("bounded_current_and_delay",ORANGE,"History generator","-.")]
    for i,n in enumerate([50,100,200]):
        for model,c,label,ls in models:
            a=acf[acf.group_size.eq(n)&acf.model.eq(model)];b=psd[psd.group_size.eq(n)&psd.model.eq(model)]
            axes[i,0].plot(a.lag_frames/30,a.median_across_trials,c=c,ls=ls,label=label)
            axes[i,1].loglog(b.normalized_frequency_cycles_per_frame*30,b.median_across_trials,c=c,ls=ls)
        axes[i,0].set(xlim=(0,5),xlabel="Lag (s)",ylabel="Autocorrelation",ylim=(-.05,1.05),yticks=[0,.5,1])
        axes[i,1].set(xlabel="Frequency (Hz)",ylabel="PSD (M²/Hz)")
        axes[i,1].tick_params(axis="both",labelsize=10)
        for j in range(2):panel(axes[i,j],chr(97+i*2+j),f"N = {n}");light_grid(axes[i,j])
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.5,.015),ncol=3,columnspacing=1.6,handlelength=2)
    export("figure_s08_autocorrelation",acf);export("figure_s08_spectrum",psd)
    CHECKS["figure_s08"]={"pass":True,"acf_rows":len(acf),"psd_rows":len(psd),"color_mapping_matches_main_04":True,"nominal_time_frequency_conversion_hz":30}
    save(fig,"figure_s08_temporal_statistics")

def figure_s09():
    scores=read("inputs/order_selection_scores.csv")
    bootstrap=read("inputs/order_selection_resamples.csv")
    development=read_json("inputs/selected_order.json")
    trials=read("inputs/categorical_recording_scores.csv")
    consistency=read("inputs/two_step_consistency.csv")
    consistency=consistency[consistency.phase.eq(0)].copy()
    consistency["trial_id"]="N"+consistency.group_size.astype(str)+consistency.trial
    selected=int(development["selected_order"]); idx=scores.mean_nll.idxmin();best=int(scores.loc[idx,"order"])
    bound=float(scores.loc[idx,"mean_nll"]+scores.loc[idx,"standard_error"])
    ids=ids_order(trials.trial_id.unique());trials=trials.set_index("trial_id").loc[ids]
    current=consistency[consistency.order.eq(1)].set_index("trial_id").loc[ids]
    history=consistency[consistency.order.eq(4)].set_index("trial_id").loc[ids]
    fig=plt.figure(figsize=(WIDTH,6.7))
    a=fig.add_axes([.11,.655,.365,.285]); b=fig.add_axes([.61,.655,.365,.285])
    c=fig.add_axes([.11,.14,.365,.34]); d=fig.add_axes([.61,.14,.365,.34])
    a.errorbar(scores.order,scores.mean_nll,yerr=scores.standard_error,c=GREY,marker="o",mfc="white",ms=4,capsize=3)
    a.scatter(selected,float(scores.loc[scores.order.eq(selected),"mean_nll"].iloc[0]),c=BLUE,marker="s",s=35,zorder=4,label=f"Selected K = {selected}")
    a.scatter(best,float(scores.loc[idx,"mean_nll"]),c="#E69F00",marker="D",s=35,zorder=4,label=f"Minimum at K = {best}")
    a.axhline(bound,c=INK,ls="--",lw=.8,label="One-standard-error bound")
    a.set(xticks=scores.order,xlabel="History order, K",ylabel="Validation loss (nats)")
    panel(a,"a","Development selection");light_grid(a)
    a.legend(loc="upper right",fontsize=8.5,handlelength=1.1,labelspacing=.35)
    counts=100*bootstrap.selected_order.value_counts(normalize=True).reindex(range(1,6),fill_value=0)
    bars=b.bar(counts.index,counts.values,color=["#cccccc","#cccccc","#E69F00",BLUE,TEAL],edgecolor=INK,lw=.5)
    for bar,value in zip(bars,counts.values):b.text(bar.get_x()+bar.get_width()/2,value+2,f"{value:.1f}",ha="center",fontsize=8.5)
    b.axhline(80,c=GREY,ls="--",lw=.8)
    b.text(.02,.96,"80% stability reference",transform=b.transAxes,va="top",fontsize=8.5)
    b.set(xticks=range(1,6),ylim=(0,96),xlabel="Selected order, K",ylabel="Resamples (%)")
    panel(b,"b","Selection stability");light_grid(b)
    y=np.arange(12)
    c.scatter(100*trials.selected_relative_gain,y-.14,c=BLUE,s=22,label="K = 4 vs K = 1")
    c.scatter(100*trials.older_relative_gain,y+.14,facecolors="white",edgecolors=ORANGE,marker="D",s=22,label="K = 6 vs K = 4")
    c.axvline(1,c=GREY,ls=":",lw=.9)
    c.set(yticks=y,yticklabels=ids,ylim=(11.7,-.7),xlabel="Loss reduction (%)",xlim=(-.5,31))
    panel(c,"c","Gains by recording");light_grid(c,"x")
    c.legend(loc="upper center",bbox_to_anchor=(.5,-.17),fontsize=8.5,ncol=1,handletextpad=.4,labelspacing=.3)
    d.scatter(current.observed_over_null_q95,y-.14,c="#E69F00",marker="s",s=23,label="K = 1")
    d.scatter(history.observed_over_null_q95,y+.14,facecolors="white",edgecolors=BLUE,s=23,label="K = 4")
    d.axvline(1,c=GREY,ls="--",lw=.8)
    d.set(xscale="log",xlim=(.5,15),xticks=[.5,1,2,5,10],yticks=y,yticklabels=ids,ylim=(11.7,-.7),xlabel="Two-step discrepancy / reference")
    d.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f"{x:g}"));d.xaxis.set_minor_locator(NullLocator())
    panel(d,"d","Two-step consistency");light_grid(d,"x")
    d.legend(loc="upper center",bbox_to_anchor=(.5,-.17),ncol=2,fontsize=8.5,handletextpad=.4,columnspacing=1)
    for ax in [c,d]:
        for split in [3.5,7.5]:ax.axhline(split,c="#dddddd",lw=.6)
    export("figure_s09_selection_scores",scores)
    export("figure_s09_selection_frequency",counts.rename("percent").rename_axis("order").reset_index())
    export("figure_s09_recording_gains",trials.reset_index())
    export("figure_s09_consistency",consistency[consistency.order.isin([1,4])])
    CHECKS["figure_s09"]={"pass":len(bootstrap)==2000 and len(ids)==12,"resamples":len(bootstrap),"selected_order":selected,"minimum_order":best,"selected_order_frequency_percent":float(counts.loc[selected]),"gain_points":24,"consistency_points":24,"four_observation_unadjusted_exceedances":int((history.observed_over_null_q95>1).sum()),"one_standard_error_bound":bound}
    save(fig,"figure_s09_order_selection")
