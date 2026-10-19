"""Scalar prediction and observation-control figures."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter
from . import panels as p

WIDTH=180/25.4
BLUE,ORANGE,TEAL,INK,GREY=p.BLUE,p.ORANGE,p.TEAL,p.INK,p.GREY

def figure_03():
    matched=p.read("inputs/capacity_matched_scores.csv")
    predictive=p.read("inputs/scalar_capacity_comparisons.csv")
    probabilistic=p.read("inputs/continuous_probability_scores.csv")
    external=p.read("inputs/scalar_generated_statistics.csv")
    fig=plt.figure(figsize=(WIDTH,5.85))
    a=fig.add_axes([.11,.65,.86,.27]);b=fig.add_axes([.11,.15,.20,.28]);c=fig.add_axes([.425,.15,.20,.28]);d=fig.add_axes([.79,.15,.18,.28])
    comparison_rows=[]
    comparisons=[("6 vs 6 coefficients",matched,"group_balanced_ratio",BLUE,"o","-"),("Flexible current polynomial",predictive[predictive.comparison.eq("history_vs_flexible_current")],"group_balanced_point","#E69F00","s","--"),("Equal-parameter neural nets",predictive[predictive.comparison.eq("neural_history_vs_neural_current")],"group_balanced_point",TEAL,"^","-.")]
    for label,frame,col,color,marker,ls in comparisons:
        f=frame.sort_values("horizon_frames");value=f[col].to_numpy();lo=f.bootstrap_interval_lower.to_numpy();hi=f.bootstrap_interval_upper.to_numpy()
        a.errorbar(f.horizon_frames,value,yerr=np.vstack([value-lo,hi-value]),c=color,marker=marker,ls=ls,mfc="white",ms=5,capsize=3,label=label)
        comparison_rows.extend({"comparison":label,"horizon_frames":h,"ratio":v,"low":l,"high":u} for h,v,l,u in zip(f.horizon_frames,value,lo,hi))
    a.axhline(1,c=GREY,ls="--",lw=.8);a.set(xticks=[3,10,30],ylim=(.82,1.01),xlabel="Forecast horizon (frames)",ylabel="History / current MSE")
    p.panel(a,"a","Additional scalar prediction comparisons")
    fig.legend(*a.get_legend_handles_labels(),loc="center",bbox_to_anchor=(.535,.525),ncol=3,handlelength=1.7,columnspacing=1,handletextpad=.45)
    nll=probabilistic[probabilistic.value_column.eq("difference_nll_history_minus_current")].sort_values("horizon_frames")
    value=-nll.group_balanced_point.to_numpy();lo=-nll.bootstrap_interval_upper.to_numpy();hi=-nll.bootstrap_interval_lower.to_numpy()
    b.errorbar(nll.horizon_frames,value,yerr=np.vstack([value-lo,hi-value]),c="#AA6FA5",marker="D",mfc="white",capsize=3)
    b.axhline(0,c=GREY,ls="--",lw=.8);b.set(xticks=[1,5],ylim=(0,.10),xlabel="Horizon (frames)",ylabel="Log-density loss\nreduction (nats)");p.panel(b,"b","Log score")
    crps=probabilistic[probabilistic.value_column.eq("ratio_crps_history_vs_current")].sort_values("horizon_frames")
    v=100*(1-crps.group_balanced_point.to_numpy());l=100*(1-crps.bootstrap_interval_upper.to_numpy());u=100*(1-crps.bootstrap_interval_lower.to_numpy())
    c.errorbar(crps.horizon_frames,v,yerr=np.vstack([v-l,u-v]),c="#56B4E9",marker="s",mfc="white",capsize=3)
    c.axhline(0,c=GREY,ls="--",lw=.8);c.set(xticks=[1,5],ylim=(0,10),xlabel="Horizon (frames)",ylabel="CRPS reduction (%)");p.panel(c,"c","Probability score")
    metrics=["acf_rmse","psd_log_rmse","distribution_wasserstein","state_fraction_l1"]
    long=external[external.analysis_set.eq("all_12")&external.evidence_family.eq("long_time")].set_index("metric").loc[metrics]
    value=long.relative_ratio_value.to_numpy();lo=long.interval_lower_value.to_numpy();hi=long.interval_upper_value.to_numpy()
    for j,(v,l,u) in enumerate(zip(value,lo,hi)):
        d.errorbar(v,3-j,xerr=[[v-l],[u-v]],color=BLUE if v<1 else ORANGE,marker=["o","s","D","^"][j],mfc="white",capsize=3)
    d.axvline(1,c=GREY,ls="--",lw=.8);d.set(yticks=[3,2,1,0],yticklabels=["Correlation","Spectrum","Distribution","Occupancy"],xlim=(.2,1.45),xticks=[.5,1],xlabel="History / current\nerror ratio");p.panel(d,"d","Generated errors")
    p.export("figure_03_capacity_ratios",pd.DataFrame(comparison_rows));p.export("figure_03_probability_scores",probabilistic);p.export("figure_03_generation_ratios",long.reset_index())
    p.CHECKS["figure_03"]={"pass":len(comparison_rows)==9,"capacity_comparisons":9,"probability_score_points":4,"generated_metrics":4}
    p.save(fig,"figure_03_scalar_comparisons")

def figure_06():
    sims=p.read("inputs/observation_control_repetitions.csv")
    matching=p.read("inputs/observation_control_matching.csv")
    fig=plt.figure(figsize=(WIDTH,5.65));a=fig.add_axes([.105,.65,.865,.27])
    scenarios=["no_measurement_error","independent_error","correlated_error","noisy_multilag"]
    a.boxplot([100*sims[sims.scenario.eq(s)].balanced_gain for s in scenarios],tick_labels=["Error-free","Independent error","Correlated error","Slow latent + noise"],showfliers=False,widths=.55)
    a.axhline(21.121138943955664,c=ORANGE,ls="--",label="Observed: 21.12%")
    a.set(ylabel="History gain (%)",ylim=(-1,24));p.panel(a,"a","History under observation controls");a.legend(loc="upper left",bbox_to_anchor=(.01,.88))
    curves=[]
    model_specs=[("one_step","observed","Observed",INK,"-"),("one_step","synthetic","Error-free",BLUE,"--"),("noisy_multilag","synthetic","Slow latent + noise",TEAL,"-.")]
    handles=[]
    for i,n in enumerate([50,100,200]):
        ax=fig.add_axes([.105+i*.305,.18,.255,.30]);part=matching[matching.group_size.eq(n)].set_index("process");lags=[1,5,15,30,150]
        for typ,prefix,label,color,ls in model_specs:
            values=[part.loc[typ,f"{prefix}_acf_{k}"] for k in lags]
            line,=ax.plot(np.array(lags)/30,values,c=color,ls=ls,marker="o",ms=3,label=label)
            if i==0:handles.append(line)
            curves.extend({"group_size":n,"model":label,"lag_frames":lag,"acf":v} for lag,v in zip(lags,values))
        ax.set(xscale="log",xlabel="Lag (s)",ylim=(-.05,1.02),yticks=[0,.5,1]);ax.set_xticks([.1,1]);ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x:g}"));ax.xaxis.set_minor_formatter(NullFormatter())
        if i==0:ax.set_ylabel("Autocorrelation")
        p.panel(ax,chr(98+i),f"N = {n}")
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.015),ncol=3,columnspacing=1.5,handlelength=2)
    from matplotlib.cbook import boxplot_stats
    boxes=[]
    for s in scenarios:
        data=100*sims[sims.scenario.eq(s)].balanced_gain;st=boxplot_stats(data,whis=1.5)[0]
        boxes.append({"scenario":s,"simulations":len(data),"q1":st["q1"],"median":st["med"],"q3":st["q3"],"low":st["whislo"],"high":st["whishi"],"undisplayed_outliers":len(st["fliers"])})
    p.export("figure_06_box_values",pd.DataFrame(boxes));p.export("figure_06_acf_values",pd.DataFrame(curves))
    p.CHECKS["figure_06"]={"pass":len(sims)==800,"simulations":len(sims),"acf_values":len(curves),"box_whisker_rule":"1.5 IQR","fliers_displayed":False}
    p.save(fig,"figure_06_observation_controls")

def figure_s03():
    lag=p.read("inputs/delay_sensitivity.csv")
    regions=p.read("inputs/circulation_band_sensitivity.csv")
    fig=plt.figure(figsize=(WIDTH,4.6));a=fig.add_axes([.115,.23,.46,.50]);b=fig.add_axes([.735,.23,.24,.50])
    for h,color,marker,ls in [(3,BLUE,"o","-"),(10,"#E69F00","s","--"),(30,TEAL,"^","-.")]:
        r=lag[lag.horizon_frames.eq(h)].sort_values("delay_frames")
        a.fill_between(r.delay_frames,r.bootstrap_interval_lower,r.bootstrap_interval_upper,color=color,alpha=.12,lw=0)
        a.plot(r.delay_frames,r.group_balanced_ratio,c=color,marker=marker,ls=ls,markevery=3,mfc="white",ms=4,label=f"{h}-frame forecast")
    a.axhline(1,c=GREY,ls="--",lw=.8);a.axvline(5,c=GREY,ls=":",lw=.9,label="Delay = 5 frames")
    a.set(xlim=(1,30),ylim=(.837,1.08),xlabel="Delay (frames)",ylabel="History / current MSE")
    a.legend(loc="upper center",ncol=2,handlelength=1.4,columnspacing=.7,handletextpad=.4)
    secondary=a.secondary_xaxis("top",functions=(lambda x:x/30,lambda s:s*30));secondary.set_xlabel("Delay (s)")
    fig.text(.115,.94,"(a) Delay sensitivity",fontsize=11)
    for region,color,marker,label,offset in [("low_circulation_band",BLUE,"o","Low |M| < 0.3",-.13),("higher_circulation_band","#E69F00","s","High |M| >= 0.3",.13)]:
        r=regions[regions.region.eq(region)].set_index("horizon_frames").loc[[3,10,30]];v=r.group_balanced_ratio.to_numpy()
        b.errorbar(np.arange(3)+offset,v,yerr=np.vstack([v-r.bootstrap_interval_lower,r.bootstrap_interval_upper-v]),c=color,marker=marker,ls="none",mfc="white",capsize=3,label=label)
    b.axhline(1,c=GREY,ls="--",lw=.8);b.set(xticks=range(3),xticklabels=[3,10,30],ylim=(.82,1.06),xlabel="Horizon (frames)",ylabel="History / current MSE")
    b.legend(loc="upper center",fontsize=10,handletextpad=.3)
    fig.text(.735,.94,"(b) Circulation bands",fontsize=11)
    p.export("figure_s03_delay_values",lag);p.export("figure_s03_region_values",regions)
    p.CHECKS["figure_s03"]={"pass":len(lag)==90,"delay_points":len(lag),"region_points":len(regions)}
    p.save(fig,"figure_s03_delay_sensitivity")

def figure_s04():
    alternatives=p.read("inputs/alternative_processes.csv")
    reversal=p.read("inputs/reversal_noise_controls.csv")
    alternatives=alternatives.iloc[::-1].reset_index(drop=True)
    names={"measurement_temporally_correlated":"Time-correlated error","measurement_state_dependent":"State-dependent error","measurement_combined":"Combined error","nonstationary_first_order":"Nonstationary first order"}
    fig=plt.figure(figsize=(WIDTH,4.40));a=fig.add_axes([.285,.28,.29,.55]);b=fig.add_axes([.72,.28,.25,.55])
    q95=100*alternatives.simulated_gain_q95;real=100*alternatives.real_balanced_gain.iloc[0]
    a.hlines(range(4),0,q95,colors="#cccccc",lw=1.5)
    a.scatter(q95,range(4),facecolors="white",edgecolors=BLUE,marker="s",s=28,label="Simulated 95th percentile")
    a.axvline(real,c=ORANGE,ls="--",label=f"Observed: {real:.1f}%")
    for i,v in enumerate(q95):a.text(v+.5,i,f"{v:.2f}%",va="center")
    a.set(yticks=range(4),yticklabels=[names[x] for x in alternatives.condition],xlim=(0,24),xlabel="Balanced loss reduction (%)")
    p.panel(a,"a","Alternative processes")
    a.text(.45,.78,"0/500 joint exceedances\nper condition",transform=a.transAxes,ha="center",va="top")
    a.legend(loc="upper left",bbox_to_anchor=(-.90,-.25),fontsize=10,handlelength=1.5)
    groups={50:(BLUE,"full"),100:("#E69F00","none"),200:(TEAL,"left")};markers={3:"o",5:"s",8:"^"}
    for (n,delay),part in reversal.groupby(["group_size","delay_frames"]):
        color,fill=groups[n]
        b.plot(part.conservative_maximum_q95,part.observed_D_reversal_strength,ls="none",marker=markers[delay],fillstyle=fill,mfc=color,mfcalt="white",mec=INK if fill!="none" else color,ms=5,mew=.8)
    b.plot([0,.46],[0,.46],c=INK,ls="--",lw=.8)
    b.set(xlim=(0,.46),ylim=(0,.46),xlabel="Largest simulated\n95th percentile",ylabel="Observed reversal strength")
    p.panel(b,"b","Trend reversal")
    b.text(.97,.16,"36/36 above\nequality",transform=b.transAxes,ha="right",va="bottom")
    handles=[Line2D([0],[0],ls="none",marker="o",fillstyle=f,mfc=c,mfcalt="white",mec=INK if f!="none" else c,label=f"N = {n}") for n,(c,f) in groups.items()]
    delays=[Line2D([0],[0],ls="none",marker=m,mfc="white",mec=INK,label=f"Delay {d}") for d,m in markers.items()]
    handles=[handle for pair in zip(handles,delays) for handle in pair]
    fig.legend(handles=handles,ncol=3,loc="lower right",bbox_to_anchor=(.99,.015),handletextpad=.4,columnspacing=.8)
    p.export("figure_s04_alternatives",alternatives);p.export("figure_s04_reversals",reversal)
    p.CHECKS["figure_s04"]={"pass":len(reversal)==36,"alternative_processes":len(alternatives),"recording_delay_horizon_points":len(reversal),"group_encoding":"colour and fill; 50 full, 100 open, 200 half-filled","delay_encoding":"3 circle, 5 square, 8 triangle"}
    p.save(fig,"figure_s04_alternative_controls")

def figure_s06():
    frame=p.read("inputs/single_fish_recording_scores.csv");r=frame.iloc[0]
    current=np.array([r.nll_order_1,r.brier_order_1,r.rps_order_1]);history=np.array([r.nll_selected_order,r.brier_selected_order,r.rps_selected_order]);gain=100*(1-history/current)
    fig=plt.figure(figsize=(WIDTH,3.3));a=fig.add_axes([.105,.27,.375,.53]);b=fig.add_axes([.62,.27,.355,.53]);x=np.arange(3);labels=["NLL","Brier score","Ranked\nprobability score"]
    a.bar(x-.17,current,.34,color="#E69F00",edgecolor=INK,lw=.5,hatch="//",label="K = 1")
    a.bar(x+.17,history,.34,color=BLUE,edgecolor=INK,lw=.5,hatch="..",label="K = 4")
    a.set(xticks=x,xticklabels=labels,ylabel="Score");a.legend(loc="upper right");p.panel(a,"a","151-fish recording")
    bars=b.bar(x,gain,color=[BLUE,"#E69F00",TEAL],edgecolor=INK,lw=.5,hatch=["//","..","xx"])
    for bar,v in zip(bars,gain):b.text(bar.get_x()+bar.get_width()/2,v+.08,f"{v:.2f}%",ha="center")
    b.set(xticks=x,xticklabels=labels,ylabel="Relative improvement (%)",ylim=(0,4.5))
    b.text(.98,.70,"Single recording\n(descriptive)",transform=b.transAxes,ha="right",va="top")
    p.panel(b,"b","Time-split prediction")
    p.export("figure_s06_scores",pd.DataFrame({"metric":["nll","brier","rps"],"current":current,"history":history,"gain_percent":gain}))
    p.CHECKS["figure_s06"]={"pass":True,"recordings":1,"score_values":6,"gain_values":3}
    p.save(fig,"figure_s06_single_recording")
