"""Original editable overview of collective-rotation analyses.

All animal arrangements, trajectories, rotation values, density and category
examples are constructed schematics, not measurements or fitted results.
The animal shapes and illustrative curves are deterministic schematics.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as PlotPath
from matplotlib.transforms import Affine2D


HERE = Path(__file__).resolve().parent
SPEC = {}
W, H = 180.0, 136.0
INK = "#233A46"
MUTED = "#536671"
LIGHT = "#C8D5DD"
FAINT = "#E8EFF3"
BLUE = "#2677A6"
PALE_BLUE = "#EEF5F9"
ORANGE = "#BF713D"
PALE_ORANGE = "#FBF3EB"
TEAL = "#337D7A"
PALE_TEAL = "#EFF6F4"
WHITE = "#FFFFFF"
TEXTS = []
CONNECTORS = []


def configure():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9.5,
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "pdf.use14corefonts": True, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.unicode_minus": False, "figure.facecolor": "white",
        "savefig.facecolor": "white", "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round", "svg.hashsalt": "rotation-framework",
    })


def text(ax, x, y, value, size=9.5, weight="normal", color=INK, ha="left", **kw):
    """Native editable text; small math labels use standard-font text runs.

    Using native runs for these Latin-only expressions avoids embedding a
    separate nonstandard mathematics font in the PDF. Subscripts retain their
    normal smaller size and are recorded separately from the primary labels.
    """
    size=max(size,SPEC["minimum_source_label_font_pt"])
    math_runs={
        r"$M(t)$":[("M(t)",1,0,"italic")],
        r"$(M, P, R, A)_t$":[("(M, P, R, A)",1,0,"italic"),("t",.7,1.0,"italic")],
        r"$M_{t-5}$":[("M",1,0,"italic"),("t-5",.7,1.0,"italic")],
        r"$(a_t,b_t)$":[("(",1,0,"normal"),("a",1,0,"italic"),("t",.7,1.0,"italic"),
                         (", ",1,0,"normal"),("b",1,0,"italic"),("t",.7,1.0,"italic"),(")",1,0,"normal")],
        r"$t-15$":[("t - 15",1,0,"italic")],
        r"$t-10$":[("t - 10",1,0,"italic")],
        r"$t-5$":[("t - 5",1,0,"italic")],
        r"$t$":[("t",1,0,"italic")],
        r"$t+h$":[("t + h",1,0,"italic")],
        r"$S_{t+5}$":[("S",1,0,"italic"),("t+5",.7,1.0,"italic")],
        r"$N=10,\ 30,\ 70$":[("N",1,0,"italic"),(" = 10, 30, 70",1,0,"normal")],
    }
    if value.startswith("$"):
        runs=math_runs[value]
    else:
        runs=[(value,1,0,"normal")]
    renderer=ax.figure.canvas.get_renderer()
    pixels_per_unit=ax.transData.transform((1,0))[0]-ax.transData.transform((0,0))[0]
    widths=[]
    for s,factor,dy,style in runs:
        prop=font_manager.FontProperties(family="sans-serif",style=style,weight=weight,size=size*factor)
        widths.append(renderer.get_text_width_height_descent(s,prop,ismath=False)[0]/pixels_per_unit)
    origin=x-(sum(widths)/2 if ha=="center" else sum(widths) if ha=="right" else 0)
    group=TEXTS[-1]._label_group+1 if TEXTS else 0
    first=None
    for (s,factor,dy,style),width in zip(runs,widths):
        obj=ax.text(origin,y+dy,s,fontsize=size*factor,color=color,weight=weight,
                    fontstyle=style,ha="left",va="center",linespacing=1.2,zorder=20,**kw)
        obj.set_gid(f"text-{len(TEXTS):03d}")
        obj._label_group=group
        obj._primary_label=factor==1
        obj._full_label=value
        TEXTS.append(obj)
        if first is None:
            first=obj
        origin+=width
    return first


def line(ax, x, y, color=LIGHT, lw=.8, zorder=2, **kw):
    return ax.plot(x, y, color=color, lw=lw, zorder=zorder, **kw)[0]


def path(ax, verts, codes, color=INK, lw=.8, fill="none", zorder=5, **kw):
    obj = PathPatch(PlotPath(verts, codes), fc=fill, ec=color, lw=lw,
                    joinstyle="round", capstyle="round", zorder=zorder, **kw)
    ax.add_patch(obj)
    return obj


def arrow(ax, start, end, color=MUTED, lw=.85, rad=0, ms=6.5,
          ls="-", check=False, **kw):
    obj = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=ms,
                          connectionstyle=f"arc3,rad={rad}", color=color,
                          lw=lw, linestyle=ls, shrinkA=0, shrinkB=0, zorder=7, **kw)
    ax.add_patch(obj)
    if check:
        CONNECTORS.append(obj)
    return obj


def flow(ax, pts, color=MUTED, lw=1, dashed=False):
    """Cubic connector with round joins and one meaningful arrowhead."""
    obj = FancyArrowPatch(path=PlotPath(pts, [PlotPath.MOVETO] + [PlotPath.CURVE4]*(len(pts)-1)),
                          arrowstyle="-|>", mutation_scale=7.0, lw=lw,
                          edgecolor=color, facecolor=color,
                          linestyle=(0, (2.5, 2.5)) if dashed else "-",
                          zorder=4)
    ax.add_patch(obj)
    CONNECTORS.append(obj)
    return obj


def panel_heading(ax, x, y, letter, title):
    text(ax, x, y, f"({letter})", size=10, weight="bold")
    text(ax, x+8.5, y, title, size=9.5, weight="bold")


def beetle(ax, x, y, angle=0, scale=1, color=INK, identifier="beetle"):
    """Original dorsal whirligig silhouette: elytra, head and paddle legs."""
    tr = Affine2D().scale(scale).rotate_deg(angle).translate(x, y) + ax.transData
    # Short flattened hind and middle swimming legs; slimmer forelegs.
    for side in (-1, 1):
        for xx, yy, xx2, yy2 in [(-.40,.43,-.85,.88),(.03,.48,-.13,.94),(.48,.31,.78,.66)]:
            ax.add_line(Line2D([xx,xx2],[side*yy,side*yy2],color=color,
                               lw=.48, transform=tr,zorder=8))
        for xx, yy, ang in [(-.79,.83,-43),(-.10,.90,-12)]:
            ax.add_patch(Ellipse((xx,side*yy),.39,.12,angle=side*ang,
                                 fc=color,ec="none",transform=tr,zorder=8))
        ax.add_line(Line2D([.95,1.13,1.22],[side*.16,side*.24,side*.21],
                           color=color,lw=.32,transform=tr,zorder=8))
    body = Ellipse((-.16,0),1.69,1.08,facecolor=color,edgecolor="none",transform=tr,zorder=9)
    body.set_gid(identifier+"-elytra")
    ax.add_patch(body)
    ax.add_patch(Ellipse((.58,0),.57,.76,fc=color,ec="none",transform=tr,zorder=10))
    ax.add_patch(Ellipse((.92,0),.45,.48,fc=color,ec="none",transform=tr,zorder=10))
    ax.add_line(Line2D([-.84,.40],[0,0],color=WHITE,alpha=.75,lw=.31,transform=tr,zorder=11))
    # A narrow highlight helps the two wing cases remain visible at print size.
    ax.add_line(Line2D([-.61,-.24,.13],[-.29,-.39,-.34],color=WHITE,alpha=.35,
                       lw=.23,transform=tr,zorder=11))


def fish(ax, x, y, angle=0, scale=1, color=TEAL, identifier="fish"):
    """Original tapered shiner, with forked caudal fin and dorsal/pectoral fins."""
    tr = Affine2D().scale(scale).rotate_deg(angle).translate(x, y)+ax.transData
    verts = [(1.40,0),(.82,-.48),(-.35,-.51),(-.91,-.12),
             (-1.48,-.59),(-1.23,0),(-1.48,.59),(-.91,.12),
             (-.35,.51),(.82,.48),(1.40,0)]
    codes = [PlotPath.MOVETO,PlotPath.CURVE4,PlotPath.CURVE4,PlotPath.CURVE4,
             PlotPath.LINETO,PlotPath.LINETO,PlotPath.LINETO,PlotPath.LINETO,
             PlotPath.CURVE4,PlotPath.CURVE4,PlotPath.CURVE4]
    obj=PathPatch(PlotPath(verts,codes),fc=color,ec="none",transform=tr,zorder=9)
    obj.set_gid(identifier+"-body")
    ax.add_patch(obj)
    ax.add_patch(PathPatch(PlotPath([(-.41,-.35),(-.20,-.73),(.24,-.40)],
                  [PlotPath.MOVETO,PlotPath.LINETO,PlotPath.LINETO]),
                  fc=color,ec="none",transform=tr,zorder=8,alpha=.8))
    ax.add_patch(PathPatch(PlotPath([(.30,.24),(.02,.64),(.60,.33)],
                  [PlotPath.MOVETO,PlotPath.LINETO,PlotPath.LINETO]),
                  fc=color,ec="none",transform=tr,zorder=8,alpha=.75))
    ax.add_patch(Circle((.96,-.13),.075,fc=WHITE,ec="none",transform=tr,zorder=11))
    ax.add_line(Line2D([-.72,.53],[0,0],color=WHITE,lw=.24,alpha=.65,transform=tr,zorder=10))
    ax.add_line(Line2D([.62,.55,.61],[-.28,0,.28],color=WHITE,lw=.24,alpha=.6,transform=tr,zorder=10))


def circular_arrow(ax, x, y, radius, clockwise=False, color=BLUE, lw=.8):
    # The canvas ordinate increases downwards. Decreasing theta is CCW.
    th=np.linspace(-.05*np.pi,1.60*np.pi,90)
    if not clockwise:
        th=th[::-1]
    xx=x+radius*np.cos(th); yy=y+radius*np.sin(th)
    line(ax,xx,yy,color,lw,zorder=5)
    arrow(ax,(xx[-4],yy[-4]),(xx[-1],yy[-1]),color,lw,ms=4.8)


def bracket(ax, left, right, y, h=1.2, color=MUTED, lw=.65):
    line(ax,[left,left,right,right],[y-h,y,y,y-h],color,lw)


def draw_observation(ax):
    panel_heading(ax,3,14,"a","Collective rotation")
    cx,cy=17.5,32
    ax.add_patch(Circle((cx,cy),12.2,fc=PALE_BLUE,ec="none",zorder=0))
    # Distinct track arcs behind each animal make the kinematic origin visible.
    for i,th in enumerate(np.linspace(.14,2*np.pi+.14,10,endpoint=False)):
        radius=9.3+.45*np.sin(i*2.3)
        trackth=np.linspace(th+.78,th+.15,40)
        line(ax,cx+radius*np.cos(trackth),cy+radius*np.sin(trackth),
             color=BLUE,lw=.58,alpha=.35)
        beetle(ax,cx+radius*np.cos(th),cy+radius*np.sin(th),
               angle=np.degrees(th)-90,scale=1.16,
               color=INK,identifier=f"beetle-{i}")
    for i,th in enumerate([.40,2.50,4.60]):
        beetle(ax,cx+4.9*np.cos(th),cy+4.9*np.sin(th),
               angle=np.degrees(th)-90,scale=1.04,color=MUTED,
               identifier=f"beetle-inner-{i}")
    circular_arrow(ax,cx,cy,2.15,color=BLUE)
    ax.add_patch(Circle((cx,cy),.28,fc=BLUE,ec="none",zorder=7))
    text(ax,17.5,47.3,"Whirligig beetles",size=9.5,ha="center")
    text(ax,17.5,51.5,"Positions + directions",size=9.5,ha="center",color=MUTED)
    text(ax,3,56.0,"3 fitting / 12 evaluation recordings",size=9.5,color=MUTED)
    arrow(ax,(31.7,31),(36.0,31),color=MUTED,lw=.85,ms=6)

    x=np.linspace(39.2,60.5,260)
    t=np.linspace(0,2.8*np.pi,x.size)
    m=.75*np.tanh(1.35*np.sin(t)-.27*np.cos(2.5*t))
    yy=32-8.2*m
    line(ax,[38.6,61],[32,32],FAINT,.6,ls=(0,(2,2)))
    line(ax,[38.6,38.6],[22.3,41.5],LIGHT,.65)
    line(ax,[38.6,61],[41.5,41.5],LIGHT,.65)
    line(ax,x,yy,INK,1.0)
    text(ax,49.5,20.9,r"$M(t)$",size=8.2,ha="center")
    text(ax,37.1,23.8,"+1",size=7.2,ha="right",color=MUTED)
    text(ax,37.1,40.0,"-1",size=7.2,ha="right",color=MUTED)
    text(ax,49.5,46,"Time",size=7.2,ha="center",color=MUTED)


def draw_prediction(ax):
    panel_heading(ax,68,14,"b","Predict future rotation")
    # An open two-route composition; no boxes obscure the mathematical relation.
    text(ax,72,23,"Present state",size=7.7)
    text(ax,112.4,23,"+ past",size=7.7)
    text(ax,72,29,r"$(M, P, R, A)_t$",size=8.0)
    text(ax,112,29,r"$M_{t-5}$",size=8.0)
    line(ax,[72,107],[33,33],BLUE,.9)
    line(ax,[112,128],[33,33],BLUE,.9)

    text(ax,72,39,"Six observations",size=7.7)
    sample_x=np.linspace(73.5,103.5,6)
    sample_y=np.array(SPEC["six_observation_y_mm"])
    coefs=np.polyfit(sample_x,sample_y,1)
    line(ax,[72,105],[54.3,54.3],LIGHT,.6)
    line(ax,sample_x,np.polyval(coefs,sample_x),BLUE,.95)
    for xx,yy in zip(sample_x,sample_y):
        ax.add_patch(Circle((xx,yy),.72,fc=WHITE,ec=BLUE,lw=.8,zorder=9))
    text(ax,73.5,57,r"$t-5$",size=7.2,ha="center",color=MUTED)
    text(ax,103.5,57,r"$t$",size=7.2,ha="center",color=MUTED)
    arrow(ax,(107,48),(112,48),color=BLUE,lw=.8,ms=5.5)
    text(ax,122.5,43.8,"State + trend",size=7.5,ha="center")
    text(ax,122.5,49,r"$(a_t,b_t)$",size=8.2,ha="center")
    flow(ax,[(130,29),(136,29),(136.5,35.5),(143,35.5)],BLUE,.95)
    flow(ax,[(132,49),(139,49),(137,40),(143,40)],BLUE,.95)

    # The future curve is intentionally illustrative, with no apparent error gain.
    ax.add_patch(Rectangle((158.5,25),18.5,25.8,fc=PALE_BLUE,ec="none",zorder=0))
    text(ax,160,22,"Future rotation",size=7.5,ha="center")
    x=np.linspace(145.5,158.5,100)
    yy=43.8-5.2*(x-145.5)/13+1.1*np.sin((x-145.5)*.45)
    line(ax,x,yy,INK,1.1)
    now=float(yy[-1])
    xf=np.linspace(158.5,174.2,70)
    yf=now-6*(1-np.exp(-(xf-158.5)/10))
    line(ax,xf,yf,BLUE,1.1,ls=(0,(3,2)))
    line(ax,[158.5,158.5],[27,51],LIGHT,.65,ls=(0,(2,2)))
    ax.add_patch(Circle((158.5,now),.85,fc=INK,ec=WHITE,lw=.4,zorder=9))
    ax.add_patch(Circle((174.2,yf[-1]),.85,fc=BLUE,ec=WHITE,lw=.4,zorder=9))
    arrow(ax,(145.5,51.5),(177,51.5),color=LIGHT,lw=.7,ms=4.5)
    text(ax,158.5,55.3,r"$t$",size=7.2,ha="center",color=MUTED)
    text(ax,173.5,55.3,r"$t+h$",size=7.2,ha="center",color=MUTED)


def draw_generation(ax):
    panel_heading(ax,3,70,"c","Observed and generated switching")
    text(ax,15.8,78,"Bounded models",size=7.5,ha="center")
    # A kernel density serves as a pictorial probabilistic transition, not a fit.
    x=np.linspace(5,25,130)
    z=(x-5)/20
    den=z**2*(1-z)**3
    yy=95-11.3*den/den.max()
    ax.fill_between(x,95,yy,facecolor=PALE_ORANGE,zorder=1)
    line(ax,x,yy,ORANGE,1)
    line(ax,[5,25],[95,95],LIGHT,.65)
    text(ax,5,98.7,"-1",size=7.2,ha="center",color=MUTED)
    text(ax,25,98.7,"+1",size=9.5,ha="center",color=MUTED)
    arrow(ax,(27,88.5),(31,88.5),color=ORANGE,lw=.9,ms=5.5,check=True)

    xp=np.linspace(40.0,85,350)
    u=(xp-40)/45
    mm=.75*np.tanh(10*(u-.06))*np.tanh(6*(.66-u))+.045*np.sin(35*u)
    yp=88.5-8.5*mm
    for m,label in [(.3,"+0.3"),(-.3,"-0.3")]:
        threshold=88.5-8.5*m
        threshold_line=line(ax,[39.3,85.3],[threshold,threshold],LIGHT,.65,ls=(0,(2,2)))
        threshold_line.set_gid("rotation-threshold-positive" if m>0 else "rotation-threshold-negative")
        text(ax,38.1,threshold,label,size=7.2,ha="right",color=MUTED)
    generated_curve=line(ax,xp,yp,ORANGE,1.15)
    generated_curve.set_gid("generated-rotation-schematic")
    # A known origin begins at entry to the positive rotational state.
    # The wait ends at first entry to the opposite state, crossing the centre.
    start_index=int(np.flatnonzero(mm>=.3)[0])
    start=float(np.interp(.3,mm[start_index-1:start_index+1],xp[start_index-1:start_index+1]))
    end_index=int(np.flatnonzero((xp>start)&(mm<=-.3))[0])
    end=float(np.interp(-.3,mm[end_index-1:end_index+1][::-1],xp[end_index-1:end_index+1][::-1]))
    for name,xx in [("complete-switch-start",start),("complete-switch-end",end)]:
        yy=float(np.interp(xx,xp,yp))
        endpoint=Circle((xx,yy),.8,fc=ORANGE,ec=WHITE,lw=.45,zorder=9)
        endpoint.set_gid(name)
        ax.add_patch(endpoint)
        line(ax,[xx,xx],[80.5,yy-1.4],ORANGE,.55,ls=(0,(2,2)))
    bracket(ax,start,end,79.6,h=-1.2,color=ORANGE,lw=.7)
    text(ax,(start+end)/2,76.7,"Complete switch",size=7.4,ha="center")
    text(ax,62.6,99.6,"Switching rate + waiting times",size=7.3,ha="center")


def draw_categorical(ax):
    panel_heading(ax,97,70,"d","State age and finite history")
    xs=[103,117,131,145]
    labels=[r"$t-15$",r"$t-10$",r"$t-5$",r"$t$"]
    states=SPEC["categorical_example_states"]
    # All four symbols use one hue; written signs carry category identity.
    for i,(xx,tm,state) in enumerate(zip(xs,labels,states)):
        text(ax,xx,78.5,tm,size=7.2,ha="center",color=MUTED)
        ax.add_patch(Circle((xx,85),2.7,fc=WHITE if i==0 else PALE_BLUE,
                            ec=LIGHT if i==0 else BLUE,lw=.8,zorder=7))
        text(ax,xx,85,state,size=8.0,ha="center")
        if i<3:
            line(ax,[xx+2.7,xs[i+1]-2.7],[85,85],LIGHT,.75)
    bracket(ax,117,145,92.0,color=BLUE,lw=.7)
    text(ax,131,97.0,"State age",size=7.6,ha="center")
    arrow(ax,(151,85),(158.5,85),color=BLUE,lw=.9,ms=6)
    text(ax,168,78.5,r"$S_{t+5}$",size=8.0,ha="center")
    # Three unequal bars depict a categorical distribution without numeric claims.
    for xx,h,lab in zip([162.2,168,173.8],[2.4,4.0,6.1],["-1","0","+1"]):
        ax.add_patch(Rectangle((xx-1.7,90-h),3.4,h,fc=BLUE,ec="none",alpha=.8,zorder=5))
        text(ax,xx,93.2,lab,size=7.3,ha="center",color=MUTED)
    text(ax,168,99.6,"Next category",size=7.4,ha="center")


def draw_fish(ax):
    ax.add_patch(Rectangle((1.5,108.0),177,27.5,fc=PALE_TEAL,ec="none",zorder=0))
    panel_heading(ax,4,112.7,"e","Cross-species evaluation")
    cx,cy=17,125.3
    # A school with heterogeneous positions and headings, no arena border.
    for i,(dx,dy,ang,s) in enumerate([(-7,-2,12,1.10),(-2,-5,25,1.2),
                                    (4,-4,39,1.12),(8,0,82,1.04),
                                    (3,4,132,1.14),(-3,4,172,1.17),
                                    (-7,2,-174,1.05),(0,0,28,1.20)]):
        # Short motion trails connect the animal shape to direction measurements.
        theta=np.radians(ang)
        bx=cx+dx-2.1*s*np.cos(theta); by=cy+dy-2.1*s*np.sin(theta)
        line(ax,[bx-2*np.cos(theta),bx],[by-2*np.sin(theta),by],TEAL,.5,alpha=.3)
        fish(ax,cx+dx,cy+dy,ang,s,TEAL,identifier=f"shiner-{i}")
    text(ax,32.0,121.8,"Golden shiners",size=8.0,weight="bold")
    text(ax,32.0,127.0,r"$N=10,\ 30,\ 70$",size=7.5,color=MUTED)
    text(ax,32.0,132.0,"9 recordings",size=7.5,color=MUTED)
    arrow(ax,(74,125.5),(83.8,125.5),color=TEAL,lw=1,ms=6.5,check=True)
    text(ax,103,122.8,"Fit on fish",size=8.0,weight="bold",ha="center")
    text(ax,103,129.0,"2 fit / 1 test",size=7.7,ha="center")
    text(ax,103,133.2,"within each size",size=7.2,ha="center",color=MUTED)
    arrow(ax,(122.1,125.5),(132,125.5),color=TEAL,lw=1,ms=6.5,check=True)
    text(ax,156,122.8,"Forecast rotation",size=8.0,ha="center")
    text(ax,156,129.0,"& next category",size=8.0,ha="center")
    text(ax,128.5,103.2,"Four-observation settings",size=7.2,ha="center",color=MUTED)
    # Only the categorical representation transfers through this dashed route.
    flow(ax,[(154,85),(154,94),(154,103),(149,107),
             (132,107),(104,107),(104,118.5)],TEAL,.85,dashed=True)
    ax.add_patch(Circle((154,85),.42,fc=TEAL,ec="none",zorder=9))


def checks_and_export(fig, ax):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    boxes=[t.get_window_extent(renderer) for t in TEXTS]
    outside=[]; overlaps=[]; connector_hits=[]
    for t,b in zip(TEXTS,boxes):
        if b.x0<0 or b.y0<0 or b.x1>fig.bbox.x1 or b.y1>fig.bbox.y1:
            outside.append(t.get_text())
    for i,a in enumerate(boxes):
        for j in range(i+1,len(boxes)):
            b=boxes[j]
            if TEXTS[i]._label_group!=TEXTS[j]._label_group and a.overlaps(b):
                overlaps.append([TEXTS[i].get_text(),TEXTS[j].get_text()])
    for i,c in enumerate(CONNECTORS):
        cp=c.get_transform().transform_path(c.get_path())
        # Iteration flattens Bezier curves while respecting MOVETO/CLOSEPOLY.
        # A CLOSEPOLY placeholder must never create a false segment to (0,0).
        segments=[]; previous=None; first=None
        for vertices,code in cp.iter_segments(curves=False,simplify=False):
            point=vertices[-2:]
            if code==PlotPath.MOVETO:
                previous=point; first=point
            elif code==PlotPath.CLOSEPOLY:
                if previous is not None and first is not None:
                    segments.append(PlotPath([previous,first]))
                previous=None
            elif previous is not None:
                segments.append(PlotPath([previous,point])); previous=point
        for t,b in zip(TEXTS,boxes):
            if any(seg.intersects_bbox(b.expanded(1.015,1.03),filled=False) for seg in segments):
                connector_hits.append({"connector":i,"text":t.get_text()})
    report={
        "width_mm":W,"height_mm":H,
        "minimum_primary_font_pt":min(t.get_fontsize() for t in TEXTS if t._primary_label),
        "minimum_math_subscript_font_pt":min(t.get_fontsize() for t in TEXTS if not t._primary_label),
        "maximum_font_pt":max(t.get_fontsize() for t in TEXTS),
        "journal_width_mm":153.0,"journal_scale":153.0/W,
        "minimum_primary_font_at_journal_width_pt":min(t.get_fontsize() for t in TEXTS if t._primary_label)*153.0/W,
        "text_count":len(TEXTS),"text_overlaps":overlaps,"text_outside_canvas":outside,
        "flow_connectors_crossing_text":connector_hits,
        "six_observation_frames":SPEC["six_observation_frames"],
        "categorical_observation_frames":SPEC["categorical_observation_frames"],"categorical_target_frame":SPEC["categorical_target_frame"],
        "beetle_recordings":SPEC["beetle_recordings"],
        "fish_recordings":SPEC["fish_recordings"],
        "fish_stochastic_generation":False,
        "plotted_empirical_statistics":False,
        "schematic_elements":["animals","motion trails","M(t)","six-value line fit", "forecast", "bounded transition density", "generated M(t)", "category sequence", "category probabilities"],
    }
    stem=HERE/"figure_01_framework"
    fig.savefig(stem.with_suffix(".pdf"),metadata={
        "Title":"Recent history and collective rotation: analysis framework",
        "Author":"","Subject":"Original animal and analysis schematics",
        "Creator":"Matplotlib","CreationDate":None,"ModDate":None})
    fig.savefig(stem.with_suffix(".svg"),metadata={"Date":None})
    fig.savefig(stem.with_suffix(".png"),dpi=450)
    fig.savefig(HERE/"figure_01_framework_preview.png",dpi=200)
    svg=ET.parse(stem.with_suffix(".svg"))
    report["svg_image_count"]=len(svg.findall(".//{http://www.w3.org/2000/svg}image"))
    report["svg_text_count"]=len(svg.findall(".//{http://www.w3.org/2000/svg}text"))
    report["svg_path_count"]=len(svg.findall(".//{http://www.w3.org/2000/svg}path"))
    (HERE/"layout_checks.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    spec_path=SPEC_PATH
    manifest={"input":"figure_spec.json",
              "sha256":hashlib.sha256(spec_path.read_bytes()).hexdigest(),
              "source_of_curve_values":"Deterministic schematic values and formulas; not empirical results.",
              "authoring_file":"figures/overview.py"}
    (HERE/"source_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return report


def main():
    TEXTS.clear(); CONNECTORS.clear(); configure()
    fig=plt.figure(figsize=(W/25.4,H/25.4),dpi=160)
    top=SPEC["canvas_top_offset_mm"]
    ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,W),ylim=(H+top,top),aspect="equal"); ax.axis("off")
    draw_observation(ax); draw_prediction(ax); draw_generation(ax); draw_categorical(ax); draw_fish(ax)
    # The other present-state variables come directly from individual positions
    # and directions; they are not reconstructed from the scalar rotation.
    flow(ax,[(27.0,21),(34.0,15.8),(64.0,16.8),(72.0,20.3)],MUTED,.7)
    # Source branches: forecast, bounded generator, and categorical analysis.
    arrow(ax,(62.2,34),(68.3,34),color=MUTED,lw=.9,ms=6.3,check=True)
    line(ax,[64.3,64.3],[34.0,60.4],LIGHT,.85)
    ax.add_patch(Circle((64.3,34),.5,fc=MUTED,ec=WHITE,lw=.25,zorder=8))
    flow(ax,[(64.3,60.4),(59,61),(25,59),(25,65.0)],ORANGE,.95)
    flow(ax,[(64.3,60.4),(77,61),(130,59),(130,65.0)],BLUE,.95)
    ax.add_patch(Circle((64.3,60.4),.65,fc=MUTED,ec=WHITE,lw=.3,zorder=6))
    # Light separators define reading rows without enclosing text in boxes.
    line(ax,[3,177],[105.2,105.2],FAINT,.75)
    report=checks_and_export(fig,ax)
    plt.close(fig)
    return report





def run(spec_path, output_dir):
    global HERE, SPEC_PATH, SPEC, W, H
    SPEC_PATH=Path(spec_path)
    SPEC=json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    W,H=SPEC["figure_size_mm"]
    HERE=Path(output_dir);HERE.mkdir(parents=True,exist_ok=True)
    return main()
