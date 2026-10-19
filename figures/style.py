"""Publication figures using saved numerical results and standard PDF fonts."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter
import pymupdf
from . import panels as p
from . import additional_panels as extra

HERE=Path(__file__).resolve().parent
ROOT=HERE
WIDTH=180/25.4
MIN_FONT=10.0
MANIFEST={}
sys.stdout.reconfigure(encoding="utf-8")

SUPPLEMENT_ORDER = {1: 5, 2: 6, 3: 2, 4: 3, 5: 1, 6: 7, 7: 8, 8: 10, 9: 4, 10: 9}

def public_name(name):
    match=re.match(r"figure_s(\d+)",name)
    if match:return f"figureS{SUPPLEMENT_ORDER[int(match.group(1))]}"
    match=re.match(r"figure_(\d+)",name)
    if match:return f"figure{int(match.group(1))}"
    return name


def style():
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica","Arial"],
        "font.size":10,"axes.labelsize":10.5,"axes.titlesize":11,
        "xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":10,
        "legend.frameon":False,"pdf.use14corefonts":True,"ps.useafm":True,
        "svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False,
        "axes.linewidth":.7,"lines.linewidth":1.5,"savefig.facecolor":"white",
        "figure.facecolor":"white","axes.formatter.use_mathtext":False,
        "axes.unicode_minus":False})

def panel(ax,letter,title):
    replacements={"Predictive value of state age":"State age and prediction", "Four observations vs current state":"Four observations vs current", "Every evaluation recording":"All evaluation recordings"}
    title=replacements.get(title,title)
    ax.set_title(f"({letter}) {title}",loc="left",pad=12,fontsize=11)

def format_log_axes(fig):
    for ax in fig.axes:
        if ax.get_xscale()=="log":
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
        if ax.get_yscale()=="log":
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())

def sanitize(fig):
    format_log_axes(fig)
    for text in fig.findobj(Text):
        if not text.get_text():continue
        s=text.get_text().replace("−","-")
        if re.fullmatch("[a-h]",s):s=f"({s})"
        text.set_text(s)
        text.set_fontfamily("sans-serif")
        text.set_fontsize(max(MIN_FONT,text.get_fontsize()))
    for ax in fig.axes:
        for title in [ax.title,ax._left_title,ax._right_title]:
            if title.get_text():title.set_fontsize(11)
        for axis in [ax.xaxis,ax.yaxis]:axis.label.set_fontsize(10.5)

def save(fig,name):
    name = public_name(name)
    sanitize(fig)
    arrays=[]
    for ax in fig.axes:
        arrays.append({"lines":[np.column_stack([line.get_xdata(),line.get_ydata()]).tolist() for line in ax.lines],
            "collections":[{"kind":type(c).__name__,"offsets":np.ma.asarray(c.get_offsets()).filled(np.nan).tolist(),"values":np.ma.asarray(c.get_array()).filled(np.nan).tolist() if c.get_array() is not None else None} for c in ax.collections if hasattr(c,"get_offsets")],
            "bars":[{"x":bar.get_x(),"height":bar.get_height(),"width":bar.get_width()} for bar in ax.patches if hasattr(bar,"get_height")]})
    drawndir=HERE/"checks"/"drawn_arrays";drawndir.mkdir(parents=True,exist_ok=True)
    (drawndir/(name+".json")).write_text(json.dumps(arrays,indent=2,default=lambda x:x.item()),encoding="utf-8")
    source_width=fig.get_figwidth()*25.4
    source_height=fig.get_figheight()*25.4
    scale=149.94/source_width
    height_limit={"figure_s01_calibration":.68*255,"figure_s05_recording_heatmaps":.72*255,"figure_s09_order_selection":.72*255}.get(name)
    if height_limit:scale=min(scale,height_limit/source_height)
    out=HERE;out.mkdir(parents=True,exist_ok=True)
    for extension in ["pdf","svg"]:
        fig.savefig(out/f"{name}.{extension}",dpi=360)
    pdf=pymupdf.open(out/f"{name}.pdf");page=pdf[0]
    page.get_pixmap(dpi=360,alpha=False).save(out/f"{name}.png")
    spans=[s for b in page.get_text("dict")["blocks"] if "lines" in b for line in b["lines"] for s in line["spans"] if s["text"].strip()]
    outside=[s["text"] for s in spans if not page.rect.contains(pymupdf.Rect(s["bbox"]))]
    minimum=min(s["size"] for s in spans)
    fonts=sorted(set(s["font"] for s in spans))
    letters=re.findall(r"\([a-h]\)",page.get_text())
    MANIFEST[name]={"files":[f"{name}.{e}" for e in ["pdf","svg","png"]],"width_mm":source_width,"height_mm":source_height,"scale_at_0p98_textwidth":scale,"minimum_pdf_font_pt":minimum,"minimum_final_font_pt":minimum*scale,"pdf_fonts":fonts,"panel_labels":letters,"outside_text":outside,"ordinary_font_pass":minimum*scale>=7.999,"standard_font_pass":all(f.startswith(("Helvetica","Times","Courier","Symbol")) for f in fonts),"pdf_vector_only":not page.get_images(),"png_dpi":360}
    p.OUTPUTS[name]=MANIFEST[name]["files"]
    plt.close(fig)
