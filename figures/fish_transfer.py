"""Recreate the golden-shiner figure using the two bundled input files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
CHECKS = OUT / "checks"
WIDTH_MM, HEIGHT_MM = 180.0, 142.0
INK = "#202830"
GREY = "#68727B"
PALE = "#F2F5F7"
BLUE, ORANGE, TEAL = "#0072B2", "#D88813", "#009E83"
GROUP_COLORS = {10: BLUE, 30: ORANGE, 70: TEAL}
GROUP_MARKERS = {10: "o", 30: "s", 70: "^"}
X = np.array([0, 1, 2, 3.2, 4.2, 5.2, 6.4, 7.4, 8.4])


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def configure():
    available = {font.name for font in font_manager.fontManager.ttflist}
    font_name = next((name for name in ["Arial", "Liberation Sans", "DejaVu Sans"] if name in available), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": font_name, "font.size": 9.7,
        "axes.labelsize": 9.7, "axes.titlesize": 10.3,
        "xtick.labelsize": 9.7, "ytick.labelsize": 9.7,
        "legend.fontsize": 9.7, "axes.linewidth": 0.7,
        "xtick.major.width": 0.65, "ytick.major.width": 0.65,
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "text.color": INK, "axes.labelcolor": INK,
        "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "legend.frameon": False, "pdf.fonttype": 42, "pdf.use14corefonts": True,
        "ps.fonttype": 42, "svg.fonttype": "none", "mathtext.default": "regular",
    })


def panel_header(fig, letter, title, x, y):
    fig.text(x / WIDTH_MM, y / HEIGHT_MM, f"({letter})", fontsize=12, fontweight="bold", va="center")
    fig.text((x + 9) / WIDTH_MM, y / HEIGHT_MM, title, fontsize=10.3, fontweight="bold", va="center")


def add_axes_mm(fig, x, y, w, h):
    return fig.add_axes([x / WIDTH_MM, y / HEIGHT_MM, w / WIDTH_MM, h / HEIGHT_MM])


def fish(ax, x, y, scale=1, color=GREY):
    # Editable schematic fish silhouette.
    verts = [(1.25, 0), (.3, -.62), (-.55, -.45), (-.85, -.12),
             (-1.42, -.67), (-1.22, 0), (-1.42, .67), (-.85, .12),
             (-.55, .45), (.3, .62), (1.25, 0)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    tr = Affine2D().scale(scale).translate(x, y) + ax.transData
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", transform=tr))
    ax.add_patch(Circle((.74, -.13), .09, facecolor="white", edgecolor="none", transform=tr))


def schematic(ax, summary, alpha):
    ax.set(xlim=(0, 80), ylim=(0, 49))
    ax.set_axis_off()
    ax.set_aspect("equal")
    boxes = []
    text_in_boxes = []
    arrows = []

    def box(x, y, w, h, face, edge):
        patch = FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0,rounding_size=1.5", facecolor=face,
            edgecolor=edge, linewidth=.8, zorder=1)
        ax.add_patch(patch)
        boxes.append(patch)
        return patch

    def label(x, y, text, target, **kwargs):
        t = ax.text(x, y, text, ha="center", va="center", fontsize=9.7, zorder=3, **kwargs)
        text_in_boxes.append((t, target))
        return t

    top = box(1, 29.5, 78, 18.2, PALE, "#9AA6AE")
    label(40, 43.0, "Beetle-selected settings", top, fontweight="bold", color=INK)
    label(40, 37.4,
          f"Thresholds: ±{summary['central_threshold']:.1f}   |   Spacing: {summary['step_frames']} frames", top)
    label(40, 32.5, f"History: K = {summary['selected_order']}   |   Smoothing: {alpha:.1f}", top)
    arrow = FancyArrowPatch((40, 29.0), (40, 24.0), arrowstyle="-|>",
                           mutation_scale=9, linewidth=1.0, color=GREY)
    ax.add_patch(arrow)
    arrows.append(arrow)

    lower = box(1, 0.0, 78, 23.3, "#F2F8F8", "#93B8B2")
    fish(ax, 7, 18.5, scale=1.55)
    label(43, 18.5, "Golden shiners · N = 10, 30, 70", lower, fontweight="bold")
    label(40, 13.6, "3 recordings per size; rotate the holdout", lower)
    label(19, 8.5, "Refit on 2 recordings", lower, fontweight="bold")
    label(19, 3.2, "Transition probabilities", lower, color=INK)
    label(61, 8.5, "Evaluate on 1", lower, fontweight="bold")
    label(61, 3.2, "Held-out recording", lower)
    arrow = FancyArrowPatch((38.0, 8.5), (43.5, 8.5), arrowstyle="-|>",
                           mutation_scale=8, linewidth=.9, color=GREY)
    ax.add_patch(arrow)
    arrows.append(arrow)
    return text_in_boxes, arrows


def recording_axis(ax, data):
    ax.set_xlim(-.55, 8.95)
    ax.set_xticks(X, [str(int(value)) for value in data["trial"]])
    ax.tick_params(axis="x", length=0, pad=5)
    trans = ax.get_xaxis_transform()
    for n, lo, hi in [(10, -.35, 2.35), (30, 2.85, 5.55), (70, 6.05, 8.75)]:
        ax.plot([lo, hi], [-.15, -.15], color=GROUP_COLORS[n], linewidth=1.6,
                transform=trans, clip_on=False)
        ax.text((lo + hi) / 2, -.24, f"N = {n}", transform=trans,
                ha="center", va="center", fontsize=9.7)
    ax.set_xlabel("Held-out recording", labelpad=27)


def run(data_dir, output_dir):
    global OUT, CHECKS, BASE
    BASE=Path(data_dir)
    OUT=Path(output_dir)
    CHECKS=OUT/"checks"/"figure7"
    CHECKS.mkdir(parents=True,exist_ok=True)
    table_path = BASE / "inputs" / "recording_scores.csv"
    settings_path = BASE / "inputs" / "settings.json"
    data = pd.read_csv(table_path, dtype={"trial": str, "trial_id": str, "training_trials": str})
    data = data.sort_values(["group_size", "trial"]).reset_index(drop=True)
    summary = json.loads(settings_path.read_text(encoding="utf-8"))
    assert len(data) == 9
    assert data.groupby("group_size").size().to_dict() == {10: 3, 30: 3, 70: 3}
    assert data.selected_order.eq(4).all()
    assert data.alpha_order_1.eq(.5).all() and data.alpha_selected_order.eq(.5).all()
    for row in data.itertuples():
        training = sorted(data.loc[(data.group_size == row.group_size) & (data.trial != row.trial), "trial"])
        assert sorted(row.training_trials.split(",")) == training
    data["nll_reduction_pct"] = 100 * data.selected_relative_gain
    data["brier_reduction_pct"] = 100 * (1 - data.brier_selected_order / data.brier_order_1)
    data["rps_reduction_pct"] = 100 * (1 - data.rps_selected_order / data.rps_order_1)
    recomputed_nll = 100 * (1 - data.nll_selected_order / data.nll_order_1)
    np.testing.assert_allclose(data.nll_reduction_pct, recomputed_nll, atol=1e-9, rtol=0)
    group_means = data.groupby("group_size").selected_relative_gain.mean() * 100
    balanced = 100 * summary["balanced_relative_gain"]
    np.testing.assert_allclose(group_means.mean(), balanced, atol=1e-9, rtol=0)
    means = [float(group_means.loc[n]) for n in [10, 30, 70]] + [balanced]
    data.to_csv(OUT / "plotted_data.csv", index=False, float_format="%.15g")
    configure()
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    panel_header(fig, "a", "Fixed settings; fish-specific fitting", 4, 137)
    panel_header(fig, "b", "Recording-level loss", 95, 137)
    panel_header(fig, "c", "Probability scores", 4, 72)
    panel_header(fig, "d", "Group means", 95, 72)

    a = add_axes_mm(fig, 4, 84, 80, 49)
    b = add_axes_mm(fig, 106, 94, 70, 37)
    c = add_axes_mm(fig, 15, 22, 69, 43)
    d = add_axes_mm(fig, 106, 22, 70, 43)
    text_in_boxes, arrows = schematic(a, summary, float(data.alpha_selected_order.iloc[0]))

    for idx, row in data.iterrows():
        b.bar(X[idx], row.nll_reduction_pct, width=.72, color=GROUP_COLORS[int(row.group_size)],
              edgecolor=INK, linewidth=.45, zorder=3)
    b.set(ylim=(0, 25), yticks=np.arange(0, 26, 5), ylabel="Loss reduction (%)")
    b.yaxis.grid(True, linewidth=.45, color="#E5E8EB", zorder=0)
    b.text(.02, .94, "9 of 9 recordings improved", transform=b.transAxes, fontsize=9.7, va="top")
    recording_axis(b, data)

    for idx, row in data.iterrows():
        color = GROUP_COLORS[int(row.group_size)]
        c.scatter(X[idx] - .14, row.brier_reduction_pct, s=21, marker="s",
                  color=color, edgecolor=INK, linewidth=.4, zorder=3)
        c.scatter(X[idx] + .14, row.rps_reduction_pct, s=24, marker="D",
                  facecolor="white", edgecolor=color, linewidth=1.0, zorder=3)
    c.set(ylim=(0, 14), yticks=np.arange(0, 13, 2), ylabel="Score reduction (%)")
    for y in [2, 4, 6, 8, 10]:
        c.axhline(y, linewidth=.45, color="#E5E8EB", zorder=0)
    handles = [Line2D([], [], color="none", marker="s", markerfacecolor=GREY, markeredgecolor=INK,
                       markeredgewidth=.4, markersize=4.5, label="Brier score"),
               Line2D([], [], color="none", marker="D", markerfacecolor="white", markeredgecolor=GREY,
                       markeredgewidth=1, markersize=4.7, label="Ranked probability score")]
    c.legend(handles=handles, loc="upper left", bbox_to_anchor=(-.025, 1.015),
             handletextpad=.45, labelspacing=.65, borderpad=.25)
    recording_axis(c, data)

    for idx, value in enumerate(means):
        color = GROUP_COLORS[[10, 30, 70][idx]] if idx < 3 else INK
        marker = GROUP_MARKERS[[10, 30, 70][idx]] if idx < 3 else "D"
        d.scatter(idx, value, s=36, marker=marker, facecolor=color if idx < 3 else "white",
                  edgecolor=INK, linewidth=.8, zorder=3)
        d.text(idx, value + 2.3, f"{value:.3f}", fontsize=9.7, ha="center", va="center",
               fontweight="bold" if idx == 3 else "normal")
    d.set(xlim=(-.55, 3.55), ylim=(0, 25), yticks=np.arange(0, 26, 5),
          ylabel="Loss reduction (%)")
    d.set_xticks(range(4), ["N = 10", "N = 30", "N = 70", "Group-\nbalanced"])
    for tick in d.get_xticklabels():
        tick.set_linespacing(1.7)
    d.tick_params(axis="x", length=0, pad=6, labelsize=9.7)
    d.axvline(2.5, color="#D6DBDF", linewidth=.65, linestyle=(0, (2, 3)))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box_checks = []
    for text_artist, box_artist in text_in_boxes:
        tb = text_artist.get_window_extent(renderer)
        bb = box_artist.get_window_extent(renderer)
        inside = tb.x0 > bb.x0 and tb.x1 < bb.x1 and tb.y0 > bb.y0 and tb.y1 < bb.y1
        box_checks.append({"text": text_artist.get_text(), "inside_box": bool(inside),
                           "clearance_points": min(tb.x0-bb.x0, bb.x1-tb.x1, tb.y0-bb.y0, bb.y1-tb.y1) * 72 / fig.dpi})
    crossings = []
    for text_artist, _ in text_in_boxes:
        tb = text_artist.get_window_extent(renderer).expanded(1.02, 1.1)
        for idx, arrow in enumerate(arrows):
            if tb.overlaps(arrow.get_window_extent(renderer)):
                crossings.append({"text": text_artist.get_text(), "arrow": idx})
    recording_label_gaps = {}
    for name, ax in [("b", b), ("c", c)]:
        extents = [tick.get_window_extent(renderer) for tick in ax.get_xticklabels()]
        gaps = [(second.x0 - first.x1) * 72 / fig.dpi for first, second in zip(extents, extents[1:])]
        recording_label_gaps[name] = min(gaps)
        assert min(gaps) > 1.0, "Recording labels need more horizontal space"
    assert not b.xaxis.label.get_window_extent(renderer).overlaps(fig.texts[7].get_window_extent(renderer))
    write_json(CHECKS / "layout_checks.json", {"box_text": box_checks, "arrow_text_intersections": crossings,
               "minimum_recording_label_gap_pt": recording_label_gaps})
    assert all(row["inside_box"] for row in box_checks), "A diagram label exceeds its box"
    assert not crossings, "A connector intersects text"

    stem = OUT / "figure7"
    metadata = {"Title": "Fixed analysis settings transferred to golden shiners", "Creator": "Matplotlib",
                "CreationDate": None, "ModDate": None}
    text_artists = fig.findobj(match=matplotlib.text.Text)
    original_families = [artist.get_fontfamily() for artist in text_artists]
    for artist in text_artists:
        artist.set_fontfamily("Helvetica")
    fig.savefig(stem.with_suffix(".pdf"), metadata=metadata)
    for artist, family in zip(text_artists, original_families):
        artist.set_fontfamily(family)
    fig.savefig(stem.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(OUT / "preview.png", dpi=220)
    plt.close(fig)

    result = {
        "recordings": len(data), "evaluation_rows": int(data.row_count.sum()),
        "group_means_pct": group_means.to_dict(), "group_balanced_pct": balanced,
        "all_box_labels_inside": all(row["inside_box"] for row in box_checks),
        "connector_text_intersections": len(crossings),
        "width_mm": WIDTH_MM, "height_mm": HEIGHT_MM, "png_dpi": 600,
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [table_path, settings_path]},
        "outputs": ["figure7.pdf", "figure7.svg", "figure7.png", "preview.png", "plotted_data.csv"],
    }
    write_json(OUT / "generation_summary.json", result)
    return result
