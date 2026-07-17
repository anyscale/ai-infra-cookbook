#!/usr/bin/env python3
"""Render the four benchmark figures for BENCHMARK_1M.md.

All numbers come from the two 2026-07-14 runs recorded in BENCHMARK_1M.md
(report.json of prodjob_ahv3si4f9k7u89l7stm3wevz2r / prodjob_lhqcatakzmt2ekghuxp7am1qjb).
The Figure 2 Ray Data curve is a reconstruction from summary stats (start 1,
peak 254-256, drain tail); its area is solved numerically to equal the
measured 233,587 GPU-seconds. Everything else is plotted verbatim.

Usage: python3 plot_benchmark.py   ->  writes plots/fig{1..4}_*.png
"""

from pathlib import Path as FSPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Patch
from matplotlib.path import Path as MplPath

# ---------------------------------------------------------------------------
# Measured results (BENCHMARK_1M.md)
# ---------------------------------------------------------------------------
CORE_WALL_S, DATA_WALL_S = 1735.6, 2601.0
CORE_GPU_S, DATA_GPU_S = 443033.6, 233586.9  # GPU-seconds held
CORE_GPU_H, DATA_GPU_H = 123.1, 64.9
UTIL = {  # (mean %, p95 %)
    ("GPU", "core"): (23.8, 100.0),
    ("GPU", "data"): (13.2, 87.0),
    ("CPU", "core"): (4.8, 7.3),
    ("CPU", "data"): (6.4, 21.1),
}
EFF_1M = {"core": 8576, "data": 16263}      # captions / GPU-hour
EFF_43K = {"core": 2028, "data": 9700}      # data value is ~9,700 across runs

# ---------------------------------------------------------------------------
# Design tokens (dataviz reference palette, light mode — validated)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
CORE = "#2a78d6"      # categorical slot 1 (blue)  = Ray Core
DATA = "#008300"      # categorical slot 2 (green) = Ray Data
CORE_LT = "#9ec5f4"   # light step of the blue ramp (meter track)
DATA_LT = "#a3d9a3"   # light step of the green ramp (meter track)

DPI = 200
CSS = DPI / 100.0     # device px per CSS px (specs below are in CSS px)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "text.color": INK,
    "svg.fonttype": "none",
})

OUT = FSPath(__file__).parent / "plots"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def data_per_css_px(ax):
    """(x, y) data units per CSS pixel — call after layout is fixed."""
    ax.figure.canvas.draw()
    bb = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return (x1 - x0) / bb.width * CSS, (y1 - y0) / bb.height * CSS


def round_top_bar(ax, cx, height, width, color, r_css=4, zorder=3):
    """Column with 4px-rounded data end, square at the baseline."""
    ux, uy = data_per_css_px(ax)
    rx = min(r_css * ux, width / 2)
    ry = min(r_css * uy, height)
    x0, x1, h = cx - width / 2, cx + width / 2, height
    verts = [(x0, 0), (x0, h - ry), (x0, h), (x0 + rx, h),
             (x1 - rx, h), (x1, h), (x1, h - ry), (x1, 0), (x0, 0)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
             MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO,
             MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color,
                           linewidth=0, zorder=zorder))


def style_axes(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.grid(axis="y", color=GRID, linewidth=0.75, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, labelsize=8, colors=MUTED)
    for lbl in ax.get_xticklabels():
        lbl.set_color(INK2)


def heading(fig, title, subtitle):
    fig.text(0.035, 0.955, title, fontsize=12.5, fontweight="semibold",
             color=INK, ha="left", va="top")
    fig.text(0.035, 0.885, subtitle, fontsize=8.5, color=INK2,
             ha="left", va="top")


def legend_swatches(fig, y=0.80):
    fig.legend(
        handles=[Patch(facecolor=CORE, label="Ray Core (pre-provisioned)"),
                 Patch(facecolor=DATA, label="Ray Data (autoscaled)")],
        loc="upper left", bbox_to_anchor=(0.033, y), ncol=2, frameon=False,
        fontsize=8.5, labelcolor=INK2, handlelength=1.1, handleheight=1.1,
        columnspacing=1.6,
    )


# ---------------------------------------------------------------------------
# Figure 1 — speed vs efficiency (two panels, one axis each)
# ---------------------------------------------------------------------------
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.3))
    fig.subplots_adjust(left=0.075, right=0.975, top=0.68, bottom=0.10,
                        wspace=0.32)
    heading(fig, "The trade-off: pre-provisioning is faster, autoscaling is 1.9× cheaper",
            "1,055,4xx captions on the same 256-GPU cluster — FineVideo × Qwen3-VL-8B, 2026-07-14. Lower is better in both panels.")
    legend_swatches(fig)

    panels = [
        (axes[0], "End-to-end wall time",
         [CORE_WALL_S, DATA_WALL_S],
         ["1,736 s", "2,601 s"], ["(28.9 min)", "(43.4 min)"],
         3000, np.arange(0, 3001, 600), "{:,.0f}"),
        (axes[1], "GPU-seconds held (billed)",
         [CORE_GPU_S, DATA_GPU_S],
         ["443,034", "233,587"], ["(123.1 GPU·h)", "(64.9 GPU·h)"],
         500_000, np.arange(0, 500_001, 100_000), "{:,.0f}"),
    ]
    for ax, title, vals, labels, sublabels, ymax, yticks, fmt in panels:
        ax.set_xlim(-0.7, 1.7)
        ax.set_ylim(0, ymax)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{int(v/1000)}k" if ymax > 10_000 else f"{v:,.0f}"
                            for v in yticks])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Ray Core", "Ray Data"])
        style_axes(ax)
        ax.set_title(f"{title} — lower is better", fontsize=9.5, color=INK2,
                     pad=8, loc="left")
        ux, uy = data_per_css_px(ax)
        w = 24 * ux
        for x, v, c in zip([0, 1], vals, [CORE, DATA]):
            round_top_bar(ax, x, v, w, c)
        for x, v, lab, sub in zip([0, 1], vals, labels, sublabels):
            ax.text(x, v + 16 * uy, lab, ha="center", va="bottom",
                    fontsize=9.5, fontweight="semibold", color=INK)
            ax.text(x, v + 4 * uy, sub, ha="center", va="bottom",
                    fontsize=7.5, color=MUTED)

    fig.savefig(OUT / "fig1_speed_vs_cost.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — GPUs held over time (area under the curve = the bill)
# ---------------------------------------------------------------------------
def data_curve():
    """Reconstruct the Ray Data gpus-held profile from summary stats.

    Shape: power-law ramp (1 -> 254 over RAMP s), short full-fleet plateau,
    power-law drain to the 2,601 s wall.  The drain exponent is solved so the
    integral equals the measured 233,587 GPU-seconds (mean 89.8 x 2,601 s).
    """
    RAMP, PLATEAU, PEAK = 900.0, 300.0, 254.0
    t = np.arange(0.0, DATA_WALL_S + 1)

    def build(p):
        n = np.empty_like(t)
        ramp = t <= RAMP
        n[ramp] = 1 + (PEAK - 1) * (t[ramp] / RAMP) ** 1.8
        flat = (t > RAMP) & (t <= RAMP + PLATEAU)
        n[flat] = PEAK
        drain = t > RAMP + PLATEAU
        u = (DATA_WALL_S - t[drain]) / (DATA_WALL_S - RAMP - PLATEAU)
        n[drain] = PEAK * u ** p
        return n

    lo, hi = 1.0, 10.0  # integral decreases monotonically in p
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.trapezoid(build(mid), t) > DATA_GPU_S:
            lo = mid
        else:
            hi = mid
    n = build((lo + hi) / 2)
    assert abs(np.trapezoid(n, t) - DATA_GPU_S) / DATA_GPU_S < 0.005
    return t, n


def fig2():
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    fig.subplots_adjust(left=0.065, right=0.975, top=0.70, bottom=0.17)
    heading(fig, "Why autoscaling is cheaper: GPUs held over time",
            "The bill is the area under each curve. Ray Core holds all 256 GPUs for the full run; Ray Data’s pool ramps from 1 to 254–256, then scales in during the drain.")
    fig.legend(handles=[Line2D([], [], color=CORE, lw=1.5, label="Ray Core (pre-provisioned)"),
                        Line2D([], [], color=DATA, lw=1.5, label="Ray Data (autoscaled)")],
               loc="upper left", bbox_to_anchor=(0.033, 0.82), ncol=2,
               frameon=False, fontsize=8.5, labelcolor=INK2, handlelength=1.4,
               columnspacing=1.6)

    ax.set_xlim(0, 2750)
    ax.set_ylim(0, 300)
    ax.set_yticks([0, 64, 128, 192, 256])
    ax.set_xticks(np.arange(0, 2701, 300))
    ax.set_xticklabels([f"{v:,}" for v in np.arange(0, 2701, 300)])
    style_axes(ax)
    for lbl in ax.get_xticklabels():
        lbl.set_color(MUTED)
    ax.set_xlabel("Time since t0 (s) — both clusters fully provisioned at t0",
                  fontsize=8.5, color=INK2)
    ax.set_ylabel("GPUs held", fontsize=8.5, color=INK2)

    # Ray Core: flat block at 256 for the whole 1,735.6 s run
    tc = [0, CORE_WALL_S, CORE_WALL_S]
    nc = [256, 256, 0]
    ax.fill_between([0, CORE_WALL_S], [256, 256], color=CORE, alpha=0.10, lw=0)
    ax.plot(tc, nc, color=CORE, lw=1.5, solid_capstyle="round", zorder=4)

    # weights-loading phase: GPUs billed, zero captions
    ax.fill_between([0, 180], [256, 256], color=CORE, alpha=0.08, lw=0)
    ax.annotate("weights loading ≈ 3 min\n(256 GPUs already held, 0 captions)",
                xy=(180, 226), xytext=(340, 220), fontsize=7.5, color=INK2,
                va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.75))

    # Ray Data: reconstructed ramp / plateau / drain, area = measured bill
    t, n = data_curve()
    ax.fill_between(t, n, color=DATA, alpha=0.10, lw=0)
    ax.plot(t, n, color=DATA, lw=1.5, solid_capstyle="round", zorder=4)

    # area (= cost) labels and endpoint labels
    ax.text(870, 268, "Ray Core — 443,034 GPU·s held (123.1 GPU·h)",
            ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.text(1060, 110, "Ray Data\n233,587 GPU·s held\n(64.9 GPU·h)",
            ha="center", va="center", fontsize=8.5, color=INK2)
    ax.text(CORE_WALL_S - 22, 8, "done at 1,736 s", ha="right", va="bottom",
            fontsize=7.5, color=INK2)
    ax.text(DATA_WALL_S, 24, "done at 2,601 s", ha="right", va="bottom",
            fontsize=7.5, color=INK2)

    fig.text(0.035, 0.035,
             "Ray Data curve reconstructed from run summary stats (peak 254–256, mean 89.8 GPUs); its area is solved to the measured 233,587 GPU·s. "
             "Ray Core drawn at its 256-GPU peak (sampled mean 255.3).",
             fontsize=7, color=MUTED, ha="left")

    fig.savefig(OUT / "fig2_gpus_held_over_time.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — utilization: mean fill vs p95 track
# ---------------------------------------------------------------------------
def fig3():
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    fig.subplots_adjust(left=0.07, right=0.975, top=0.68, bottom=0.16)
    heading(fig, "The bottleneck is feeding frames, not captioning them",
            "Solid fill = mean utilization · light track = p95 (5 s samples over the timed run). The gap between\n"
            "mean and p95 GPU utilization is time the fleet spends waiting for decoded frames.")
    legend_swatches(fig)

    pos = {("GPU", "core"): 0.0, ("GPU", "data"): 0.9,
           ("CPU", "core"): 2.3, ("CPU", "data"): 3.2}
    ax.set_xlim(-0.65, 3.85)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels(["Ray Core", "Ray Data"] * 2)
    style_axes(ax)

    ux, uy = data_per_css_px(ax)
    w = 24 * ux
    for (hw, engine), x in pos.items():
        mean, p95 = UTIL[(hw, engine)]
        hue, track = (CORE, CORE_LT) if engine == "core" else (DATA, DATA_LT)
        round_top_bar(ax, x, p95, w, track, zorder=2)
        round_top_bar(ax, x, mean, w, hue, zorder=3)
        ax.text(x, p95 + 5 * uy, f"p95 {p95:g}%", ha="center", va="bottom",
                fontsize=8, color=INK2)
        ax.text(x + w / 2 + 5 * ux, mean, f"mean {mean:g}%", ha="left",
                va="center", fontsize=7.5, color=MUTED)

    for center, label in [(0.45, "GPU utilization"), (2.75, "CPU utilization")]:
        ax.text(center, -0.16, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color=INK2,
                fontweight="semibold")

    fig.savefig(OUT / "fig3_utilization_mean_vs_p95.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — cost efficiency by workload scale
# ---------------------------------------------------------------------------
def fig4():
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    fig.subplots_adjust(left=0.085, right=0.975, top=0.68, bottom=0.15)
    heading(fig, "Autoscaling’s advantage compounds on burstier workloads",
            "Captions per GPU-hour held — higher is better. Same corpus at two caption densities.\n"
            "The shorter the useful-fleet window, the more pre-provisioned idle time costs.")
    legend_swatches(fig)

    groups = [
        (0.0, "1M captions — dense\n(24 windows / clip)", EFF_1M, "1.9×"),
        (1.6, "43.7k captions — shallow\n(1 caption / clip)", EFF_43K, "4.8×"),
    ]
    ax.set_xlim(-0.75, 2.35)
    ax.set_ylim(0, 19_500)
    ax.set_yticks(np.arange(0, 19_001, 4000))
    ax.set_yticklabels([f"{v//1000}k" if v else "0"
                        for v in np.arange(0, 19_001, 4000)])
    ax.set_xticks([g[0] for g in groups])
    ax.set_xticklabels([g[1] for g in groups])
    style_axes(ax)

    ux, uy = data_per_css_px(ax)
    w = 24 * ux
    gap = 2 * ux
    for center, _, eff, mult in groups:
        xc, xd = center - (w + gap) / 2, center + (w + gap) / 2
        round_top_bar(ax, xc, eff["core"], w, CORE)
        round_top_bar(ax, xd, eff["data"], w, DATA)
        approx = "~" if eff["data"] == 9700 else ""
        # Core label right-aligned at the bar edge so it clears the taller
        # green neighbor
        ax.text(xc + w / 2, eff["core"] + 5 * uy, f"{eff['core']:,}",
                ha="right", va="bottom", fontsize=8.5, color=INK2)
        ax.text(xd, eff["data"] + 5 * uy, f"{approx}{eff['data']:,}",
                ha="center", va="bottom", fontsize=8.5, color=INK2)
        top = max(eff.values())
        ax.text(center, top + 42 * uy, f"Ray Data {mult}", ha="center",
                va="bottom", fontsize=10.5, fontweight="semibold", color=INK)
        ax.text(center, top + 30 * uy, "more captions per GPU-hour",
                ha="center", va="bottom", fontsize=7.5, color=MUTED)

    fig.savefig(OUT / "fig4_efficiency_by_scale.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    for p in sorted(OUT.glob("fig*.png")):
        print(p)
