"""Render the four narrative plots from the 1M-caption benchmark.

Data sources (all in BENCHMARK_1M.md):
  Ray Core     prodjob_ahv3si4f9k7u89l7stm3wevz2r  (2026-07-14)
  Ray Data     prodjob_6ak48lybysi4cff3wz33pdzz3w  (2026-07-15, tuned:
               pool floor (8,256), max_concurrent_batches=8, num_cpus=0.25 reads)
  43.7k runs   prodjob_wiblhv3b4881khv9dfhq9ifbcz / prodjob_fpt8v6msx4jahxyllimx6zhz5t
gpus_held_tuned.csv holds the tuned run's 10 s GPU-allocation samples.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT = os.path.dirname(os.path.abspath(__file__))

# Reference dataviz palette (light mode); Core=slot 1 blue, Data=slot 2 aqua.
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
CORE_C, DATA_C = "#2a78d6", "#1baf7a"
CORE_LT, DATA_LT = "#9ec5f4", "#a7e3cd"  # light track fills for mean-vs-p95

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": BASELINE,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.labelcolor": INK2, "font.size": 10,
})

CORE = dict(wall=1735.6, gpu_s=443034, gpu_h=123.1,
            gpu_util=(23.8, 100.0), cpu_util=(4.8, 7.3), cph=8576)
DATA = dict(wall=1124.6, gpu_s=118481, gpu_h=32.9,
            gpu_util=(30.3, 99.4), cpu_util=(16.7, 94.5), cph=32068)


def style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def legend(fig):
    fig.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=CORE_C),
                 plt.Rectangle((0, 0), 1, 1, color=DATA_C)],
        labels=["Ray Core (pre-provisioned)", "Ray Data (autoscaled, tuned)"],
        loc="upper left", bbox_to_anchor=(0.01, 0.90), ncol=2, frameon=False,
    )


# ---- fig 1: speed vs cost — no longer a trade-off ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), dpi=200)
panels = [
    ("End-to-end wall time — lower is better",
     [CORE["wall"], DATA["wall"]], ["1,736 s\n(28.9 min)", "1,125 s\n(18.7 min)"]),
    ("GPU-seconds held (billed) — lower is better",
     [CORE["gpu_s"], DATA["gpu_s"]], ["443,034\n(123.1 GPU·h)", "118,481\n(32.9 GPU·h)"]),
]
for ax, (title, vals, labels) in zip(axes, panels):
    style(ax)
    ax.bar([0, 1], vals, width=0.5, color=[CORE_C, DATA_C])
    for x, v, lab in zip([0, 1], vals, labels):
        ax.text(x, v, lab + "\n", ha="center", va="bottom", color=INK,
                fontsize=10, fontweight="bold", linespacing=1.3)
    ax.set_xticks([0, 1], ["Ray Core", "Ray Data"], color=INK2)
    ax.set_ylim(0, max(vals) * 1.32)
    ax.set_title(title, loc="left", fontsize=11, color=INK2, pad=10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
fig.suptitle("No trade-off left: autoscaling is 1.5× faster and 3.7× cheaper",
             x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
fig.text(0.01, 0.895, "1,055,4xx captions on the same 256-GPU cluster — FineVideo × "
         "Qwen3-VL-8B. Tuned Ray Data: pool floor (8,256), 8 in-flight batches/engine.",
         color=INK2, fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.84))
fig.savefig(f"{OUT}/fig1_speed_vs_cost.png", bbox_inches="tight")

# ---- fig 2: GPUs held over time — real samples ------------------------------
t, g = [], []
with open(f"{OUT}/gpus_held_tuned.csv") as f:
    for row in csv.DictReader(f):
        t.append(float(row["t_s"]))
        g.append(float(row["gpus_held"]))
fig, ax = plt.subplots(figsize=(10, 4.8), dpi=200)
style(ax)
ax.fill_between([0, CORE["wall"]], 256, step="post", color=CORE_C, alpha=0.14)
ax.plot([0, CORE["wall"], CORE["wall"]], [256, 256, 0], color=CORE_C, linewidth=2)
ax.fill_between(t, g, step="post", color=DATA_C, alpha=0.22)
ax.plot(t, g, color=DATA_C, linewidth=2, drawstyle="steps-post")
ax.annotate("Ray Core — 443,034 GPU·s held (123.1 GPU·h)", (CORE["wall"] / 2, 256),
            xytext=(0, 8), textcoords="offset points", ha="center",
            color=INK, fontsize=10, fontweight="bold")
ax.annotate("Ray Data — 118,481 GPU·s held\n(32.9 GPU·h)", (430, 95),
            ha="center", color=INK, fontsize=10, fontweight="bold")
ax.annotate("weights loading ≈ 3 min\n(256 GPUs held, 0 captions)", (95, 210),
            color=INK2, fontsize=9, ha="left")
ax.annotate("done at 1,125 s", (DATA["wall"], 12), xytext=(8, 10),
            textcoords="offset points", color=INK2, fontsize=9)
ax.annotate("done at 1,736 s", (CORE["wall"], 12), xytext=(8, 10),
            textcoords="offset points", color=INK2, fontsize=9)
ax.set_xlim(0, 1900)
ax.set_ylim(0, 292)
ax.set_xlabel("Time since t0 (s) — both clusters fully provisioned at t0")
ax.set_ylabel("GPUs held")
ax.set_title("Why autoscaling is cheaper: GPUs held over time", loc="left",
             fontsize=14, fontweight="bold", color=INK, pad=30)
ax.text(0, 1.05, "The bill is the area under each curve. Ray Core pins 256 GPUs "
        "for the full run; Ray Data's pool tracks the queue (10 s allocation samples).",
        transform=ax.transAxes, color=INK2, fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_gpus_held_over_time.png", bbox_inches="tight")

# ---- fig 3: utilization mean vs p95 -----------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.8), dpi=200)
style(ax)
groups = [
    ("GPU utilization", [("Ray Core", CORE["gpu_util"], CORE_C, CORE_LT),
                         ("Ray Data", DATA["gpu_util"], DATA_C, DATA_LT)]),
    ("CPU utilization", [("Ray Core", CORE["cpu_util"], CORE_C, CORE_LT),
                         ("Ray Data", DATA["cpu_util"], DATA_C, DATA_LT)]),
]
xpos, xticklabels = [], []
x = 0.0
for gtitle, bars in groups:
    for name, (mean, p95), c, lt in bars:
        ax.bar([x], [p95], width=0.5, color=lt)
        ax.bar([x], [mean], width=0.5, color=c)
        ax.text(x, p95, f"p95 {p95:.0f}%", ha="center", va="bottom",
                color=INK2, fontsize=9)
        ax.text(x + 0.32, mean, f" mean {mean:.1f}%", ha="left", va="center",
                color=INK, fontsize=9.5, fontweight="bold")
        xpos.append(x)
        xticklabels.append(name)
        x += 1.35
    x += 0.9
ax.text(0.675, -0.17, "GPU utilization", ha="center", color=INK2, fontsize=11,
        transform=ax.get_xaxis_transform())
ax.text(3.6, -0.17, "CPU utilization", ha="center", color=INK2, fontsize=11,
        transform=ax.get_xaxis_transform())
ax.set_xticks(xpos, xticklabels, color=INK2)
ax.set_xlim(-0.75, 5.0)
ax.set_ylim(0, 118)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_title("Bursty by design: mean vs p95 utilization", loc="left",
             fontsize=14, fontweight="bold", color=INK, pad=44)
ax.text(0, 1.05, "Solid = mean, light track = p95 (5 s samples over the timed run). Ray Core's GPU gap is a\n"
        "pinned fleet starved by driver-throttled decode; Ray Data's is a pool that only holds GPUs while they burst.",
        transform=ax.transAxes, color=INK2, fontsize=10, va="bottom", linespacing=1.4)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_utilization_mean_vs_p95.png", bbox_inches="tight")

# ---- fig 4: efficiency by scale ----------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)
style(ax)
scales = [
    ("1M captions — dense\n(24 windows / clip)", 8576, 32068, "3.7×"),
    ("43.7k captions — shallow\n(1 caption / clip)", 2028, 9700, "4.8×"),
]
for i, (label, core_v, data_v, ratio) in enumerate(scales):
    xc, xd = i * 2.2, i * 2.2 + 0.55
    ax.bar([xc], [core_v], width=0.5, color=CORE_C)
    ax.bar([xd], [data_v], width=0.5, color=DATA_C)
    ax.text(xc, core_v, f"{core_v:,}", ha="center", va="bottom", color=INK2, fontsize=10)
    ax.text(xd, data_v, f"{data_v:,}", ha="center", va="bottom", color=INK,
            fontsize=10, fontweight="bold")
    ax.text(xd, data_v + 4300, f"Ray Data {ratio}", ha="center", color=INK,
            fontsize=12, fontweight="bold")
    ax.text((xc + xd) / 2, -4300, label, ha="center", color=INK2, fontsize=10)
ax.set_xticks([])
ax.set_ylim(0, 39500)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
ax.set_ylabel("Captions per GPU-hour held")
ax.set_title("Autoscaling wins at every scale — most where the fleet window is shortest",
             loc="left", fontsize=14, fontweight="bold", color=INK, pad=30)
ax.text(0, 1.05, "Same corpus at two caption densities; higher is better. Tuning "
        "lifted the dense-run ratio from 1.9× to 3.7× (pool floor + deeper engine queues).",
        transform=ax.transAxes, color=INK2, fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_efficiency_by_scale.png", bbox_inches="tight")

print("wrote 4 plots to", OUT)
