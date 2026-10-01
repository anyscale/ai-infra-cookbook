"""Parse the two 1M-caption job driver logs and render benchmark figures."""

import re
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

CORE_LOG = "/tmp/vc_logs/core_1m_driver.log"
DATA_LOG = "/tmp/vc_logs/data_1m_driver.log"
OUT = "/Users/xinyu/code/work/anyscale/examples/video_captioning/figures"

# ---- palette / chrome (dataviz reference palette, light mode) --------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
CORE_C = "#2a78d6"  # slot 1: Ray Core
DATA_C = "#1baf7a"  # slot 2: Ray Data

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "font.size": 10,
    }
)

TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")


def ts(line):
    m = TS.match(line)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp() + int(m.group(2)) / 1000


def style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ---- parse Ray Core: "captions=N" progress ---------------------------------
core_pts = []
core_t0 = None
for line in open(CORE_LOG, errors="replace"):
    if core_t0 is None and "Loading 256 vLLM engines" in line:
        core_t0 = ts(line)
    m = re.search(r"\[INFO\] video_captioning: captions=(\d+)", line)
    if m:
        t = ts(line)
        if t:
            core_pts.append((t, int(m.group(1))))
core_pts.sort()
if core_t0 is None:
    core_t0 = core_pts[0][0]
CORE_WALL = 1735.6
core_x = [0.0] + [(t - core_t0) / 60 for t, _ in core_pts] + [CORE_WALL / 60]
core_y = [0] + [c for _, c in core_pts] + [1055410]

# ---- parse Ray Data: vLLM stage progress + GPUs held ------------------------
data_prog, data_gpu = {}, {}
for line in open(DATA_LOG, errors="replace"):
    t = ts(line)
    if t is None:
        continue
    m = re.search(r"MapBatches\(vLLMEngineStageUDF\): (\d+)/\d+", line)
    if m:
        data_prog[t] = max(data_prog.get(t, 0), int(m.group(1)))
    m = re.search(r"Active & requested resources: ", line)
    if m:
        g = re.search(r"([0-9.]+)/256 GPU", line)
        data_gpu[t] = float(g.group(1)) if g else 0.0

data_t0 = min(min(data_prog), min(data_gpu))
DATA_WALL = 1124.6
prog = sorted(data_prog.items())
# enforce monotonic (duplicated log sections)
mono, best = [], 0
for t, v in prog:
    best = max(best, v)
    mono.append((t, best))
data_x = [0.0] + [(t - data_t0) / 60 for t, _ in mono] + [DATA_WALL / 60]
data_y = [0] + [v for _, v in mono] + [1055409]
gpu_pts = sorted(data_gpu.items())
gx = [0.0] + [(t - data_t0) / 60 for t, _ in gpu_pts] + [DATA_WALL / 60]
gy = [0.0] + [v for _, v in gpu_pts] + [0.0]

# ---- parse model-load samples (CAIOS weight streaming) ----------------------
loads = {"Ray Core": [], "Ray Data": []}
for name, path in (("Ray Core", CORE_LOG), ("Ray Data", DATA_LOG)):
    for line in open(path, errors="replace"):
        m = re.search(r"Model loading took ([0-9.]+) GiB memory and ([0-9.]+) seconds", line)
        if m:
            loads[name].append(float(m.group(2)))
print("model-load samples:", {k: (len(v), min(v), max(v)) for k, v in loads.items() if v})

MILLIONS = FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e5 else f"{int(v)}")

# ---- fig 1: caption progress ------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.4), dpi=200)
style(ax)
ax.plot(core_x, core_y, color=CORE_C, linewidth=2, solid_capstyle="round")
ax.plot(data_x, data_y, color=DATA_C, linewidth=2, solid_capstyle="round")
ax.annotate("Ray Core — 28.9 min", (core_x[-1], core_y[-1]), xytext=(6, 4),
            textcoords="offset points", color=INK, fontsize=10, fontweight="bold")
ax.annotate("Ray Data — 18.7 min", (data_x[-1], data_y[-1]), xytext=(6, -12),
            textcoords="offset points", color=INK, fontsize=10, fontweight="bold")
for x95, color, dy in ((27.9, CORE_C, -30), (14.6, DATA_C, 14)):
    ax.scatter([x95], [0.95 * 1055300], s=42, color=color, edgecolors=SURFACE,
               linewidths=2, zorder=4)
    ax.annotate(f"95% at {x95} min", (x95, 0.95 * 1055300), xytext=(-4, dy),
                textcoords="offset points", ha="right", color=INK2, fontsize=9)
ax.set_xlim(0, 34)
ax.set_ylim(0, 1.12e6)
ax.yaxis.set_major_formatter(MILLIONS)
ax.set_xlabel("Minutes since pipeline start (end-to-end clock)")
ax.set_ylabel("Captions completed")
ax.set_title("1.06M captions: progress over time", loc="left", fontsize=13,
             fontweight="bold", color=INK, pad=28)
ax.text(0, 1.06, "Ray Core pre-provisions 256 engines before its first caption; "
        "Ray Data streams immediately; tuned pool floor + queue depth",
        transform=ax.transAxes, color=INK2, fontsize=9.5)
fig.tight_layout()
fig.savefig(f"{OUT}/progress_1m.png", bbox_inches="tight")

# ---- fig 2: GPUs held --------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.4), dpi=200)
style(ax)
ax.plot([0, CORE_WALL / 60, CORE_WALL / 60], [256, 256, 0], color=CORE_C,
        linewidth=2, solid_capstyle="round")
ax.plot(gx, gy, color=DATA_C, linewidth=2, drawstyle="steps-post", solid_capstyle="round")
ax.annotate("Ray Core — fleet pinned at 256\n(443,034 GPU-s)", (14, 256),
            xytext=(0, 8), textcoords="offset points", color=INK, fontsize=10,
            fontweight="bold", ha="center")
ax.annotate("Ray Data — pool sized to the queue\n(118,481 GPU-s)", (25, 130),
            color=INK, fontsize=10, fontweight="bold", ha="left")
ax.set_xlim(0, 34)
ax.set_ylim(0, 300)
ax.set_xlabel("Minutes since pipeline start (end-to-end clock)")
ax.set_ylabel("GPUs held by caption engines")
ax.set_title("GPUs held over time — the area is what you pay", loc="left",
             fontsize=13, fontweight="bold", color=INK, pad=28)
ax.text(0, 1.06, "Same 1.06M captions: the autoscaled pool ramps to the full "
        "fleet only while the queue justifies it (3.7× fewer GPU-seconds than the pinned fleet)",
        transform=ax.transAxes, color=INK2, fontsize=9.5)
fig.tight_layout()
fig.savefig(f"{OUT}/gpus_held_1m.png", bbox_inches="tight")

# ---- fig 3: efficiency bars --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), dpi=200)
panels = [
    ("GPU-hours consumed", [123.1, 32.9], "{:.0f} GPU-h"),
    ("Captions per GPU-hour", [8576, 32068], "{:,.0f}"),
]
for ax, (title, vals, fmt) in zip(axes, panels):
    style(ax)
    bars = ax.bar([0, 1], vals, width=0.55, color=[CORE_C, DATA_C])
    for x, v in zip([0, 1], vals):
        ax.text(x, v, " " + fmt.format(v), ha="center", va="bottom",
                color=INK, fontsize=10, fontweight="bold")
    ax.set_xticks([0, 1], ["Ray Core", "Ray Data"], color=INK2)
    ax.set_ylim(0, max(vals) * 1.22)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
fig.suptitle("Same 1.06M captions, same 256-GPU cluster", x=0.01, ha="left",
             fontsize=13, fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(f"{OUT}/efficiency_1m.png", bbox_inches="tight")

# ---- fig 4: CAIOS weight streaming -------------------------------------------
fig, ax = plt.subplots(figsize=(8, 3.4), dpi=200)
style(ax)
import random

random.seed(7)
for i, (name, color) in enumerate((("Ray Core", CORE_C), ("Ray Data", DATA_C))):
    vals = loads[name]
    ys = [i + random.uniform(-0.13, 0.13) for _ in vals]
    ax.scatter(vals, ys, s=64, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
    med = sorted(vals)[len(vals) // 2]
    ax.annotate(f"{name} · median {med:.0f} s", (med, i), xytext=(0, 18),
                textcoords="offset points", ha="center", color=INK,
                fontsize=10, fontweight="bold")
ax.set_yticks([])
ax.set_xlim(left=0)
ax.set_ylim(-0.6, 1.75)
ax.set_xlabel("Per-engine weight load, S3 to GPU (seconds; 16.97 GiB each)")
ax.set_title("CAIOS weight streaming: Qwen3-VL-8B via Run:ai streamer",
             loc="left", fontsize=13, fontweight="bold", color=INK, pad=28)
ax.text(0, 1.09, "Log-deduplicated sample of the 256 engine boots per run; "
        "~1–1.6 GiB/s per engine under 256-way concurrent load",
        transform=ax.transAxes, color=INK2, fontsize=9.5)
fig.tight_layout()
fig.savefig(f"{OUT}/caios_weight_streaming.png", bbox_inches="tight")

print("core points:", len(core_pts), "| data progress points:", len(mono),
      "| gpu points:", len(gpu_pts))
print("saved 4 figures to", OUT)
