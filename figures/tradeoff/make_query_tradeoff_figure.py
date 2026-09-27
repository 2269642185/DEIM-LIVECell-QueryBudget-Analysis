from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from alignment_optional import require_matplotlib_panel_alignment

HERE = Path(__file__).resolve().parent
DATA = pd.read_csv(HERE / "fig_query_tradeoff_data.csv")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

Q = DATA["queries"]
blue = "#2F5D8C"
teal = "#2A9D8F"
orange = "#D97732"
purple = "#7656A5"
gray = "#5B6573"

fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.65), constrained_layout=True)
axes = axes.ravel()

ax = axes[0]
ax.plot(Q, DATA["mask_ap50"], color=blue, marker="o", lw=1.8, ms=4.5, label="Mask AP$_{50}$")
ax.plot(Q, DATA["mask_map"], color=teal, marker="s", lw=1.8, ms=4.3, label="Mask mAP")
ax.set_title("(a) Accuracy")
ax.set_xlabel("Number of queries")
ax.set_ylabel("COCO mask score")
ax.set_xticks(Q)
ax.set_ylim(0.22, 0.66)
ax.grid(axis="y", color="#D9DEE5", lw=0.55, alpha=0.8)
ax.set_xlim(285, 635)
handles, labels = ax.get_legend_handles_labels()

ax = axes[1]
ax.plot(Q, DATA["pipeline_latency_ms"], color=orange, marker="o", lw=1.8, ms=4.5)
ax.set_title("(b) End-to-end pipeline latency")
ax.set_xlabel("Number of queries")
ax.set_ylabel("Mean latency (ms/image)")
ax.set_xticks(Q)
ax.set_ylim(115, 260)
ax.grid(axis="y", color="#D9DEE5", lw=0.55, alpha=0.8)

ax = axes[2]
ax.plot(Q, DATA["peak_allocated_mib"], color=purple, marker="o", lw=1.8, ms=4.5)
ax.set_title("(c) Peak allocated memory")
ax.set_xlabel("Number of queries")
ax.set_ylabel("Memory (MiB)")
ax.set_xticks(Q)
ax.set_ylim(280, 620)
ax.grid(axis="y", color="#D9DEE5", lw=0.55, alpha=0.8)

ax = axes[3]
ax.plot(Q, DATA["model_flops_g"], color=gray, marker="o", lw=1.8, ms=4.5)
ax.set_title("(d) Model-forward computation")
ax.set_xlabel("Number of queries")
ax.set_ylabel("FLOPs (G)")
ax.set_xticks(Q)
ax.set_ylim(60.5, 64.1)
ax.grid(axis="y", color="#D9DEE5", lw=0.55, alpha=0.8)

for i, ax in enumerate(axes):
    ax.spines["left"].set_color("#6B7280")
    ax.spines["bottom"].set_color("#6B7280")
    ax.tick_params(length=3, width=0.7, color="#6B7280")

fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.015),
           ncol=2, frameon=False, handlelength=2.5, columnspacing=1.5)

out = HERE / "fig_query_tradeoff"
require_matplotlib_panel_alignment(
    fig,
    json_out=f"{out}.alignment.json",
    overlay_svg=f"{out}.alignment.svg",
    tolerance_pt=1.5,
    gutter_tolerance_pt=1.5,
    strict=True,
)
fig.savefig(f"{out}.pdf", bbox_inches="tight")
fig.savefig(f"{out}.svg", bbox_inches="tight")
fig.savefig(f"{out}.png", dpi=600, bbox_inches="tight")
fig.savefig(f"{out}.tiff", dpi=600, bbox_inches="tight")
plt.close(fig)
