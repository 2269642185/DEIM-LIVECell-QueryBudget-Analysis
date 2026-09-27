"""Publication figures from verified CSV outputs; Python/matplotlib only."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
from alignment_optional import require_matplotlib_panel_alignment

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.7, "legend.frameon": False,
    "savefig.dpi": 300,
})


def read(name):
    with (HERE / "analysis" / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    dest = HERE / "figures" / name
    dest.parent.mkdir(exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, json_out=str(dest) + ".alignment.json",
        overlay_svg=str(dest) + ".alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    )
    fig.savefig(str(dest) + ".pdf", bbox_inches="tight")
    fig.savefig(str(dest) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def stratified():
    rows = read("four_budget_stratified_primary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.25), sharex=True)
    colors = {"<=300": "#246AA3", "301-500": "#D08029", ">500": "#555B66"}
    marks = {"<=300": "o", "301-500": "s", ">500": "^"}
    for ax, metric, title in zip(axes, ("recall", "candidate_coverage"),
                                 ("a  Observed recall", "b  Candidate coverage")):
        for group in colors:
            rr = sorted((r for r in rows if r["stratum"] == group), key=lambda r:int(r["Q"]))
            ax.plot([int(r["Q"]) for r in rr], [float(r[metric]) for r in rr],
                    color=colors[group], marker=marks[group], markersize=4.4,
                    linewidth=1.55, label=group)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Query budget Q")
        ax.set_ylabel("Fraction of ground-truth instances")
        ax.set_xticks([300,400,500,600])
        ax.set_ylim(0.25,0.91)
        ax.grid(axis="y", color="#E3E6E8", linewidth=0.5)
    fig.text(.55,.975,"GT/image:",ha="left",va="top",fontsize=7)
    for x,group in ((.67,"<=300"),(.77,"301-500"),(.89,">500")):
        fig.text(x,.975,group,ha="left",va="top",fontsize=7,color=colors[group],fontweight="bold")
    fig.subplots_adjust(left=.09,right=.98,top=.89,bottom=.19,wspace=.28)
    save(fig,"four_budget_stratified_response")


def threshold_pr():
    rows = read("four_budget_threshold_scan.csv")
    fig, axes = plt.subplots(1,2,figsize=(7.15,3.25),sharex=True,sharey=True)
    colors = {300:"#246AA3",400:"#D08029",500:"#6C8F43",600:"#854D79"}
    for ax,iou,title in zip(axes,(.5,.75),("a  Mask IoU ≥ 0.50","b  Mask IoU ≥ 0.75")):
        for q in (300,400,500,600):
            rr=sorted((r for r in rows if int(r["Q"])==q and float(r["mask_iou"])==iou),
                      key=lambda r:float(r["recall"]))
            x=[float(r["recall"]) for r in rr]
            y=[float(r["precision"]) for r in rr]
            ax.plot(x,y,"-o",color=colors[q],linewidth=1.4,markersize=2.6,label=f"Q{q}")
            r50=next(r for r in rr if float(r["score_threshold"])==.5)
            ax.scatter([float(r50["recall"])],[float(r50["precision"])],
                       facecolors="white",edgecolors=colors[q],s=34,linewidths=1.2,zorder=5)
        ax.set_title(title,loc="left",fontweight="bold")
        ax.set_xlabel("Recall")
        ax.grid(color="#E3E6E8",linewidth=.5)
        ax.set_xlim(.08,.70);ax.set_ylim(.25,1.0)
    axes[0].set_ylabel("Precision")
    for x,q in ((.59,300),(.68,400),(.77,500),(.86,600)):
        fig.text(x,.975,f"Q{q}",ha="left",va="top",fontsize=7,color=colors[q],fontweight="bold")
    fig.text(.5,.035,"Five measured score thresholds (0.05–0.70); open rings mark score 0.50",
             ha="center",fontsize=7)
    fig.subplots_adjust(left=.09,right=.98,top=.89,bottom=.22,wspace=.13)
    save(fig,"four_budget_threshold_pr")


if __name__ == "__main__":
    stratified()
    threshold_pr()
