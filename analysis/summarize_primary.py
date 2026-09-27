import csv, os
from pathlib import Path

import numpy as np


ROOT = Path(os.environ["AUDIT_ROOT"])
rows = list(csv.DictReader((ROOT / "per_image.csv").open(encoding="utf-8")))
models = ["r18_300", "r18_500"]
ious = ["0.5", "0.75"]
thresholds = ["0.05", "0.1", "0.3", "0.5", "0.7"]
groups = ["gt_le_300", "gt_301_500", "gt_gt_500"]


def subset(model, iou, threshold, group=None):
    return [r for r in rows if r["model"] == model and r["iou"] == iou and r["threshold"] == threshold and (group is None or r["density_group"] == group)]


def metrics(rs):
    gt = sum(int(r["gt"]) for r in rs)
    tp = sum(int(r["tp"]) for r in rs)
    fp = sum(int(r["fp"]) for r in rs)
    fn = sum(int(r["fn"]) for r in rs)
    return {"images": len(rs), "gt": gt, "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else "NA",
            "recall": tp / gt if gt else "NA",
            "mean_image_recall": float(np.mean([int(r["tp"]) / int(r["gt"]) for r in rs if int(r["gt"])])) if gt else "NA",
            "fn_per_image": fn / len(rs) if rs else "NA",
            "fp_per_image": fp / len(rs) if rs else "NA",
            "candidate_coverage_recall": sum(int(r["candidate_count"]) for r in rs) / gt if gt else "NA"}


def write(name, data):
    if not data:
        return
    with (ROOT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


summary = []
density = []
for model in models:
    for iou in ious:
        for threshold in thresholds:
            summary.append({"model": model, "iou": float(iou), "threshold": float(threshold), **metrics(subset(model, iou, threshold))})
        for group in groups:
            density.append({"model": model, "iou": float(iou), "threshold": 0.5, "density_group": group, **metrics(subset(model, iou, "0.5", group))})
write("threshold_metrics.csv", summary)
write("density_summary.csv", density)

diffs = []
for iou in ious:
    for group in groups:
        a = next(x for x in density if x["model"] == "r18_300" and x["iou"] == float(iou) and x["density_group"] == group)
        b = next(x for x in density if x["model"] == "r18_500" and x["iou"] == float(iou) and x["density_group"] == group)
        diffs.append({"iou": float(iou), "density_group": group, **{k: float(b[k]) - float(a[k]) for k in ["precision", "recall", "mean_image_recall", "fn_per_image", "fp_per_image", "candidate_coverage_recall"]}})
write("density_differences.csv", diffs)

boot = []
rng = np.random.default_rng(20260904)
for iou in ious:
    a = sorted(subset("r18_300", iou, "0.5"), key=lambda r: int(r["image_id"]))
    b = sorted(subset("r18_500", iou, "0.5"), key=lambda r: int(r["image_id"]))
    gt = np.array([int(r["gt"]) for r in a])
    delta = np.array([int(y["tp"]) - int(x["tp"]) for x, y in zip(a, b)])
    idx = rng.integers(0, len(gt), (2000, len(gt)))
    values = delta[idx].sum(1) / gt[idx].sum(1)
    lo, hi = np.quantile(values, [0.025, 0.975])
    boot.append({"comparison": "r18_500_minus_r18_300", "iou": float(iou), "threshold": 0.5,
                 "delta_micro_recall": float(delta.sum() / gt.sum()), "ci95_low": float(lo), "ci95_high": float(hi),
                 "replicates": 2000, "seed": 20260904})
write("paired_bootstrap.csv", boot)

budget = []
for model in models:
    rs = {int(r["image_id"]): int(r["gt"]) for r in rows if r["model"] == model and r["iou"] == "0.5" and r["threshold"] == "0.5"}
    q = 300 if model == "r18_300" else 500
    gaps = np.array([max(g - q, 0) for g in rs.values()])
    g = np.array(list(rs.values()))
    budget.append({"model": model, "images": len(g), "gt": int(g.sum()), "mean_gt": float(g.mean()), "median_gt": float(np.median(g)),
                   "max_gt": int(g.max()), "budget_gap_sum": int(gaps.sum()), "budget_gap_mean": float(gaps.mean()),
                   "budget_gap_median": float(np.median(gaps)), "affected_images": int((gaps > 0).sum()),
                   "affected_image_fraction": float((gaps > 0).mean()), "budget_gap_fraction_of_gt": float(gaps.sum() / g.sum())})
write("budget_summary.csv", budget)
print("SUMMARY_REBUILT", len(summary), len(density), len(boot), len(diffs), len(budget))
