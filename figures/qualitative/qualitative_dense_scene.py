"""Create a qualitative Q300/Q500 comparison from saved predictions only.

The script reads the released LIVECell validation images/annotations and the
archived R18 prediction caches. It performs the same score/IoU matching used
by the audit, selects one representative image per instance-count stratum,
and writes a provenance JSON beside the figure.
"""
from pathlib import Path
import csv, gzip, json, sys, os
import numpy as np
from PIL import Image, ImageDraw
import tifffile
from pycocotools import mask as mask_utils
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.5,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from alignment_optional import require_matplotlib_panel_alignment

PROJECT = Path(os.environ["PROJECT_ROOT"])
CODE = PROJECT / "research/code/deim_livecell"
AUDIT = PROJECT / "research/experiments/audits/LIVECell_R18_P0_prediction_and_target_20260912"
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
ANN = CODE / "datasets/livecell/annotations/livecell_coco_val.json"
IMG_ROOT = CODE / "datasets/livecell/images/livecell_train_val_images"

def decode_rle(rle):
    rle = dict(rle)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("ascii")
    return np.asarray(mask_utils.decode(rle), dtype=np.uint8)

def polygon_mask(ann, h, w):
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, h, w)
        return np.asarray(mask_utils.decode(rles).max(axis=2), dtype=np.uint8)
    return decode_rle(seg)

def bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0: return (0, 0, 0, 0)
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)

def iou(a, b, ba, bb):
    x0, y0 = max(ba[0], bb[0]), max(ba[1], bb[1])
    x1, y1 = min(ba[2], bb[2]), min(ba[3], bb[3])
    if x1 <= x0 or y1 <= y0: return 0.0
    aa = a[y0:y1, x0:x1]; bbm = b[y0:y1, x0:x1]
    inter = np.logical_and(aa, bbm).sum()
    union = np.logical_or(aa, bbm).sum()
    return float(inter / union) if union else 0.0

def load_cache(model, image_id):
    p = AUDIT / "cache" / model / f"{image_id}.json.gz"
    with gzip.open(p, "rt", encoding="utf-8") as f: return json.load(f)

def match(cache, gt_masks, threshold=0.5):
    preds = []
    h, w = cache["size_wh"][1], cache["size_wh"][0]
    for score, rle in zip(cache["scores"], cache["rles"]):
        if score < threshold: continue
        m = decode_rle(rle); preds.append((float(score), m, bbox(m)))
    preds.sort(key=lambda x: x[0], reverse=True)
    gt_boxes = [bbox(m) for m in gt_masks]
    used = set(); pairs = []; unmatched = []
    for pi, (_, pm, pb) in enumerate(preds):
        best, best_iou = -1, 0.0
        for gi, (gm, gb) in enumerate(zip(gt_masks, gt_boxes)):
            if gi in used: continue
            val = iou(pm, gm, pb, gb)
            if val > best_iou: best, best_iou = gi, val
        if best_iou >= 0.5:
            used.add(best); pairs.append((pi, best, best_iou))
        else:
            unmatched.append(pi)
    return preds, pairs, unmatched

def contour(ax, mask, color, lw=0.65, alpha=0.65):
    ax.contour(mask, levels=[0.5], colors=[color], linewidths=lw, alpha=alpha)

def choose_roi(gt_masks, recovered, image_shape):
    """Choose a deterministic ROI around the largest Q500-recovered GT mask."""
    if not recovered:
        raise RuntimeError("No recovered GT instance available for ROI selection")
    gi = max(recovered, key=lambda k: int(gt_masks[k].sum()))
    ys, xs = np.where(gt_masks[gi])
    h, w = image_shape[:2]
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    side = max(x1 - x0, y1 - y0) * 2.8 + 32
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx0 = max(0, int(round(cx - side / 2)))
    ry0 = max(0, int(round(cy - side / 2)))
    rx1 = min(w, int(round(cx + side / 2)))
    ry1 = min(h, int(round(cy + side / 2)))
    return (rx0, ry0, rx1, ry1), gi

def crop_contour(ax, mask, roi, color, lw=0.9, alpha=0.95):
    x0, y0, x1, y1 = roi
    crop = mask[y0:y1, x0:x1]
    if crop.any():
        ax.contour(crop, levels=[0.5], colors=[color], linewidths=lw, alpha=alpha)

def add_roi_inset(ax, image, gt_masks, preds, matched, unmatched, roi, panel):
    """Add a real-data zoom inset to one of the three prediction panels."""
    x0, y0, x1, y1 = roi
    ins = inset_axes(ax, width="43%", height="43%", loc="lower left",
                     borderpad=0.35)
    ins.imshow(image[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
    ins.set_xticks([]); ins.set_yticks([])
    for spine in ins.spines.values():
        spine.set_color("white"); spine.set_linewidth(1.0)
    ins.text(0.03, 0.96, "ROI", transform=ins.transAxes, ha="left", va="top",
             fontsize=6.5, color="white", weight="bold",
             bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.2, "edgecolor": "none"})
    if panel == 0:
        for _, gi, _ in matched:
            crop_contour(ins, gt_masks[gi], roi, "#39B54A")
    elif panel == 1:
        for gi in unmatched:
            crop_contour(ins, gt_masks[gi], roi, "#E64B35", lw=0.8)
    else:
        for _, gi, _ in matched:
            crop_contour(ins, gt_masks[gi], roi, "#39B54A")
        for pi in unmatched[:3]:
            crop_contour(ins, preds[pi][1], roi, "#F2A900", lw=1.0)

with ANN.open("r", encoding="utf-8") as f: coco = json.load(f)
images = {x["id"]: x for x in coco["images"]}
anns = {}
for a in coco["annotations"]: anns.setdefault(a["image_id"], []).append(a)

rows = []
with (AUDIT / "per_image.csv").open("r", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["protocol"] == "diagnostic_one_class_per_query" and r["iou"] == "0.5" and r["threshold"] == "0.5":
            rows.append(r)
by_image = {}
for r in rows: by_image.setdefault(int(r["image_id"]), {})[r["model"]] = r

def group(n):
    if n <= 300: return "N_GT ≤ 300"
    if n <= 500: return "301 ≤ N_GT ≤ 500"
    return "N_GT > 500"

selected = {}
for image_id, r in by_image.items():
    if "r18_300" not in r or "r18_500" not in r: continue
    n = int(r["r18_500"]["gt"])
    g = group(n)
    if g == "N_GT > 500" and n > 900: continue
    score = int(r["r18_500"]["tp"]) - int(r["r18_300"]["tp"])
    candidate = (score, -abs(n - (180 if g.startswith("N_GT ≤") else 400 if g.startswith("301") else 650)), image_id)
    if g not in selected or candidate > selected[g][0]: selected[g] = (candidate, image_id)
selected_ids = {g: v[1] for g, v in selected.items()}
if len(selected_ids) != 3: raise RuntimeError(f"Could not select all strata: {selected_ids}")

fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.0))
fig.subplots_adjust(left=0.12, right=0.985, top=0.94, bottom=0.09,
                    wspace=0.035, hspace=0.12)
col_titles = ["Ground truth", "Q300", "Q500"]
for j, title in enumerate(col_titles): axes[0, j].set_title(title, fontsize=11, weight="bold")
provenance = {"selection": {}, "source": {"annotation": str(ANN), "image_root": str(IMG_ROOT), "r18_q300_cache": str(AUDIT / "cache/r18_300"), "r18_q500_cache": str(AUDIT / "cache/r18_500")}}

for i, (stratum, image_id) in enumerate(selected_ids.items()):
    info = images[image_id]; path = IMG_ROOT / info["file_name"]
    # LIVECell files are LZW-compressed TIFFs; tifffile uses imagecodecs,
    # whereas PIL's native TIFF path can abort on this archive.
    image = tifffile.imread(path).astype(np.uint8)
    gt_masks = [polygon_mask(a, info["height"], info["width"]) for a in anns[image_id] if not a.get("iscrowd", 0)]
    c300 = load_cache("r18_300", image_id); c500 = load_cache("r18_500", image_id)
    p300, m300, u300 = match(c300, gt_masks); p500, m500, u500 = match(c500, gt_masks)
    gt300 = {g for _, g, _ in m300}; gt500 = {g for _, g, _ in m500}
    recovered = sorted(gt500 - gt300)
    extra_fp = u500[:3]
    roi, roi_gt = choose_roi(gt_masks, recovered, image.shape)
    provenance["selection"][stratum] = {"image_id": image_id, "file_name": info["file_name"], "gt": len(gt_masks), "q300_tp": len(m300), "q500_tp": len(m500), "recovered_gt": len(recovered), "q500_additional_fp_shown": len(extra_fp), "roi_gt_index": roi_gt, "roi_xyxy": list(roi), "roi_rule": "largest-area Q500-recovered GT instance, with deterministic square context margin"}

    for j in range(3):
        ax = axes[i, j]; ax.imshow(image, cmap="gray", vmin=0, vmax=255, aspect="auto"); ax.axis("off")
        ax.add_patch(Rectangle((roi[0], roi[1]), roi[2] - roi[0], roi[3] - roi[1],
                               fill=False, edgecolor="#FFFFFF", linewidth=0.8, alpha=0.9))
    # Keep the figure readable for dense images: all GT instances are shown as
    # small centers, while only audit-defining instances receive outlines.
    centers = np.array([[np.mean(np.where(m)[1]), np.mean(np.where(m)[0])] for m in gt_masks])
    axes[i, 0].scatter(centers[:, 0], centers[:, 1], s=2, c="#BFD7EA", alpha=.45, linewidths=0)
    for gi in recovered: contour(axes[i, 0], gt_masks[gi], "#39B54A", 0.9, 0.95)
    missed = sorted(set(range(len(gt_masks))) - gt300)
    for gi in missed: contour(axes[i, 1], gt_masks[gi], "#E64B35", 0.75, 0.9)
    for pi, gi, _ in m300:
        if gi in recovered: contour(axes[i, 1], p300[pi][1], "#4C78A8", 0.7, 0.8)
    for gi in recovered: contour(axes[i, 2], gt_masks[gi], "#39B54A", 0.95, 0.95)
    for pi in extra_fp: contour(axes[i, 2], p500[pi][1], "#F2A900", 1.1, 1.0)
    add_roi_inset(axes[i, 0], image, gt_masks, p500, m500, [], roi, 0)
    add_roi_inset(axes[i, 1], image, gt_masks, p300, m300, missed, roi, 1)
    add_roi_inset(axes[i, 2], image, gt_masks, p500, m500, extra_fp, roi, 2)
    axes[i, 0].set_ylabel(f"{stratum}\nID {image_id}", fontsize=9, weight="bold")

fig.text(0.5, 0.012, "Full images with white ROI boxes; insets show the same real GT/Q300/Q500 evidence. Green: Q500-recovered GT; red: GT missed by Q300; orange: selected additional Q500 FP.", ha="center", fontsize=8.1)
require_matplotlib_panel_alignment(fig, axes=axes.ravel().tolist(), json_out=str(OUT / "fig_qualitative_dense_scene.alignment.json"), overlay_svg=str(OUT / "fig_qualitative_dense_scene.alignment.svg"), tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
fig.savefig(OUT / "fig_qualitative_dense_scene.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_qualitative_dense_scene.svg", bbox_inches="tight")
fig.savefig(OUT / "fig_qualitative_dense_scene.tiff", dpi=600, bbox_inches="tight")
fig.savefig(OUT / "fig_qualitative_dense_scene.png", dpi=300, bbox_inches="tight")
(OUT / "fig_qualitative_dense_scene.provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
print(json.dumps(provenance, indent=2))
