"""Analyze cached R50-Q300/Q500 predictions at final-mask level."""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from faster_coco_eval import COCO, COCOeval_faster
import faster_coco_eval.core.mask as mask_utils


MODELS = ("r50_300", "r50_500")
THRESHOLDS = (0.05, 0.10, 0.30, 0.50, 0.70)
IOUS = (0.50, 0.75)
GROUPS = ("gt_le_300", "gt_301_500", "gt_gt_500")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cardinality(adjacency: np.ndarray) -> int:
    if not adjacency.shape[0] or not adjacency.shape[1]:
        return 0
    return int((maximum_bipartite_matching(csr_matrix(adjacency), perm_type="column") >= 0).sum())


def greedy_match(ious: np.ndarray, scores: np.ndarray, det_classes: np.ndarray,
                 gt_classes: np.ndarray, crowd: np.ndarray, threshold: float):
    order = np.argsort(-scores, kind="stable")
    used: set[int] = set()
    matches = np.full(len(scores), -1, dtype=int)
    ignored = np.zeros(len(scores), dtype=bool)
    gt_order = sorted(range(len(gt_classes)), key=lambda index: bool(crowd[index]))
    for det_index in order:
        best, best_iou = -1, min(threshold, 1 - 1e-10)
        for gt_index in gt_order:
            if det_classes[det_index] != gt_classes[gt_index] or (gt_index in used and not crowd[gt_index]):
                continue
            if best >= 0 and not crowd[best] and crowd[gt_index]:
                break
            if ious[det_index, gt_index] < best_iou:
                continue
            best, best_iou = gt_index, ious[det_index, gt_index]
        if best >= 0:
            matches[det_index] = best
            ignored[det_index] = crowd[best]
            if not crowd[best]:
                used.add(best)
    return matches, ignored


def aggregate(rows: list[dict]) -> dict:
    images = len(rows)
    gt = sum(int(row["gt"]) for row in rows)
    tp, fp = sum(int(row["tp"]) for row in rows), sum(int(row["fp"]) for row in rows)
    return {
        "images": images, "gt": gt, "tp": tp, "fp": fp, "fn": gt - tp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / gt if gt else None,
        "candidate_coverage_recall": sum(int(row["candidate_count"]) for row in rows) / gt if gt else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.audit_dir.resolve()
    result_dir = root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert all(manifest["models"][name]["complete"] for name in MODELS)
    assert all(manifest["models"][name]["last_epoch"] == 35 for name in MODELS)
    annotations = manifest["models"]["r50_300"]["annotation"]
    assert manifest["models"]["r50_300"]["annotation_sha256"] == manifest["models"]["r50_500"]["annotation_sha256"]
    coco = COCO(annotations)
    image_ids = sorted(coco.imgs)
    for name in MODELS:
        info = manifest["models"][name]
        assert sorted(info["image_ids"]) == image_ids and len(image_ids) == 570

    per_image: list[dict] = []
    candidate_rows: list[dict] = []
    reproduction: list[dict] = []
    density_rows: list[dict] = []
    per_gt: dict[tuple[int, int, float], dict] = {}
    predictions_by_model: dict[str, list[dict]] = {}

    for name in MODELS:
        q = int(manifest["models"][name]["queries"])
        original_predictions: list[dict] = []
        diagnostic_predictions: list[dict] = []
        by_iou = {iou: [] for iou in IOUS}
        count = {iou: {threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in THRESHOLDS} for iou in IOUS}

        for index, image_id in enumerate(image_ids):
            with gzip.open(root / "cache" / name / f"{image_id}.json.gz", "rt", encoding="utf-8") as stream:
                pred = json.load(stream)
            assert pred["image_id"] == image_id and len(pred["rles"]) == q
            gt = coco.imgToAnns[image_id]
            crowd = np.asarray([bool(annotation.get("iscrowd", 0)) for annotation in gt])
            gt_classes = np.asarray([annotation["category_id"] for annotation in gt])
            valid = ~crowd
            n_gt = int(valid.sum())
            gt_rles = [coco.annToRLE(annotation) for annotation in gt]
            ious = (mask_utils.iou(pred["rles"], gt_rles, crowd.astype(int).tolist())
                    if gt else np.empty((q, 0), dtype=float))
            assert np.isfinite(ious).all()
            scores = np.asarray(pred["scores"], dtype=float)
            det_classes = np.asarray(pred["labels"], dtype=int)
            q_to_mask = {query: position for position, query in enumerate(pred["query_ids"])}
            assert len(q_to_mask) == q

            for position, rle in enumerate(pred["rles"]):
                diagnostic_predictions.append({
                    "image_id": image_id, "category_id": int(det_classes[position]),
                    "score": float(scores[position]), "segmentation": rle,
                })
            for query, category, score in zip(pred["original_query_ids"], pred["original_labels"], pred["original_scores"]):
                original_predictions.append({
                    "image_id": image_id, "category_id": int(category), "score": float(score),
                    "segmentation": pred["rles"][q_to_mask[int(query)]],
                })

            group = "gt_le_300" if n_gt <= 300 else "gt_301_500" if n_gt <= 500 else "gt_gt_500"
            for iou_threshold in IOUS:
                candidate_count = cardinality(ious[:, valid] >= iou_threshold)
                class_aware_count = cardinality(
                    (ious[:, valid] >= iou_threshold)
                    & (det_classes[:, None] == gt_classes[valid][None, :])
                ) if n_gt else 0
                candidate_rows.append({
                    "model": name, "image_id": image_id, "iou": iou_threshold,
                    "gt": n_gt, "candidate_count": candidate_count,
                    "class_aware_candidate_count": class_aware_count,
                })
                matches, ignored = greedy_match(ious, scores, det_classes, gt_classes, crowd, iou_threshold)
                ignored |= ~np.isin(det_classes, coco.getCatIds())
                for threshold in THRESHOLDS:
                    keep = (scores >= threshold) & ~ignored
                    hits = set(matches[keep & (matches >= 0)].tolist())
                    tp = len(hits)
                    fp = int(keep.sum()) - tp
                    fn = n_gt - tp
                    assert tp + fn == n_gt and tp + fp == int(keep.sum())
                    count[iou_threshold][threshold]["tp"] += tp
                    count[iou_threshold][threshold]["fp"] += fp
                    count[iou_threshold][threshold]["fn"] += fn
                    per_image.append({
                        "model": name, "image_id": image_id, "density_group": group,
                        "iou": iou_threshold, "threshold": threshold, "gt": n_gt,
                        "tp": tp, "fp": fp, "fn": fn,
                        "candidate_count": candidate_count,
                    })
                    if threshold == 0.5:
                        for gt_index, annotation in enumerate(gt):
                            if crowd[gt_index]:
                                continue
                            key = (image_id, int(annotation["id"]), iou_threshold)
                            row = per_gt.setdefault(key, {
                                "image_id": image_id, "gt_id": int(annotation["id"]), "iou": iou_threshold,
                                "density_group": group, "image_gt_count": n_gt,
                                "mask_area_pixels_squared": int(mask_utils.area(gt_rles[gt_index])),
                                "category_id": int(annotation["category_id"]),
                            })
                            row[name + "_detected"] = int(gt_index in hits)
            if index % 100 == 0:
                print(f"ANALYSIS_PROGRESS {name} images={index}", flush=True)

        predictions_by_model[name] = original_predictions
        for protocol, predictions in (("original", original_predictions),
                                      ("one_class_per_query_diagnostic", diagnostic_predictions)):
            detections = coco.loadRes(copy.deepcopy(predictions))
            evaluator = COCOeval_faster(coco, detections, "segm")
            evaluator.params.maxDets = [1, 10, 100, 3000]
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            precision = evaluator.eval["precision"]
            valid_precision = precision[:, :, :, 0, -1]
            ap = float(valid_precision[valid_precision >= 0].mean())
            ap50_values = precision[0, :, :, 0, -1]
            ap50 = float(ap50_values[ap50_values >= 0].mean())
            reproduction.append({
                "model": name, "protocol": protocol, "mask_mAP": ap, "mask_AP50": ap50,
                "mask_AP75": float(precision[5, :, :, 0, -1][precision[5, :, :, 0, -1] >= 0].mean()),
                "maxDets": 3000, "queries": q,
            })

        for iou_threshold in IOUS:
            for threshold in THRESHOLDS:
                values = count[iou_threshold][threshold]
                total_gt = sum(row["gt"] for row in per_image
                               if row["model"] == name and row["iou"] == iou_threshold and row["threshold"] == threshold)
                density_rows.append({
                    "model": name, "iou": iou_threshold, "score_threshold": threshold,
                    "gt": total_gt, **values,
                    "precision": values["tp"] / (values["tp"] + values["fp"]) if values["tp"] + values["fp"] else None,
                    "recall": values["tp"] / total_gt if total_gt else None,
                })

    q500_original = next(row for row in reproduction
                         if row["model"] == "r50_500" and row["protocol"] == "original")
    expected_q500 = {"mask_mAP": 0.3132926021241118,
                     "mask_AP50": 0.6164966042202911,
                     "mask_AP75": 0.295862010275716}
    for metric, expected in expected_q500.items():
        if abs(q500_original[metric] - expected) > 1e-4:
            raise AssertionError(f"R50-Q500 terminal {metric} mismatch: "
                                 f"audit={q500_original[metric]} logged={expected}")

    # Image-level density strata summaries at the primary operating point.
    strata: list[dict] = []
    for iou_threshold in IOUS:
        for group in GROUPS:
            for name in MODELS:
                subset = [row for row in per_image if row["model"] == name and row["iou"] == iou_threshold
                          and row["threshold"] == 0.5 and row["density_group"] == group]
                stats = aggregate(subset)
                strata.append({"model": name, "iou": iou_threshold, "score_threshold": 0.5,
                               "density_group": group, **stats})

    paired: list[dict] = []
    rng = np.random.default_rng(20260917)
    for iou_threshold in IOUS:
        low = sorted((row for row in per_image if row["model"] == "r50_300" and row["iou"] == iou_threshold
                      and row["threshold"] == 0.5), key=lambda row: int(row["image_id"]))
        high = sorted((row for row in per_image if row["model"] == "r50_500" and row["iou"] == iou_threshold
                       and row["threshold"] == 0.5), key=lambda row: int(row["image_id"]))
        assert [row["image_id"] for row in low] == [row["image_id"] for row in high]
        gt_counts = np.asarray([row["gt"] for row in low])
        recall_delta = np.asarray([int(b["tp"]) - int(a["tp"]) for a, b in zip(low, high)])
        samples = rng.integers(0, len(low), size=(2000, len(low)))
        boot = recall_delta[samples].sum(axis=1) / gt_counts[samples].sum(axis=1)
        lower, upper = np.quantile(boot, [0.025, 0.975])
        paired.append({
            "comparison": "r50_500_minus_r50_300", "iou": iou_threshold, "threshold": 0.5,
            "delta_micro_recall": float(recall_delta.sum() / gt_counts.sum()),
            "ci95_low_image_bootstrap": float(lower), "ci95_high_image_bootstrap": float(upper),
            "images": len(low), "replicates": 2000, "seed": 20260917,
            "uncertainty_unit": "validation image; not training seed",
        })

    # Pairwise per-GT outcomes describe which masks change detection status.
    paired_gt: list[dict] = []
    for row in per_gt.values():
        low = row.get("r50_300_detected", 0)
        high = row.get("r50_500_detected", 0)
        row["pair_status"] = "both" if low and high else "only500" if high else "only300" if low else "neither"
    for iou_threshold in IOUS:
        for status in ("both", "only500", "only300", "neither"):
            subset = [row for row in per_gt.values() if row["iou"] == iou_threshold and row["pair_status"] == status]
            paired_gt.append({
                "iou": iou_threshold, "pair_status": status, "instances": len(subset),
                "median_mask_area_pixels_squared": float(np.median([row["mask_area_pixels_squared"] for row in subset])) if subset else None,
            })

    write_csv(result_dir / "per_image.csv", per_image)
    write_csv(result_dir / "candidate_coverage.csv", candidate_rows)
    write_csv(result_dir / "reproduction.csv", reproduction)
    write_csv(result_dir / "threshold_metrics.csv", density_rows)
    write_csv(result_dir / "density_summary.csv", strata)
    write_csv(result_dir / "paired_image_bootstrap.csv", paired)
    write_csv(result_dir / "per_gt.csv", list(per_gt.values()))
    write_csv(result_dir / "paired_gt_summary.csv", paired_gt)

    overall = []
    for name in MODELS:
        primary = next(row for row in density_rows if row["model"] == name and row["iou"] == 0.5 and row["score_threshold"] == 0.5)
        ap = next(row for row in reproduction if row["model"] == name and row["protocol"] == "original")
        coverage = [row for row in candidate_rows if row["model"] == name and row["iou"] == 0.5]
        gt_total = sum(row["gt"] for row in coverage)
        overall.append({
            "model": name, "queries": ap["queries"], "mask_mAP": ap["mask_mAP"],
            "mask_AP50": ap["mask_AP50"], "mask_AP75": ap["mask_AP75"],
            "TP": primary["tp"], "FP": primary["fp"], "FN": primary["fn"],
            "precision": primary["precision"], "recall": primary["recall"],
            "candidate_coverage": sum(row["candidate_count"] for row in coverage) / gt_total,
        })
    write_csv(result_dir / "overall.csv", overall)
    bootstrap_by_iou = {row["iou"]: row for row in paired}
    lines = [
        "# R50–300q vs R50–500q 终轮预测审计", "",
        "两组均使用 seed-0、36-epoch terminal EMA checkpoint，并在相同 LIVECell validation 570 张图像上重新推理。", "",
        "## 总体结果", "",
        "| Query | Mask mAP | AP50 | AP75 | TP | FP | FN | Precision | Recall | Candidate coverage |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            f"| {row['queries']} | {row['mask_mAP']:.4f} | {row['mask_AP50']:.4f} | {row['mask_AP75']:.4f} "
            f"| {row['TP']} | {row['FP']} | {row['FN']} | {row['precision']:.4f} "
            f"| {row['recall']:.4f} | {row['candidate_coverage']:.4f} |"
        )
    lines += ["", "## 召回差异与密度分层", ""]
    for iou_threshold in IOUS:
        boot = bootstrap_by_iou[iou_threshold]
        lines.append(
            f"- IoU={iou_threshold:.2f}: R50–500q minus R50–300q micro-recall "
            f"{boot['delta_micro_recall']:+.4f}, image-bootstrap 95% CI "
            f"[{boot['ci95_low_image_bootstrap']:.4f}, {boot['ci95_high_image_bootstrap']:.4f}]."
        )
    lines += ["", "| IoU | GT/image stratum | Q300 recall | Q500 recall | Q300 coverage | Q500 coverage |",
              "|---:|:---|---:|---:|---:|---:|"]
    for iou_threshold in IOUS:
        for group in GROUPS:
            rows = {row["model"]: row for row in strata
                    if row["iou"] == iou_threshold and row["density_group"] == group}
            coverage = {}
            for name in MODELS:
                subset = [row for row in candidate_rows if row["model"] == name and row["iou"] == iou_threshold
                          and ("gt_le_300" if row["gt"] <= 300 else "gt_301_500" if row["gt"] <= 500 else "gt_gt_500") == group]
                denominator = sum(row["gt"] for row in subset)
                coverage[name] = sum(row["candidate_count"] for row in subset) / denominator if denominator else float("nan")
            group_label = {"gt_le_300": "GT≤300", "gt_301_500": "301–500", "gt_gt_500": "GT>500"}[group]
            lines.append(f"| {iou_threshold:.2f} | {group_label} | {rows['r50_300']['recall']:.4f} "
                         f"| {rows['r50_500']['recall']:.4f} | {coverage['r50_300']:.4f} | {coverage['r50_500']:.4f} |")
    lines += ["", "完整阈值扫描见 `threshold_metrics.csv`；逐图、逐 GT 结果分别见 `per_image.csv`、`per_gt.csv`。", "",
              "## 解释边界", "",
              "实际 FN 来自 score-thresholded、同类一对一 mask-IoU 匹配；matcher 未匹配 GT 和 `max(GT−Q,0)` 均不作为预测 FN。Candidate coverage 是全部 query mask 的类别无关几何最大匹配比例，不等于实际召回。图像 bootstrap 反映验证图抽样不确定性，不代表训练 seed 变异。每个设置目前只有一个训练 seed。", ""]
    (result_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    accounting_rows = [row for row in per_image if row["iou"] == 0.5 and row["threshold"] == 0.5]
    assert all(int(row["tp"]) + int(row["fn"]) == int(row["gt"]) for row in accounting_rows)
    print(f"R50_PAIR_AUDIT_COMPLETE models=2 images_per_model=570 "
          f"per_image_accounting_rows={len(accounting_rows)}", flush=True)


if __name__ == "__main__":
    main()
