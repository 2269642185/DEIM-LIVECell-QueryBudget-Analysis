"""Read-only four-budget audit derivation from the archived terminal prediction audit."""
from __future__ import annotations

import csv
import os
import ast
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

HERE = Path(os.environ["OUTPUT_ROOT"])
RESEARCH = Path(os.environ["RESEARCH_ROOT"])
SOURCE = Path(os.environ["FOUR_BUDGET_ROOT"])
RESULTS = SOURCE / "results"
TRAIN = RESEARCH / "experiments/training"
CONFIGS = Path(os.environ["MODEL_PROJECT"]) / "configs"
ANNOTATION = Path(os.environ["ANNOTATION"])
STRATA = ("gt_le_300", "gt_301_500", "gt_gt_500")
QS = (300, 400, 500, 600)
THRESHOLDS = (0.05, 0.10, 0.30, 0.50, 0.70)
IOUS = (0.50, 0.75)


def readcsv(name):
    with (RESULTS / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def writecsv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def deep_merge(a, b):
    a = dict(a)
    for k, v in b.items():
        a[k] = deep_merge(a[k], v) if k in a and isinstance(a[k], dict) and isinstance(v, dict) else v
    return a


def resolve_yaml(path):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    includes = data.pop("__include__", [])
    merged = {}
    for inc in includes:
        merged = deep_merge(merged, resolve_yaml((path.parent / inc).resolve()))
    return deep_merge(merged, data)


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    return {prefix: obj}


def main():
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))["models"]
    annotation_hash = digest(ANNOTATION)
    annotation = json.loads(ANNOTATION.read_text(encoding="utf-8"))
    gt_by_image = Counter(a["image_id"] for a in annotation["annotations"] if not a.get("iscrowd", 0))
    ids = set(x["id"] for x in annotation["images"])
    assert len(ids) == 570
    cache_hashes = {}
    for q in QS:
        key = f"r18_{q}"
        cache = SOURCE / "prediction_cache/cache" / key
        files = sorted(cache.glob("*.json.gz"))
        cache_ids = {int(p.name.split(".")[0]) for p in files}
        assert len(files) == len(cache_ids) == 570
        assert cache_ids == ids == set(manifest[key]["image_ids"])
        assert manifest[key]["annotation_sha256"] == annotation_hash
        assert manifest[key]["image_count"] == 570 and manifest[key]["complete"]
        for p in files:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                d = json.load(f)
            assert d["image_id"] == int(p.name.split(".")[0])
            assert len(d["scores"]) == len(d["labels"]) == len(d["rles"]) == q
        cache_hashes[key] = len(files)

    per = readcsv("per_image.csv")
    coverage = readcsv("candidate_coverage.csv")
    threshold_source = readcsv("threshold_metrics.csv")
    budget_source = readcsv("budget_per_image.csv")
    assert len(per) == 4 * 570 * len(IOUS) * len(THRESHOLDS)
    assert len(coverage) == 4 * 570 * len(IOUS)
    assert len(budget_source) == 4 * 570
    assert {r["protocol"] for r in per} == {"diagnostic_one_class_per_query"}
    cov = {(r["model"], int(r["image_id"]), float(r["iou"])): int(r["candidate_count"]) for r in coverage}
    for r in per:
        q = int(r["model"].split("_")[-1]); iid = int(r["image_id"])
        assert int(r["gt"]) == gt_by_image[iid]
        assert int(r["tp"]) + int(r["fn"]) == int(r["gt"])
        assert int(r["tp"]) + int(r["fp"]) >= 0
        assert int(r["candidate_count"]) == cov[(r["model"], iid, float(r["iou"]))]
        assert q in QS

    primary = []
    strata_names = {"gt_le_300": "<=300", "gt_301_500": "301-500", "gt_gt_500": ">500"}
    per_index = defaultdict(list)
    for r in per:
        per_index[(int(r["model"].split("_")[-1]), float(r["iou"]), float(r["threshold"]), r["density_group"])].append(r)
    for group in STRATA:
        for q in QS:
            rr = per_index[(q, 0.5, 0.5, group)]
            gt = sum(int(x["gt"]) for x in rr)
            tp = sum(int(x["tp"]) for x in rr)
            fp = sum(int(x["fp"]) for x in rr)
            fn = sum(int(x["fn"]) for x in rr)
            cand = sum(int(x["candidate_count"]) for x in rr)
            assert gt == tp + fn
            primary.append(dict(stratum=strata_names[group], num_images=len(rr), Q=q, GT=gt, TP=tp, FP=fp, FN=fn,
                                precision=tp/(tp+fp), recall=tp/gt, candidate_coverage=cand/gt,
                                candidate_matched_gt=cand, aggregation="micro_ratio_of_sums"))
    assert [(x["stratum"], x["num_images"]) for x in primary if x["Q"] == 300] == [("<=300",362),("301-500",103),(">500",105)]
    writecsv(HERE / "analysis/four_budget_stratified_primary.csv", primary)

    pub = []
    for group in ("<=300", "301-500", ">500"):
        for metric in ("recall", "candidate_coverage", "precision"):
            row = {"stratum": group, "metric": metric}
            row.update({f"Q{q}": next(r[metric] for r in primary if r["stratum"] == group and r["Q"] == q) for q in QS})
            pub.append(row)
    writecsv(HERE / "analysis/four_budget_stratified_publication_table.csv", pub)

    threshold = []
    for q in QS:
        for iou in IOUS:
            for t in THRESHOLDS:
                rr = [r for g in STRATA for r in per_index[(q,iou,t,g)]]
                tp = sum(int(r["tp"]) for r in rr)
                fp = sum(int(r["fp"]) for r in rr)
                fn = sum(int(r["fn"]) for r in rr)
                gt = sum(int(r["gt"]) for r in rr)
                assert len(rr) == 570 and tp+fn == gt == 181610
                p = tp/(tp+fp); rec = tp/gt
                row = dict(Q=q, score_threshold=t, mask_iou=iou, TP=tp, FP=fp, FN=fn,
                           precision=p, recall=rec, f1=2*p*rec/(p+rec))
                original = next(x for x in threshold_source if x["model"] == f"r18_{q}" and float(x["iou"]) == iou and float(x["threshold"]) == t)
                assert (tp,fp,fn) == tuple(int(original[k]) for k in ("tp","fp","fn"))
                threshold.append(row)
    writecsv(HERE / "analysis/four_budget_threshold_scan.csv", threshold)

    exposure = []
    for q in QS:
        rr = [x for x in budget_source if x["model"] == f"r18_{q}"]
        assert len(rr) == 570 and {int(x["image_id"]) for x in rr} == ids
        assert all(int(x["gt"]) == gt_by_image[int(x["image_id"])] and int(x["budget_gap"]) == max(int(x["gt"])-q,0) for x in rr)
        n = sum(int(x["gt"]) > q for x in rr)
        gap = sum(int(x["budget_gap"]) for x in rr)
        exposure.append(dict(Q=q, validation_images=570, images_gt_exceeds_q=n, image_fraction=n/570,
                             total_GT=181610, arithmetic_gap=gap, arithmetic_excess_GT_fraction=gap/181610))
    writecsv(HERE / "analysis/query_budget_exposure.csv", exposure)

    matched = readcsv("recall_at_precision.csv")
    rows = [dict(Q=int(r["model"].split("_")[-1]), mask_iou=float(r["iou"]), target_precision=float(r["target_precision"]),
                 achieved_precision=float(r["achieved_precision"]), recall=float(r["recall_at_precision"]), score=float(r["score"]), method=r["method"])
            for r in matched]
    assert set(r["target_precision"] for r in rows) == {0.8,0.9}
    writecsv(HERE / "analysis/four_budget_common_precision_existing.csv", rows)

    rng = np.random.default_rng(20260924)
    boot = []
    for group in STRATA:
        by_q = {q: {int(r["image_id"]): r for r in per_index[(q,0.5,0.5,group)]} for q in QS}
        group_ids = sorted(by_q[300])
        assert all(set(by_q[q]) == set(group_ids) for q in QS)
        gt = np.array([int(by_q[300][i]["gt"]) for i in group_ids])
        for q in (400,500,600):
            delta = np.array([int(by_q[q][i]["tp"]) - int(by_q[300][i]["tp"]) for i in group_ids])
            index = rng.integers(0,len(group_ids),size=(2000,len(group_ids)))
            values = delta[index].sum(axis=1)/gt[index].sum(axis=1)
            lo,hi = np.quantile(values,[.025,.975])
            boot.append(dict(stratum=strata_names[group], comparison=f"Q{q}-Q300", delta_micro_recall=delta.sum()/gt.sum(),
                             ci95_low=lo,ci95_high=hi,replicates=2000,seed=20260924,
                             uncertainty="paired validation-image sampling; not training-seed uncertainty"))
    writecsv(HERE / "analysis/four_budget_stratified_bootstrap.csv", boot)

    roots = {
        300: CONFIGS / "deim_sdf/exp_binary_mask_36e.yml",
        400: RESEARCH / "experiments/config_snapshots/query_curve_20260913/exp_p2_query400_36e.yml",
        500: CONFIGS / "deim_sdf/exp_query500_36e.yml",
        600: RESEARCH / "experiments/training/LIVECell_R18_Q600_S0_36E/exp_p2_query600_36e.yml",
    }
    # Config snapshots live outside the original configs/ tree; resolve their shared includes against that tree.
    resolved = {}
    for q,path in roots.items():
        if q in (400,600):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            base = {}
            for inc in raw.pop("__include__",[]):
                base = deep_merge(base, resolve_yaml((CONFIGS / "deim_sdf" / inc).resolve()))
            resolved[q] = deep_merge(base,raw)
        else:
            resolved[q] = resolve_yaml(path)
    full_logs = {
        300: HERE / "audit/server_check_20260924/exp_binary_mask_36e_retrain.log",
        400: TRAIN / "LIVECell_R18_Q400_S0_36E/provenance/paper_workspace_export/server_training.log",
        500: HERE / "audit/server_check_20260924/exp_query500_36e_retrain.log",
        600: TRAIN / "LIVECell_R18_Q600_S0_36E/exp_p2_query600_36e_run.log",
    }
    if all(p.exists() for p in full_logs.values()):
        logged = {}
        for q,path in full_logs.items():
            with path.open(encoding="utf-8", errors="replace") as f:
                line = next(x for x in f if x.startswith("cfg:"))
            logged[q] = ast.literal_eval(line[5:].strip())
        flat = {q:flatten(x) for q,x in logged.items()}
        scope = "four historical full training-log launch cfg records (verified Q300/Q500 server logs, Q400/Q600 archived logs)"
    else:
        flat = {q:flatten(x) for q,x in resolved.items()}
        scope = "fallback reconstructed merge of archived root YAMLs with current shared includes; not original resolved training records"
    all_keys = sorted(set().union(*(set(x) for x in flat.values())))
    allowed = {"output_dir", "RTDETRTransformerv2.num_queries", "PostProcessor.num_top_queries",
               "yaml_cfg.output_dir", "yaml_cfg.config", "yaml_cfg.RTDETRTransformerv2.num_queries",
               "yaml_cfg.PostProcessor.num_top_queries"}
    diffs = []
    for k in all_keys:
        vals = {f"Q{q}":flat[q].get(k,"<ABSENT>") for q in QS}
        if len({json.dumps(x,sort_keys=True) for x in vals.values()}) > 1:
            diffs.append(dict(path=k,classification="allowed" if k in allowed else "unexpected_or_unproven",values=vals))
    obj = dict(scope=scope,full_training_logs={f"Q{q}":str(p) for q,p in full_logs.items()},
               root_configs={f"Q{q}":str(p) for q,p in roots.items()},differences=diffs,
               annotation_sha256=annotation_hash,cache_files=cache_hashes)
    (HERE / "audit").mkdir(exist_ok=True)
    (HERE / "audit/four_budget_config_diff.json").write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf-8")
    md = ["# Four-budget normalized config diff", "",obj["scope"],"","| Field | Class | Q300 | Q400 | Q500 | Q600 |","|---|---|---|---|---|---|"]
    for d in diffs:
        vals = [str(d["values"][f"Q{q}"]).replace("|","\\|") for q in QS]
        md.append("| "+" | ".join([d["path"],d["classification"],*vals])+" |")
    (HERE / "audit/four_budget_config_diff.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print(json.dumps(dict(annotation_sha256=annotation_hash,cache_files=cache_hashes,stratified_rows=len(primary),threshold_rows=len(threshold),config_differences=diffs,exposure=exposure),indent=2))


if __name__ == "__main__":
    main()
