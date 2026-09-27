"""Cache all-query binary masks using external dataset, checkpoints and compatible model code.
Output location is controlled by --output-dir; no model weights are included.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import faster_coco_eval.core.mask as mask_utils


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


@torch.inference_mode()
def run_model(name: str, max_batches: int | None, project: Path,
              data: Path, out: Path, checkpoint: Path | None, device: str) -> None:
    spec = {
        "r18_300": {"queries": 300, "config": project / "configs/deim_sdf/exp_binary_mask_36e.yml"},
        "r18_500": {"queries": 500, "config": project / "configs/deim_sdf/exp_query500_36e.yml"},
        "r18_400": {"queries": 400, "config": project / "configs/deim_sdf/exp_p2_query400_36e.yml"},
        "r18_600": {"queries": 600, "config": project / "configs/deim_sdf/exp_p2_query600_36e.yml"},
    }[name]
    spec["checkpoint"] = checkpoint or (HERE / f"p2_query_results/q{name[-3:]}/checkpoint0035.pth")
    global OUT
    OUT = out
    sys.path.insert(0, str(project))
    from engine.core import YAMLConfig

    q = spec["queries"]
    for required in (spec["config"], spec["checkpoint"], data / "annotations/livecell_coco_val.json"):
        if not required.exists():
            raise FileNotFoundError(required)

    cfg = YAMLConfig(str(spec["config"]))
    if "PResNet" in cfg.yaml_cfg:
        cfg.yaml_cfg["PResNet"]["pretrained"] = False
    cfg.yaml_cfg["val_dataloader"]["dataset"]["return_masks"] = False
    cfg.yaml_cfg["val_dataloader"]["dataset"]["img_folder"] = str(data / "images/livecell_train_val_images")
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = 4 if device.startswith("cuda") else 0

    model = cfg.model.to(device).eval()
    ckpt = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    weights = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    model.load_state_dict(weights, strict=True)
    post = cfg.postprocessor.to(device).eval()
    assert post.mask_source == "binary", post.mask_source
    assert post.num_top_queries == q, post.num_top_queries

    loader = cfg.val_dataloader
    outdir = OUT / "cache" / name
    outdir.mkdir(parents=True, exist_ok=True)
    ids = []
    start = time.time()
    for bi, (samples, targets) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        image_ids = [int(t["image_id"].item()) for t in targets]
        ids.extend(image_ids)
        if all((outdir / f"{iid}.json.gz").exists() for iid in image_ids):
            continue

        samples = samples.to(device)
        outputs = model(samples)
        assert outputs["pred_logits"].shape[1] == q
        for key in ("pred_logits", "pred_boxes", "pred_masks"):
            assert key in outputs and torch.isfinite(outputs[key]).all(), key

        sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
        class_scores = outputs["pred_logits"].sigmoid()
        original_scores, original_flat = class_scores.flatten(1).topk(q, dim=1)
        original_qids = original_flat // post.num_classes
        original_labels = original_flat % post.num_classes

        # For candidate coverage, retain one best class per query and all Q queries.
        best_scores, best_labels = class_scores.max(-1)
        one_class_logits = torch.full_like(outputs["pred_logits"], -float("inf"))
        one_class_logits.scatter_(2, best_labels.unsqueeze(-1),
                                  outputs["pred_logits"].gather(2, best_labels.unsqueeze(-1)))
        all_outputs = dict(outputs)
        all_outputs["pred_logits"] = one_class_logits
        _, flat = one_class_logits.sigmoid().flatten(1).topk(q, dim=1)
        qids = flat // post.num_classes
        results = post(all_outputs, sizes)

        for b, (iid, result) in enumerate(zip(image_ids, results)):
            masks = result["masks"]
            rles = []
            for mask in masks:
                rle = mask_utils.encode(np.asfortranarray(mask[0].numpy().astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("ascii")
                rles.append(rle)
            record = {
                "image_id": iid,
                "query_ids": qids[b].tolist(),
                "labels": result["labels"].tolist(),
                "scores": result["scores"].tolist(),
                "rles": rles,
                "original_query_ids": original_qids[b].tolist(),
                "original_labels": original_labels[b].tolist(),
                "original_scores": original_scores[b].tolist(),
                "size_wh": sizes[b].tolist(),
            }
            assert len(record["rles"]) == q
            assert len(set(record["query_ids"])) == q
            with gzip.open(outdir / f"{iid}.json.gz", "wt", encoding="utf-8") as f:
                json.dump(record, f)

        if bi % 10 == 0:
            print(name, bi + 1, "batches", round(time.time() - start, 1), "seconds", flush=True)

    unique_ids = sorted(set(ids))
    if max_batches is None:
        expected = len(loader.dataset)
        assert len(ids) == expected and len(unique_ids) == expected, (len(ids), len(unique_ids), expected)
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"models": {}}
    manifest["models"][name] = {
        "queries": q,
        "config": str(spec["config"]),
        "checkpoint": str(spec["checkpoint"]),
        "checkpoint_sha256": sha256(spec["checkpoint"]),
        "annotation": str(data / "annotations/livecell_coco_val.json"),
        "annotation_sha256": sha256(data / "annotations/livecell_coco_val.json"),
        "last_epoch": ckpt.get("last_epoch"),
        "weights": "ema.module" if "ema" in ckpt else "model",
        "device": device,
        "image_ids": unique_ids,
        "image_count_seen": len(unique_ids),
        "complete": max_batches is None and len(unique_ids) == len(loader.dataset),
    }
    write_json(manifest_path, manifest)
    print("CACHE_COMPLETE" if manifest["models"][name]["complete"] else "SMOKE_COMPLETE",
          name, len(unique_ids), "images", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["r18_300", "r18_400", "r18_500", "r18_600"], required=True)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--project-root", type=Path, default=Path('<MODEL_PROJECT>'))
    parser.add_argument("--data-root", type=Path, default=Path('<DATASET_ROOT>'))
    parser.add_argument("--output-dir", type=Path, default=HERE / "q400_q600_terminal_audit")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run_model(args.model, args.max_batches, args.project_root, args.data_root,
              args.output_dir, args.checkpoint, args.device)


if __name__ == "__main__":
    main()
