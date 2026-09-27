"""Cache terminal R50-Q300/Q500 predictions with every query mask retained."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import faster_coco_eval.core.mask as mask_utils


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


@torch.inference_mode()
def infer(name: str, config: Path, checkpoint: Path, project: Path,
          data: Path, output: Path, device: str, max_batches: int | None) -> None:
    sys.path.insert(0, str(project.resolve()))
    from engine.core import YAMLConfig

    annotation = data / "annotations/livecell_coco_val.json"
    for path in (config, checkpoint, annotation):
        if not path.is_file():
            raise FileNotFoundError(path)

    cfg = YAMLConfig(str(config))
    q = 300 if name == "r50_300" else 500
    assert cfg.yaml_cfg["PResNet"]["depth"] == 50
    assert cfg.yaml_cfg["RTDETRTransformerv2"]["num_queries"] == q
    assert cfg.yaml_cfg["PostProcessor"]["num_top_queries"] == q
    cfg.yaml_cfg["PResNet"]["pretrained"] = False
    cfg.yaml_cfg["val_dataloader"]["dataset"]["return_masks"] = False
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = 4 if device.startswith("cuda") else 0

    model = cfg.model.to(device).eval()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "ema" not in ckpt or "module" not in ckpt["ema"]:
        raise RuntimeError("terminal checkpoint must contain EMA module weights")
    if ckpt.get("last_epoch") != 35:
        raise RuntimeError(f"expected last_epoch=35, got {ckpt.get('last_epoch')}")
    model.load_state_dict(ckpt["ema"]["module"], strict=True)
    post = cfg.postprocessor.to(device).eval()
    assert post.mask_source == "binary" and post.num_top_queries == q

    cache = output / "cache" / name
    cache.mkdir(parents=True, exist_ok=True)
    ids: list[int] = []
    start = time.time()
    for batch_index, (samples, targets) in enumerate(cfg.val_dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        image_ids = [int(target["image_id"].item()) for target in targets]
        ids.extend(image_ids)
        paths = [cache / f"{image_id}.json.gz" for image_id in image_ids]
        if all(path.is_file() for path in paths):
            continue

        samples = samples.to(device)
        outputs = model(samples)
        assert outputs["pred_logits"].shape[1] == q
        for key in ("pred_logits", "pred_boxes", "pred_masks"):
            assert key in outputs and torch.isfinite(outputs[key]).all(), key
        sizes = torch.stack([target["orig_size"] for target in targets]).to(device)
        class_scores = outputs["pred_logits"].sigmoid()
        original_scores, original_flat = class_scores.flatten(1).topk(q, dim=1)
        original_qids = original_flat // post.num_classes
        original_labels = original_flat % post.num_classes

        best_labels = class_scores.argmax(-1)
        one_class_logits = torch.full_like(outputs["pred_logits"], -float("inf"))
        one_class_logits.scatter_(2, best_labels.unsqueeze(-1),
                                  outputs["pred_logits"].gather(2, best_labels.unsqueeze(-1)))
        all_outputs = dict(outputs)
        all_outputs["pred_logits"] = one_class_logits
        _, flat = one_class_logits.sigmoid().flatten(1).topk(q, dim=1)
        query_ids = flat // post.num_classes
        results = post(all_outputs, sizes)

        for batch_index_local, (image_id, result) in enumerate(zip(image_ids, results)):
            rles = []
            for mask in result["masks"]:
                rle = mask_utils.encode(np.asfortranarray(mask[0].numpy().astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("ascii")
                rles.append(rle)
            record = {
                "image_id": image_id,
                "query_ids": query_ids[batch_index_local].tolist(),
                "labels": result["labels"].tolist(),
                "scores": result["scores"].tolist(),
                "rles": rles,
                "original_query_ids": original_qids[batch_index_local].tolist(),
                "original_labels": original_labels[batch_index_local].tolist(),
                "original_scores": original_scores[batch_index_local].tolist(),
            }
            assert len(rles) == q and len(set(record["query_ids"])) == q
            with gzip.open(cache / f"{image_id}.json.gz", "wt", encoding="utf-8") as stream:
                json.dump(record, stream)
        if batch_index % 20 == 0:
            print(f"INFER_PROGRESS {name} batches={batch_index + 1} "
                  f"seconds={time.time() - start:.1f}", flush=True)

    unique_ids = sorted(set(ids))
    complete = max_batches is None and len(unique_ids) == len(cfg.val_dataloader.dataset)
    if max_batches is None:
        assert len(ids) == len(unique_ids) == len(cfg.val_dataloader.dataset)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"models": {}}
    manifest["models"][name] = {
        "queries": q,
        "config": str(config.resolve()),
        "config_sha256": sha256(config),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "last_epoch": ckpt["last_epoch"],
        "weights": "ema.module",
        "annotation": str(annotation.resolve()),
        "annotation_sha256": sha256(annotation),
        "device": device,
        "image_ids": unique_ids,
        "image_count": len(unique_ids),
        "complete": complete,
    }
    atomic_json(manifest_path, manifest)
    print(f"{'CACHE_COMPLETE' if complete else 'SMOKE_COMPLETE'} {name} images={len(unique_ids)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["r50_300", "r50_500"], required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    torch.set_num_threads(4)
    infer(args.model, args.config, args.checkpoint, args.project_root,
          args.data_root, args.output_dir, args.device, args.max_batches)


if __name__ == "__main__":
    main()
