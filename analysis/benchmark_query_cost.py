"""Benchmark compute cost for the R18 Q300/Q400/Q500/Q600 models.

The benchmark uses one fixed protocol for every query budget:
  * input: one 640x640 RGB image (batch size 1)
  * inference: FP32, torch.inference_mode(), terminal EMA weights
  * latency: data loading excluded; model-only and model+postprocess reported
  * memory: CUDA peak allocated/reserved during model+postprocess
  * FLOPs/MACs: calflops estimate on the model forward only

Results are written as JSON and CSV so the exact measurements can be reused in
the manuscript and independently checked.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def git_commit(project: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_manifest(path: Path, project: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("manifest must contain a non-empty 'models' list")
    resolved = []
    for item in models:
        entry = dict(item)
        for key in ("config", "checkpoint"):
            value = Path(entry[key])
            entry[key] = value if value.is_absolute() else project / value
        resolved.append(entry)
    return resolved


def load_weights(model: torch.nn.Module, checkpoint: Path) -> tuple[str, int | None]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "ema" in state and isinstance(state["ema"], dict) and "module" in state["ema"]:
        weights = state["ema"]["module"]
        source = "ema.module"
    elif "model" in state:
        weights = state["model"]
        source = "model"
    else:
        weights = state
        source = "root"
    if weights and next(iter(weights)).startswith("module."):
        weights = {key.removeprefix("module."): value for key, value in weights.items()}
    model.load_state_dict(weights, strict=True)
    return source, state.get("last_epoch") if isinstance(state, dict) else None


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    named = list(model.named_parameters())
    return {
        "params_total": sum(parameter.numel() for _, parameter in named),
        "params_trainable": sum(parameter.numel() for _, parameter in named if parameter.requires_grad),
        "params_backbone": sum(parameter.numel() for name, parameter in named if "backbone" in name),
        "params_encoder": sum(parameter.numel() for name, parameter in named if "encoder" in name),
        "params_decoder": sum(parameter.numel() for name, parameter in named if "decoder" in name),
    }


def estimate_flops(model: torch.nn.Module) -> dict[str, Any]:
    try:
        from calflops import calculate_flops

        flops, macs, params = calculate_flops(
            model,
            input_shape=(1, 3, 640, 640),
            output_as_string=False,
            print_detailed=False,
            print_results=False,
        )
        return {
            "flops": float(flops),
            "macs": float(macs),
            "calflops_params": int(params),
            "flops_method": "calflops",
            "flops_error": None,
        }
    except Exception as exc:  # retain the failure instead of inventing a value
        return {
            "flops": None,
            "macs": None,
            "calflops_params": None,
            "flops_method": "calflops_failed",
            "flops_error": repr(exc),
        }


def collect_real_inputs(cfg, count: int) -> list[torch.Tensor]:
    cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = 1
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = 0
    cfg.yaml_cfg["val_dataloader"]["dataset"]["return_masks"] = False
    tensors: list[torch.Tensor] = []
    for samples, _targets in cfg.val_dataloader:
        tensors.append(samples.cpu())
        if len(tensors) >= count:
            break
    if len(tensors) != count:
        raise RuntimeError(f"requested {count} inputs but collected {len(tensors)}")
    return tensors


def timed_call(callable_, device: torch.device) -> tuple[Any, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter_ns()
    result = callable_()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    return result, elapsed_ms


def benchmark_runtime(
    model: torch.nn.Module,
    postprocessor: torch.nn.Module,
    inputs: list[torch.Tensor],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    orig_size = torch.tensor([[640, 640]], device=device)

    def full_pipeline(sample: torch.Tensor):
        return postprocessor(model(sample), orig_size)

    first = inputs[0].to(device)
    with torch.inference_mode():
        for _ in range(warmup):
            outputs = model(first)
            _ = postprocessor(outputs, orig_size)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        model_times: list[float] = []
        pipeline_times: list[float] = []
        for _ in range(repeats):
            for cpu_input in inputs:
                sample = cpu_input.to(device)
                outputs, model_ms = timed_call(lambda: model(sample), device)
                del outputs
                pipeline_result, pipeline_ms = timed_call(lambda: full_pipeline(sample), device)
                del pipeline_result
                model_times.append(model_ms)
                pipeline_times.append(pipeline_ms)

    result: dict[str, Any] = {
        "timed_images": len(pipeline_times),
        "model_latency_mean_ms": statistics.fmean(model_times),
        "model_latency_median_ms": statistics.median(model_times),
        "model_latency_p95_ms": percentile(model_times, 0.95),
        "pipeline_latency_mean_ms": statistics.fmean(pipeline_times),
        "pipeline_latency_median_ms": statistics.median(pipeline_times),
        "pipeline_latency_p95_ms": percentile(pipeline_times, 0.95),
        "pipeline_fps": 1000.0 / statistics.fmean(pipeline_times),
    }
    if device.type == "cuda":
        result.update(
            peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20,
        )
    else:
        result.update(peak_allocated_mib=None, peak_reserved_mib=None)
    return result


def environment(device: torch.device) -> dict[str, Any]:
    data: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version(),
        "batch_size": 1,
        "input_shape": [1, 3, 640, 640],
        "dtype": "float32",
    }
    if device.type == "cuda":
        data.update(
            gpu=torch.cuda.get_device_name(device),
            cuda_runtime=torch.version.cuda,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
        )
    return data


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--skip-flops", action="store_true")
    parser.add_argument("--inputs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    project = args.project_root.resolve()
    sys.path.insert(0, str(project))
    os.chdir(project)
    from engine.core import YAMLConfig

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.backends.cudnn.benchmark = False
    models = load_manifest(args.manifest.resolve(), project)
    rows: list[dict[str, Any]] = []
    shared_inputs: list[torch.Tensor] | None = None

    for spec in models:
        config = Path(spec["config"])
        checkpoint = Path(spec["checkpoint"])
        if not config.exists():
            raise FileNotFoundError(config)
        cfg = YAMLConfig(str(config))
        if "PResNet" in cfg.yaml_cfg:
            cfg.yaml_cfg["PResNet"]["pretrained"] = False
        model = cfg.model.eval()
        postprocessor = cfg.postprocessor.eval()
        query_count = int(spec["queries"])
        actual_queries = int(cfg.yaml_cfg["RTDETRTransformerv2"]["num_queries"])
        assert actual_queries == query_count, (actual_queries, query_count)
        assert int(postprocessor.num_top_queries) == query_count

        row: dict[str, Any] = {
            "model": spec["name"],
            "queries": query_count,
            "config": str(config),
            "config_sha256": sha256(config),
            **parameter_counts(model),
        }
        if not args.skip_flops:
            row.update(estimate_flops(model))

        if not args.static_only:
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            source, last_epoch = load_weights(model, checkpoint)
            row.update(
                checkpoint=str(checkpoint),
                checkpoint_sha256=sha256(checkpoint),
                checkpoint_weights=source,
                checkpoint_last_epoch=last_epoch,
            )
            if shared_inputs is None:
                shared_inputs = collect_real_inputs(cfg, args.inputs)
            model = model.to(device).eval()
            postprocessor = postprocessor.to(device).eval()
            row.update(
                benchmark_runtime(
                    model,
                    postprocessor,
                    shared_inputs,
                    device,
                    args.warmup,
                    args.repeats,
                )
            )
            del model, postprocessor
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    payload = {
        "protocol": {
            "input": "real LIVECell validation tensors resized to 640x640",
            "batch_size": 1,
            "data_loading_included": False,
            "precision": "FP32",
            "warmup": args.warmup,
            "unique_inputs": 0 if args.static_only else args.inputs,
            "repeats": 0 if args.static_only else args.repeats,
            "postprocess": "binary mask reconstruction at 640x640",
            "flops_scope": "model forward only",
        },
        "environment": environment(device),
        "project_git_commit": git_commit(project),
        "manifest": str(args.manifest.resolve()),
        "results": rows,
    }
    (args.output_dir / "query_compute_benchmark.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "query_compute_benchmark.csv", rows)
    print(f"BENCHMARK_COMPLETE {len(rows)} models -> {args.output_dir}")


if __name__ == "__main__":
    main()
