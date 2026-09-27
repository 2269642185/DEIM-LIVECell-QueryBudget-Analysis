# Third-party notices

## DEIM

Upstream: [Intellindust-AI-Lab/DEIM](https://github.com/Intellindust-AI-Lab/DEIM).
License: Apache License 2.0.

Copyright (C) INTELLINDUST INFORMATION TECHNOLOGY (SHENZHEN) CO., LTD. and all its affiliates.

Original upstream copyright and complete license text are retained verbatim in `provenance/DEIM_LICENSE.txt`. Root `LICENSE` is the unmodified standard Apache License 2.0 text obtained from https://www.apache.org/licenses/LICENSE-2.0.txt . No upstream author name or copyright notice has been removed.

### Material originating from DEIM

`provenance/DEIM_LICENSE.txt` is the original notice/license copy. Shared configuration scaffolding under `configs/replay_tree/base/`, `configs/replay_tree/deim_rtdetrv2/`, `configs/replay_tree/dataset/`, and `configs/replay_tree/runtime.yml` originates from the local DEIM tree. These release copies have notices/comment or path modifications; they are not claimed to be pristine historical upstream versions.

### Modified or derived DEIM material

All configuration YAML files and serialized configuration records listed below are conservatively treated as DEIM-derived. Study-specific configurations alter experiment settings; the release pass only adds attribution and modification notices. YAML parsed values are unchanged. `provenance/det_engine_recovery_diff.patch` contains study recovery modifications and upstream context; patch hunks are preserved. JSON files remain byte-identical and carry adjacent same-name `.NOTICE.txt` files. These records do not prove historical training-source identity.

| File | Modification/attribution notice |
|---|---|
| `configs/inference_resolved/primary_r18_r18_300.json` | `configs/inference_resolved/primary_r18_r18_300.json.NOTICE.txt` |
| `configs/inference_resolved/primary_r18_r18_500.json` | `configs/inference_resolved/primary_r18_r18_500.json.NOTICE.txt` |
| `configs/r18_records/q300_launch.yml` | `configs/r18_records/q300_launch.yml` |
| `configs/r18_records/q300_logged_cfg.json` | `configs/r18_records/q300_logged_cfg.json.NOTICE.txt` |
| `configs/r18_records/q400_launch.yml` | `configs/r18_records/q400_launch.yml` |
| `configs/r18_records/q400_logged_cfg.json` | `configs/r18_records/q400_logged_cfg.json.NOTICE.txt` |
| `configs/r18_records/q500_launch.yml` | `configs/r18_records/q500_launch.yml` |
| `configs/r18_records/q500_logged_cfg.json` | `configs/r18_records/q500_logged_cfg.json.NOTICE.txt` |
| `configs/r18_records/q600_launch.yml` | `configs/r18_records/q600_launch.yml` |
| `configs/r18_records/q600_logged_cfg.json` | `configs/r18_records/q600_logged_cfg.json.NOTICE.txt` |
| `configs/r50_records/exp_r50_query300_36e_clean.yml` | `configs/r50_records/exp_r50_query300_36e_clean.yml` |
| `configs/r50_records/exp_r50_query500_36e.yml` | `configs/r50_records/exp_r50_query500_36e.yml` |
| `configs/replay_tree/base/dataloader.yml` | `configs/replay_tree/base/dataloader.yml` |
| `configs/replay_tree/base/rt_deim.yml` | `configs/replay_tree/base/rt_deim.yml` |
| `configs/replay_tree/base/rt_optimizer.yml` | `configs/replay_tree/base/rt_optimizer.yml` |
| `configs/replay_tree/base/rtdetrv2_r50vd.yml` | `configs/replay_tree/base/rtdetrv2_r50vd.yml` |
| `configs/replay_tree/dataset/coco_detection.yml` | `configs/replay_tree/dataset/coco_detection.yml` |
| `configs/replay_tree/deim_rtdetrv2/rtdetrv2_r18vd_120e_coco.yml` | `configs/replay_tree/deim_rtdetrv2/rtdetrv2_r18vd_120e_coco.yml` |
| `configs/replay_tree/deim_rtdetrv2/rtdetrv2_r50vd_6x_coco.yml` | `configs/replay_tree/deim_rtdetrv2/rtdetrv2_r50vd_6x_coco.yml` |
| `configs/replay_tree/deim_sdf/exp_binary_mask_36e.yml` | `configs/replay_tree/deim_sdf/exp_binary_mask_36e.yml` |
| `configs/replay_tree/deim_sdf/exp_p2_query400_36e.yml` | `configs/replay_tree/deim_sdf/exp_p2_query400_36e.yml` |
| `configs/replay_tree/deim_sdf/exp_p2_query600_36e.yml` | `configs/replay_tree/deim_sdf/exp_p2_query600_36e.yml` |
| `configs/replay_tree/deim_sdf/exp_query500_36e.yml` | `configs/replay_tree/deim_sdf/exp_query500_36e.yml` |
| `configs/replay_tree/deim_sdf/exp_r50_query300_36e_clean.yml` | `configs/replay_tree/deim_sdf/exp_r50_query300_36e_clean.yml` |
| `configs/replay_tree/deim_sdf/exp_r50_query500_36e.yml` | `configs/replay_tree/deim_sdf/exp_r50_query500_36e.yml` |
| `configs/replay_tree/runtime.yml` | `configs/replay_tree/runtime.yml` |
| `provenance/det_engine_recovery_diff.patch` | `provenance/det_engine_recovery_diff.patch` |

### Files that only reference or use DEIM interfaces

`analysis/*.py` are study-specific audit, inference-interface, benchmark or reproduction scripts. They may import a separately installed compatible model/evaluation implementation; the imported third-party library source is not bundled. `figures/*/*.py`, numeric tables, cache/checkpoint manifests and provenance audits refer to study results and DEIM cohort/configuration identifiers; they are not a redistribution of the full model repository. Some generated manifests embed recorded settings; original DEIM attribution also applies to those configuration portions.

## LIVECell

LIVECell images and annotations are NOT redistributed. Users must obtain LIVECell from its [original source](https://github.com/sartorius-research/LIVECell) and comply with its original terms. Apache-2.0 licensing of archive code/configuration/documentation does not replace or amend LIVECell's dataset license or grant rights in omitted data.

## Other external materials

Checkpoints and prediction caches are NOT INCLUDED. No MDPI template/logo, complete DEIM source repository, third-party evaluation-library implementation or external alignment QA helper is included. Python dependencies keep their own licenses; importing them does not relicense them. The small alignment-warning adapters are study packaging code, not copied helper implementations.
