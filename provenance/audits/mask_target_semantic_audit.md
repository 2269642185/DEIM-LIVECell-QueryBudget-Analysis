# Mask-target semantic audit — 2026-09-24

## Evidence scope

The four historical training logs print `cfg:`. Their Q300/Q500 trajectories match all 36 archived epoch summaries; Q400/Q600 logs are byte-identical to archived server copies. The source trace below uses the current local code copy. The prior current-server target audit recorded identical hashes for dataset, criterion, and workspace source files. There is no frozen source tree or annotation hash from each training start, so historical equality remains unproven.

## A–D. Four configuration paths

| Run | Logged config path | Dataset key `train_dataloader.dataset.mask_grid_size` | Model key `RTDETRTransformerv2.mask_grid_size` | Target source |
|---|---|---:|---:|---|
| Q300 | `configs/deim_sdf/exp_binary_mask_36e.yml` | absent → `CocoDetection.__init__` default 0 | 16 | annotation `mask_grid` |
| Q400 | `configs/deim_sdf/exp_p2_query400_36e.yml` | 16 | 16 | annotation polygon regenerated |
| Q500 | `configs/deim_sdf/exp_query500_36e.yml` | 16 | 16 | annotation polygon regenerated |
| Q600 | `configs/deim_sdf/exp_p2_query600_36e.yml` | 16 | 16 | annotation polygon regenerated |

## E–F. Resolved values and runtime source trace

`train.py -c <run YAML>` → `engine/core/yaml_utils.py:load_config` recursively merges `__include__` → `engine/core/yaml_config.py:YAMLConfig` retains `yaml_cfg` and calls `engine/core/workspace.py:create` → registered `CocoDetection` in `engine/data/dataset/coco_dataset.py` → `ConvertCocoPolysToMask.__call__` → model `RTDETRTransformerv2` in `engine/deim/rtdetrv2_decoder.py` → `DEIMCriterion.get_loss('binary_masks')` → `loss_masks` in `engine/deim/deim_criterion.py` → `PostProcessor` binary branch for validation. The source-snippet manifest hashes the exact included copies.

Q300: `configs/deim_sdf/exp_binary_mask_36e.yml` → logged dataset setting absent → `CocoDetection(mask_grid_size=0)` → `ConvertCocoPolysToMask.__call__` reads stored `obj['mask_grid']` → float32 [N,256] target → binary 3-layer MLP `pred_masks` → Hungarian matched `loss_masks` → BCE-with-logits, sum / `num_boxes`, weighted 0.5.

Q500: `configs/deim_sdf/exp_query500_36e.yml` → logged dataset `mask_grid_size=16` → `CocoDetection(mask_grid_size=16)` → `ConvertCocoPolysToMask.__call__` rasterizes polygon, uses OpenCV `boundingRect` crop, `INTER_LINEAR` resize to 16×16, threshold `>0.5`, flatten float32 → same mask head and loss call. Q400/Q600 take this same route under their logged settings.

The dynamic branch uses polygon coordinates cast to int32, a full-image uint8 raster as an intermediate, crop at `cv2.boundingRect(poly)` (not annotation bbox), then bilinear interpolation. The stored branch reads a 256-element 0/1 grid. Both exclude `iscrowd != 0` and invalid boxes before emitting targets; no separate ignored-region mask is configured. Target is foreground 1/background 0. It is binary, not soft. Current-server audit verified exact float32 equality for **1,018,576/1,018,576** valid instances, **260,755,456** pixels, across 3,253 training images (recorded server annotation SHA-256 `4df70f126ade5858ff4637f2f7c543b89774f50bd581b3e772fea9f586acab36`). This is a current-file result, not four historical hashes.

All four logged configs specify `use_binary_mask_head=True`, model grid 16, `losses=['vfl','boxes','binary_masks']`, mask weight 0.5, SDF weight 0 and `mask_source='binary'`. Current criterion code uses `F.binary_cross_entropy_with_logits(..., reduction='none')`, sums 256 elements and divides by the number of GT boxes (clamped at 1, distributed average); it does not divide by 256. No Dice or focal **mask** loss is selected. Auxiliary and denoising decoder outputs receive their own mask loss through the same routine. The binary postprocessor uses sigmoid, box-relative bilinear resizing with `align_corners=False`, then `>0.5` and full-image placement; it differs across runs only in `num_top_queries=Q`. The separate saved-cache evaluator used the same terminal evaluation code for all four.

## G. Comparison

| Feature | Q300 | Q400/Q500/Q600 | Assessment |
|---|---|---|---|
| Target-generation code path | Stored JSON 16×16 grid | Dynamic polygon→16×16 grid | PATH DIFFERENT |
| Observed target values in current server audit | 1,018,576 exact pairs | Same | VALUES MATCHED for current file/source |
| Head, tensor shape, BCE normalization/weight | 3-layer MLP, 256 logits, sum/GT, 0.5 | Same | MATCHED in logged cfg/current code |
| Crowd/invalid boxes | Filtered by same converter | Same | MATCHED in current code |
| Postprocessor | binary sigmoid/bilinear/>0.5; top 300 | Same operation; top Q | MATCHED except intended Q budget |
| Historical source and annotation identity | Missing per-run hashes | Missing per-run hashes | UNKNOWN historically |

## H. Result

**PATH MATCH: NO.**

**CURRENT AUDITED TARGET VALUES / CONFIGURED SUPERVISION SEMANTIC MATCH: YES.** Q300 vs Q400: PASS; Q300 vs Q500: PASS; Q300 vs Q600: PASS, each scoped to the logged configuration, current verified source copy, and current server annotation audit. The paths are different but no target-value or loss difference was observed in that scope.

**HISTORICAL FOUR-RUN SEMANTIC MATCH: UNKNOWN.** Per-run training-time code/data identity was not captured. The log has resolved config, mask-head/loss/target-dataset keys and printed `loss_mask`, but does not log the actual target tensor bytes or a source commit. Runtime config evidence: Q300 YES; Q400 YES; Q500 YES; Q600 YES. Historical source/data attestation: NO for all four.

**CLASSIFICATION: LEVEL 2.** Evaluation matched; training history remains provenance-incomplete. Q300's different configuration path alone is not evidence of a numerical training target mismatch. The additional `val_dataloader.num_workers` difference (0 vs 4) is retained in the config diff and is not a mask-target operation.
