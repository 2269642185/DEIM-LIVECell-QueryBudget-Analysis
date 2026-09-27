# FOUR-BUDGET PROTOCOL + CACHE ANALYSIS REPORT

Audit date: 2026-09-24. Scope: the existing terminal R18 Q300/Q400/Q500/Q600 saved-prediction cohort. No training, inference, checkpoint editing, cache editing, or manuscript revision was performed. This report makes a **Level 2** judgment, not a four-run causal claim.

## 1. Protocol classification

**LEVEL 2 — EVALUATION-MATCHED, TRAINING MATCH NOT FULLY PROVEN.** All four cached prediction sets were produced and analyzed by the same terminal audit scripts on the same 570-image validation set, with the same validation annotation SHA-256. A read-only server follow-up recovered the Q300/Q500 full training logs and both launch scripts; all four historical logs now provide their serialized launch configurations and 36 complete epochs. Per-run frozen source trees, attributable Git commits, and training-annotation hashes at launch remain unavailable. The Q300 launch configuration also declares a different mask-target preparation route from Q400/Q500/Q600. A separate current-server, full-training-set audit found those two target routes numerically identical for 1,018,576 instances, but it cannot establish historical source/data identity at all four training starts.

## 2. Allowed differences

`RTDETRTransformerv2.num_queries` and `PostProcessor.num_top_queries` are 300/400/500/600. Output directories, checkpoint filenames, run IDs, and learned checkpoint SHA-256 values differ as expected. The four local epoch-35 checkpoint SHA-256 values were recomputed and match the terminal manifest exactly; see `four_budget_runs.csv`.

## 3. Unexpected config differences

The normalized comparison of all four **historical logged launch configurations** identifies two additional fields (full machine-readable values in `four_budget_config_diff.json`):

| Field | Q300 | Q400/Q500/Q600 | Interpretation |
|---|---|---|---|
| `train_dataloader.dataset.mask_grid_size` | absent, current dataset default 0 | 16 | Different target preparation paths. The archived current-server target audit reports exact target equality across all 1,018,576 valid instances; historical run-time file/source identity is not recorded. |
| `val_dataloader.num_workers` | 0 | 4 | Validation loading/concurrency differs in declared config. This is not evidence of a mask metric change, but it is outside the strict allowed-difference list. |

The Q300 and Q500 logs were found at the server project root and copied read-only into `server_check_20260924/`; Q400 and Q600 full logs were already archived locally. All four `cfg:` lines parse and, after allowing Q/top-Q and output/config paths, differ only in the two fields above. The original YAMLs remain separate from these logged resolved configurations. The older `exp_query500_36e_run.log` follows a **different** loss trajectory and was not used: the `exp_query500_36e_retrain.log` loss averages match all 36 epochs of the terminal Q500 `log.txt` to its four-decimal print precision. Q300's recovered retrain log likewise matches all 36 terminal Q300 epoch summaries.

The server-side target-equality audit recorded a then-current training JSON SHA-256 of `4df70f126ade5858ff4637f2f7c543b89774f50bd581b3e772fea9f586acab36`. The local training JSON now available has a different SHA-256 (`553ddc7f9a33aa16d5144c9b377fc0b8bc7ce7ba14205c0a19d1466ca8e12128`); neither hash was captured for each historical training launch. The locally verified **validation** annotation hash is separate and shared by the four cached evaluations.

## 4. Source-code differences

**No per-run training source snapshots or attributable Git commit IDs were archived; the server project is not a Git repository.** A semantic tree diff of model, matcher, loss, dataloader, augmentation, optimizer, scheduler, EMA, trainer, invalid-output handling, and mask-target code cannot be certified. Evidence-status check against Q300: Q400 **FAIL to prove equivalence**; Q500 **FAIL to prove equivalence**; Q600 **FAIL to prove equivalence**. This is a provenance failure, not an observed code difference. Current server file mtimes show no relevant `engine/` Python or common-config modifications between the Q300/Q500 retrains and Q400/Q600 runs; mtimes are useful corroboration, not a historical source hash. The later R50 recovery edit to `engine/solver/det_engine.py` is dated 2026-09-21, after all four R18 runs.

Evaluation is different: the archived command scripts invoke the same `audit_infer.py` for all four checkpoints, and the terminal `analyze.py` processes all four models in one loop. Thus inference, mask reconstruction, score handling, category handling, greedy score-order class-aware one-to-one matching, and class-agnostic all-query maximum-cardinality candidate coverage have the same implementation in this cohort. The evaluation script's `metrics()` uses **micro ratios of summed TP/FP/FN**; candidate coverage is summed maximum-cardinality matches divided by summed GT. The separate `mean_image_recall` column is macro and was **not** used as primary recall.

## 5. Training-log anomalies

| Run | Observed records | Anomaly/provenance boundary |
|---|---|---|
| Q300 | Full recovered retrain log and 36 matching JSONL summaries; all 36 epochs finish step 1625/1626; launch `resume=None`, seed 0, AMP/EMA enabled; no recorded NaN, skip, or OOM | Historical source-tree and training annotation hash at launch not archived. |
| Q400 | Full server training log and 36 matching summaries; all 36 epochs finish step 1625/1626; launch `resume=None`, seed 0, AMP/EMA enabled; no recorded NaN, skip, or OOM | Historical source-tree and training annotation hash at launch not archived. |
| Q500 | Full recovered **retrain** log and 36 matching summaries; all 36 epochs finish step 1625/1626; launch `resume=None`, seed 0, AMP/EMA enabled; no recorded NaN, skip, or OOM | The earlier same-name Q500 run log is a different trajectory. Historical source-tree and training annotation hash at launch not archived. |
| Q600 | Full server training log and 36 matching summaries; all 36 epochs finish step 1625/1626; launch `resume=None`, seed 0, AMP/EMA enabled; no recorded NaN, skip, or OOM | Historical source-tree and training annotation hash at launch not archived. |

All four logged epoch-mean LR series are identical. Each selected checkpoint has `last_epoch=35` and evaluation weights `ema.module` in the terminal manifest. The full logs contain no recorded R50-style recovery, skipped batch, NaN, or OOM event. The Q300/Q500 launches are documented in `_retrain_binary.sh`; Q400/Q600 in `_run_p2_query400_600.sh`, both preserved in `server_check_20260924/`.

## 6. Final matched-protocol judgment

**Can the manuscript upgrade to “matched four-budget R18 comparison”? NO under the strict proof rule.** Keep “compatibility-verified response characterization” pending historical training source and training-annotation provenance. The recovered complete launch logs substantially narrow the uncertainty and confirm settings, 1626-step epochs, seed, AMP, EMA, and lack of logged recovery events. They do not supply immutable source/data snapshots from each launch. The current-server target-equality audit removes one plausible numerical target confound for the currently audited files, but it does not close that historical gap. Four different checkpoint hashes are expected and are not a mismatch. This remains a single-seed, single-dataset characterization.

## Integrity gate and definitions

- Cache file counts: Q300/Q400/Q500/Q600 = **570/570/570/570**. Every gzip JSON was read and checked for its image ID and exact Q-length score/label/mask vectors. Image ID sets are identical to one another, the manifest, and the validation annotation's 570 images.
- Validation annotation SHA-256: **`eb72a68f5065d9ade1805b496eecdcd14ca275e5f1bc6aae7e07279bddadfdc7`**, verified locally and identical in all four manifest records. It is a validation hash, **not** proof of historical training annotation hashes.
- Stratum image counts: **362 / 103 / 105**, exactly the prespecified `N_GT≤300`, `301≤N_GT≤500`, and `N_GT>500` groups. Total 570 images and 181,610 non-crowd GT instances.
- Primary operating point: score ≥0.50, mask IoU ≥0.50. Precision = ΣTP/(ΣTP+ΣFP), recall = ΣTP/ΣGT. Candidate coverage = Σ maximum-cardinality matched GT / ΣGT with every final-query mask and no score/category filter, mask IoU ≥0.50. It is a geometric candidate upper bound, not deployment recall. All 22,800 per-image primary/threshold rows satisfy TP+FN=GT.
- The primary diagnostic retains one class per final query. The separate original class-pair top-k COCO reproduction stays distinct; no protocol substitution or fresh inference was made.

## 7. Stratified results

Each cell below is **precision / recall / candidate coverage**, micro-aggregated over images in that stratum. Exact counts and ratios are in `../analysis/four_budget_stratified_primary.csv`.

| GT per image (n) | Q300 | Q400 | Q500 | Q600 |
|---|---|---|---|---|
| ≤300 (362) | .7755 / .7264 / .7875 | .7555 / .7418 / .8303 | .7424 / .7501 / .8503 | .7491 / .7382 / .8643 |
| 301–500 (103) | .8419 / .5843 / .5917 | .8005 / .6746 / .6906 | .7717 / .7196 / .7488 | .7716 / .7280 / .7851 |
| >500 (105) | .8876 / .2987 / .2997 | .8590 / .3788 / .3807 | .8293 / .4488 / .4520 | .8055 / .5055 / .5136 |

## 8. Threshold results

The 40-row `../analysis/four_budget_threshold_scan.csv` contains Q × five prespecified score thresholds (0.05, 0.10, 0.30, 0.50, 0.70) × mask IoU (0.50, 0.75), with TP/FP/FN, micro precision/recall, and diagnostic F1. At **every one of the five common score thresholds**, aggregate recall orders Q300 < Q400 < Q500 < Q600 at both IoU levels. This does **not** mean all strata improve or that precision is fixed: precision generally falls as Q rises at a fixed threshold. At the primary score 0.50, aggregate recall is `.4813/.5454/.5927/.6195` at IoU 0.50 and `.3011/.3342/.3564/.3673` at IoU 0.75.

The previously implemented complete-score-tie common-precision diagnostic was reused at targets 0.80 and 0.90; no new matching rule was invented. At IoU 0.50 and target precision 0.80, recall is `.4875/.5459/.5832/.6062`; at target 0.90 it is `.4293/.4598/.4808/.4888`. At IoU 0.75, the matched-precision ordering is **not** monotone (for example target 0.80: `.2159/.2222/.2276/.2214`). See `../analysis/four_budget_common_precision_existing.csv` for achieved precision and score.

Candidate coverage is score-independent and appears only in the stratified/overall files, not as a threshold-specific quantity.

## 9. Non-monotonic findings

- `N_GT≤300` **recall decreases** from Q500 `.7501` to Q600 `.7382` (−.0119), despite candidate coverage increasing `.8503→.8643`.
- Precision falls Q300→Q400 and Q400→Q500 in all three strata; it also falls Q500→Q600 in the 301–500 and >500 strata. The ≤300 precision rises slightly Q500→Q600 (`.7424→.7491`). Candidate coverage rises at every Q step in every stratum.
- At the reused common-precision diagnostic with mask IoU 0.75, Q600 and sometimes Q400/Q500 do not exceed the preceding Q's recall. This precludes a universal monotone statement even after controlling the operating precision.

## 10. Arithmetic exposure

| Q | Images with GT>Q | Fraction of 570 | Σmax(GT−Q,0) | Excess/181,610 GT |
|---:|---:|---:|---:|---:|
| 300 | 208 | 36.49% | 68,271 | 37.59% |
| 400 | 148 | 25.96% | 50,728 | 27.93% |
| 500 | 105 | 18.42% | 38,376 | 21.13% |
| 600 | 81 | 14.21% | 29,041 | 15.99% |

**Arithmetic exposure ≠ observed FN.** The gap is a structural count statistic only.

## 11. Bootstrap

**RUN.** Paired-image, within-stratum, 2,000 resamples per Q400/Q500/Q600 minus Q300; fixed seed `20260924`, micro recall recomputed as resampled ΣΔTP/ΣGT. Exact nine intervals are in `../analysis/four_budget_stratified_bootstrap.csv`. Examples: ≤300 Q600−Q300 `.0118` (95% image-bootstrap CI `.0062–.0171`); >500 Q600−Q300 `.2067` (`.1929–.2212`). These intervals reflect **validation-image sampling uncertainty only**, not training-seed variation.

## 12. Files generated

- `audit/four_budget_protocol_audit.md`, `audit/four_budget_runs.csv`, `audit/four_budget_protocol_criteria.csv`, `audit/four_budget_config_diff.md`, `audit/four_budget_config_diff.json`, and read-only server copies in `audit/server_check_20260924/`.
- `analysis/four_budget_stratified_primary.csv`, `analysis/four_budget_stratified_publication_table.csv`, `analysis/four_budget_threshold_scan.csv`, `analysis/four_budget_common_precision_existing.csv`, `analysis/four_budget_stratified_bootstrap.csv`, `analysis/query_budget_exposure.csv`.
- `figures/four_budget_stratified_response.pdf/.png` and `figures/four_budget_threshold_pr.pdf/.png`, plus panel-alignment/collision QA records. Figures use measured points only, without smoothing.
- `reproduce_from_saved_audit.py` and `plot_saved_analysis.py` regenerate the tables and figures from the archived terminal cache-derived records and validation annotation. The upstream archived `analyze_q400_q600.py` is the cache-to-per-image derivation; this audit reads its immutable output and independently checks counts against all cache JSONs and annotation IDs/hashes.
- `REPRODUCIBILITY.md` records the input chain, commands, and the scope of the checks. This round did not repeat the CPU-intensive mask-IoU matcher over all cached masks.

## 13. Manuscript recommendation

**B. Retain compatibility-verified wording.** Do not edit the manuscript yet. After author review, Methods could specify the single shared terminal evaluation code and micro aggregation; Results could add the two supplementary plots, the three-stratum table, threshold sensitivity, and Q500→Q600 low-count reversal; Discussion should preserve the single-seed and historical training-provenance limitations; Supplementary Materials could house the exact thresholds, bootstrap intervals, and arithmetic exposure definition. Any change to stronger four-budget language would require attributable historical run-time source and training-annotation evidence, followed by a fresh source semantic audit.
