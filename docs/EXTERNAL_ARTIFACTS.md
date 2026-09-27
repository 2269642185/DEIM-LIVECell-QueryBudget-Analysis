# External artifacts and untested workflows

These algorithms are included for protocol inspection. They were syntax-checked, but not executed against missing external artifacts in this release.

| Script | External requirements |
|---|---|
| analysis/run_saved_audit.py and analyze_*.py | Complete saved prediction caches and hash-matching prepared validation annotation; compatible NumPy/SciPy/faster-coco-eval stack |
| analysis/reproduce_four_budget.py | FOUR_BUDGET_ROOT results and prediction-cache tree, RESEARCH_ROOT recorded configs, MODEL_PROJECT includes, ANNOTATION, fresh OUTPUT_ROOT; optional old logs for the provenance branch |
| analysis/infer_r18.py / infer_r50.py | LIVECell data, hash-matching terminal EMA checkpoints, compatible customized DEIM binary-mask implementation and correct configs |
| analysis/audit_training_targets.py | MODEL_PROJECT, prepared training annotations/images, actual loader source and original p0_audit manifest layout |
| analysis/benchmark_query_cost.py | Compatible model, checkpoint/data manifest, GPU and recorded benchmark environment |
| figures/qualitative/qualitative_dense_scene.py | PROJECT_ROOT original research layout, LIVECell imagery/annotations and the correct qualitative prediction cohort |

No runnable cache/inference command is claimed verified here. Use script interfaces only after supplying and verifying the external inputs. The cache adapter verifies annotation hash and image inventory; it does not authenticate cache contents by SHA-256 because cache-payload hashes are absent from the supplied manifests.
The available manifests record checkpoint/annotation hashes and model/image inventories. Cache-byte hashes are UNAVAILABLE in the input package and have not been invented. The release manifest hashes the included manifest FILES, not omitted cache payloads.
The original LIVECell release may not be byte-identical to locally prepared annotations with stored mask grids. Those prepared annotations are not redistributed.
