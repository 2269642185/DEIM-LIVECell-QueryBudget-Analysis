# Table fields and missing records

CSV/JSON tables preserve the input candidate's scientific values. No cohorts are combined.
Primary R18, four-budget R18 and R50 directories are separate evidence cohorts; use their own manifests.
`model` encodes backbone and query budget. `Q`/`queries`/`query_budget` count prediction slots; `image_id` is a dataset identifier, not a personal identifier.
`gt`/`GT` count ground-truth instances; `tp`, `fp`, `fn` and uppercase equivalents are final-mask evaluation counts at the recorded score and mask-IoU thresholds.
`precision = sum(TP)/(sum(TP)+sum(FP))`; micro recall is `sum(TP)/sum(GT)`; `mean_image_recall` is a distinct per-image mean.
Candidate counts/coverage use score-free matching and are not deployment recall. `budget_gap` is arithmetic `max(N_GT-Q,0)`, not measured FN.
Threshold/IoU columns identify the operating point. Bootstrap intervals resample images, not training seeds; recorded seed and replicate count must be preserved.
Engineering latency is milliseconds per image, memory MiB, FLOPs raw operations (or G when explicitly named). Benchmark outputs are recorded observations, not remeasured by Level A.
Missing/UNKNOWN/NOT_ARCHIVED/NA entries are retained; they are not zero values.

`tables/frozen_manuscript/*_table_*.json` contains exact LaTeX tabular transcriptions; exporting these is format reproduction, not independent numerical validation. Primary summary CSVs are recalculated and compared by the Level A command.
