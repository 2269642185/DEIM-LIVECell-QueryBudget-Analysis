# Query-Budget Effects in Dense Full-Image Cell Instance Segmentation: A Prediction-Level Analysis on LIVECell

Authors: Baohua Wang, Shengbin Yu, Yunlong Ma and Jian Zhang.

## Purpose and status

This archive accompanies an empirical prediction-level analysis of query-budget effects in full-image LIVECell instance segmentation.
It supports reproduction of selected reported derived analyses and documents the provenance available for the experimental cohorts. This is an empirical study archive, not a new model release or a self-contained model repository.
GITHUB-READY LOCAL RELEASE: Apache-2.0 has been selected for code/configuration and author-owned documentation. No public upload or identifier exists yet. See GITHUB_RELEASE_INSTRUCTIONS.md.

## Evidence hierarchy

- Primary: R18 Q300/Q500.
- Secondary robustness: score threshold / mask IoU / paired image bootstrap.
- Complementary: R18 Q300–Q600 compatibility-verified saved-cache analysis; historical provenance Level 2.
- Corroborating: R50 Q300/Q500; Q300 recovery-fixed2.
- Exploratory: mask-head screens.

Arithmetic exposure is not evaluation FN. Candidate coverage is geometric availability, not thresholded recall. No universal/optimal query budget or fully controlled four-level causal experiment is claimed.

## Included

- configs/: available launch YAML, serialized configurations, inference records, missing-record markers and a present-day replay include tree.
- analysis/: study evaluation/prediction audits, candidate coverage, thresholds, bootstrap/count summaries, arithmetic exposure, inference/benchmark interfaces and Level A wrappers.
- tables/: derived CSVs, bootstrap summaries, engineering JSON/CSV and exact manuscript table transcriptions.
- manifests/: checkpoint/annotation hashes where available and prediction-cache model/image inventories; cache-payload hashes were not supplied.
- provenance/: protocol and target-semantic audits, recovery diff, file origins and modification records, preserved DEIM license.
- figures/: numeric plotting scripts/inputs, frozen numeric figure exports, conceptual diagram source and qualitative generator only.
- environment/: recorded benchmark environment and separately recorded Level A test environment.
- docs/: data dictionary, verified reproduction evidence, external-input requirements and scope checks.

## Not included

LIVECell raw images and annotation files; full model checkpoints; large prediction caches; raw training outputs; per-ground-truth intermediate tables; unavailable historical source snapshots; unavailable historical training-time annotation hashes. The rendered qualitative Figure 2 is excluded because it contains LIVECell-derived imagery. No model weights are bundled.

Obtain LIVECell from its official source: https://github.com/sartorius-research/LIVECell . Original dataset paper: Edlund et al., Nature Methods 18 (2021), 1038–1045, https://doi.org/10.1038/s41592-021-01249-6 . Data are not redistributed here. Locally prepared mask-grid annotations may differ from the original files.

## Reproducibility levels

| Level | What this candidate supports | Required inputs / limit |
|---|---|---|
| A — Table/Figure reproduction | Recompute five primary R18 summary tables from supplied per-image CSV; export all 14 frozen tabular transcriptions; regenerate Fig. 1, Fig. 3 and Fig. S1 | Included derived CSV/JSON plus the documented Level A Python packages. Engineering/AP/screen values are retained observations, not regenerated measurements. Conceptual source is included; qualitative Fig. 2 needs external data |
| B — Prediction-audit reproduction | Audit algorithms and provenance available | External complete saved prediction cache AND matching prepared annotation required; not reproducible from release alone |
| C — Full inference reproduction | Inference interfaces and available configurations | External dataset, full checkpoints and compatible customized DEIM implementation required; not reproducible from release alone |
| D — Full historical training reproduction | Available historical records can be inspected | Cannot be fully guaranteed: some source snapshots, contemporaneous annotation hashes and environment records are unavailable |

## Verified reproduction commands

Run from this archive root in an environment with the versions listed in environment/requirements_level_a_tested.txt. Choose new, empty output directories outside the archive (the examples are relative sibling directories). These exact commands were executed successfully for this candidate:

```text
python analysis/reproduce_tables.py --output ../table_reproduction
python analysis/reproduce_figures.py --output ../figure_reproduction
```

The table command executes the original summarize_primary.py unchanged, compares numeric cells against the archived summaries and exports exact LaTeX table bodies from JSON. It does not reevaluate masks or estimate new scientific results. The figure command executes the existing plotting algorithms in a temporary working copy and copies three PDF/PNG outputs. All input tables remain unchanged. See docs/REPRODUCTION_TESTS.json for actual comparisons and output hashes.
Plotting uses the existing disclosed alignment-warning adapter; the unavailable external alignment helper is not redistributed. Regenerated PDFs were visually inspected. Byte-identical historical PDFs are not promised across font/backend versions.

Other scripts require external artifacts and were NOT executed. See docs/EXTERNAL_ARTIFACTS.md. No training, model inference, cache evaluation or new GPU benchmark was run.

## Configuration and environment cautions

configs/replay_tree contains current shared include files; it is not a historical source snapshot. Dataset scaffold defaults are overridden by study root configs and must be checked against the cohort manifest before external inference. Developer comments were removed without changing any parsed YAML value. Literal path placeholders are descriptive and must be replaced in a working copy; they are not executable locations.
Recorded engineering environment: RTX 4090 D, PyTorch 2.1.2, CUDA 11.8, Python 3.9.25. This is not asserted to describe every historical training run. Level A was tested on the separately documented current packaging host.

## Study archive versus public release

The manuscript Methods 3.6 and Supplementary S3 refer to a larger author-controlled study archive, including caches and training artifacts. This public-release subset deliberately omits those payloads. Their presence in the study archive does not imply public availability. Per-image derived tables and available artifact hashes are included; per-ground-truth intermediates are not.
No promise of future checkpoint/cache release is made. The manuscript Data Availability statement remains unchanged until a real public URL/DOI is obtained.

## Integrity and provenance

MANIFEST_SHA256.txt hashes every release file except itself. The ZIP checksum is stored outside the ZIP. PUBLIC_RELEASE_FILE_AUDIT.csv classifies every input file and every retained/new release file; A/D identify study material and generated records; these categories do not relicense underlying third-party data. B includes ancillary upstream license notices. See ARCHIVE_CONTENT_SUMMARY.md for counts and limitations.

## License

Archive code/configuration materials and author-owned documentation are released under Apache License 2.0 unless otherwise noted. See [LICENSE](LICENSE) for the complete standard text and [LICENSE_OR_USAGE.md](LICENSE_OR_USAGE.md) for scope.
Files derived from DEIM retain upstream attribution and Apache-2.0 requirements; [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists their modification notices.
LIVECell images and annotations are NOT redistributed and remain subject to their original terms. Users must obtain LIVECell from its original source and comply with those terms. Checkpoints and prediction caches are not included.

Table/figure reproduction: YES within the tested Level A scope above. Full prediction audit from the public release alone: NO. Full inference: NO. Full historical training reproduction: NO.
