# Data Availability consistency check

No manuscript source or PDF was edited for this release.

| Manuscript location | Meaning | Public subset treatment |
|---|---|---|
| Methods 3.6, archived configuration/checkpoint/cache records | Larger study archive includes prediction caches and supports rerunning analysis when those caches are available | Public README explicitly excludes cache payloads and full checkpoints; Level B requires external caches and matching annotations |
| Supplementary S3 | Study archive contains training outputs, available source/config records and per-ground-truth intermediates, with missing historical snapshots | Public subset includes available configurations, summaries, hashes and provenance; no training outputs, full source repository or per-ground-truth intermediates |
| Current Data Availability | Research archive retained; no public identifier established yet | Still true; this candidate is not uploaded and has no identifier |
| Draft after-release statement | Code/configuration/derived-table/provenance/figure materials at a real public identifier | Valid only after actual public release and replacement of the identifier placeholder |

Consistency: PASS with the explicit study-archive/public-subset distinction. A reader must not interpret archived caches as publicly released caches. Full historical training reproduction is not guaranteed. The supplied manifests do not include cache-payload SHA-256; only available checkpoint/annotation hashes and cache inventories are claimed.

The separate internal DATA_AVAILABILITY_FINAL_OPTIONS.md contains the before-release draft and after-release template. It has not been inserted in main.tex.
