# Server follow-up: four R18 training runs

Read-only inspection on 2026-09-24. No server file was modified, and no training or inference was started.

## Found

| Run | Full training log | Launch script | Link to terminal checkpoint |
|---|---|---|---|
| Q300 | `<MODEL_PROJECT>/exp_binary_mask_36e_retrain.log` | `<MODEL_PROJECT>/_retrain_binary.sh` | All 36 printed epoch-mean losses agree with the archived terminal `log.txt` to four-decimal display precision; local `checkpoint0035.pth` SHA-256 matches the terminal prediction manifest. |
| Q400 | `<MODEL_PROJECT>/exp_p2_query400_36e_run.log` | `<MODEL_PROJECT>/_run_p2_query400_600.sh` | The already archived `server_training.log` is byte-identical to the server log; checkpoint SHA-256 matches the manifest. |
| Q500 | `<MODEL_PROJECT>/exp_query500_36e_retrain.log` | `<MODEL_PROJECT>/_retrain_binary.sh` | All 36 epoch-mean losses match terminal `log.txt`; checkpoint SHA-256 matches the manifest. The older `exp_query500_36e_run.log` has a different training trajectory and is excluded. |
| Q600 | `<MODEL_PROJECT>/exp_p2_query600_36e_run.log` | `<MODEL_PROJECT>/_run_p2_query400_600.sh` | The archived full log is byte-identical to the server log; checkpoint SHA-256 matches the manifest. |

All four matching logs contain a serialized complete launch `cfg` and report seed 0, `resume=None`, AMP and EMA enabled, 36 epochs, and **1626/1626 completed steps in every epoch**. No log records a skipped batch, NaN, or OOM. The full logged configuration diff has only the allowed Q/top-Q/output-path changes plus Q300's absent `train_dataloader.dataset.mask_grid_size` (versus 16) and `val_dataloader.num_workers=0` (versus 4). See `four_budget_config_diff.json`.

The four matching log SHA-256 values, in Q order, are:

```text
Q300 f357b861da80e05cf344b321f36a3ca17935d8ed853ee2259858d18a0c5046a4
Q400 f2eb359a7c5627034c4c364f45c7aaf3aaa29cfd26447130bd688f27081af264
Q500 522aca007fd76e3154b33361ba281364b6e8db4b402a7293c9054448221bacb5
Q600 bbeef9469fa4f2871564e63abe2bc5e7aaef1ec2351c2008c0496dd179d61804
```

The recovered Q300/Q500 logs and both scripts were copied to `server_check_20260924/`. The Q400/Q600 logs were already in the local training archive and their hashes were checked against the server.

## Still absent

- The server working tree has no `.git` directory; no per-run commit IDs or immutable training-source snapshots were found in the inspected DEIM/output/audit locations. Source-file mtimes show no relevant engine or common-config changes between the four R18 runs, but mtimes are not a semantic source diff.
- The training JSON path is declared in all four launch configurations, but **no at-launch SHA-256 per run** was found. The current server training JSON has an older mtime (2026-07-21), and a separate audit recorded its then-current SHA-256; those observations cannot substitute for four historical hashes. The local training JSON differs from the server copy.
- Q300/Q500 original output directories have been removed from `<CHECKPOINT_ROOT>`. Their terminal checkpoints and epoch summaries are preserved in the local research archive and match the terminal evaluation manifest.

**Protocol impact:** the recovered logs close the previous missing-log/launch-config gap, but strict Level 1 source/data provenance remains unproved. Classification stays **Level 2**. The manuscript was not edited.
