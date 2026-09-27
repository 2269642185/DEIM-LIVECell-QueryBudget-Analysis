# Environment evidence

Recorded engineering benchmark only: Python 3.9.25; PyTorch 2.1.2+cu118; CUDA runtime 11.8; cuDNN 8700; NVIDIA GeForce RTX 4090 D; Linux; FP32; batch size 1. See recorded_benchmark_environment.json.
This is NOT asserted to be the training environment for every historical run. Historical exact dependency locks and some source snapshots are unavailable.

packaging_host_versions.json records the earlier packaging host. release_test_environment.json records the actual Level A verification environment. The release tests need Python, NumPy, pandas and matplotlib; no PyTorch/CUDA is needed for Level A. requirements_level_a_tested.txt records tested versions, not a complete historical environment.
requirements_snapshot.txt records benchmark PyTorch and dependency names for other tools. Prediction audits additionally need compatible scientific/evaluation libraries and external caches/annotations. Inference needs compatible customized DEIM code, dataset, weights, PyTorch and its dependencies. No training, inference or GPU benchmark was executed for this release.
