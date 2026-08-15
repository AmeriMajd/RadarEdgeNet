# Baseline audit

Audit date: 2026-08-10

This document records the state of the legacy prototype before the reproducibility rebuild. It is
intentionally direct: these are engineering findings, not a judgment on the value of the original
learning project.

## What exists

- 999 lines of Python across one fusion script and five utility scripts.
- 131 per-sequence preprocessed HDF5 files, 1,700 samples, and existing train/validation files.
- Random-forest, MLP, PointNet-like, and TFLite optimization experiments.
- Historical plots, a notebook, one pickle artifact, and several TFLite artifacts.
- Four Git commits, all project data/model/plot artifacts stored directly in Git.

## Verified data facts

The current per-sequence files contain 1,438 pedestrian samples (label 7) and 262 bicycle samples
(label 5), or 84.6% versus 15.4%. About 76.6% of allocated point slots are NaN padding.

The two views in every inspected preprocessed sample are not independent sensors:

- y, compensated velocity, and RCS are exactly equal between views for every valid point;
- x differs by one constant normalized offset (mean `0.030013`, standard deviation `0`);
- this matches `preprocess_all_sequences.py`, which copies a point cloud and adds `0.5` to x.

Consequently, existing “fusion” DL metrics must be treated as prototype-only and must not be used as
evidence of multi-sensor improvement.

## Critical findings

| Priority | Finding | Impact | Resolution |
| --- | --- | --- | --- |
| P0 | DL preprocessing creates its second view by translating a copy | Invalidates multi-radar fusion claims | Rebuild from real time-aligned sensor scenes |
| P0 | Samples are randomly split after all sequences are concatenated | Related tracks/context can leak into validation | Group by complete sequence and preserve provenance |
| P0 | Global normalization uses every sequence before splitting | Validation information leaks into training transforms | Split first; fit statistics on train sequences only |
| P0 | HDF5 contracts disagree: preprocessed `fused_point_clouds` is 4-D, processed `point_clouds` is 3-D, benchmark expects 4-D, optimizer expects 3-D | Pipeline stages cannot compose | One versioned schema with contract tests |
| P0 | Fusion filters the full table before applying scene `radar_indices` | Scene ranges can address the wrong rows | Slice the original HDF5 array first, then filter the scene |
| P1 | Hard-coded Windows paths and top-level execution | Code cannot be reused or safely imported | Package code and expose configuration through CLI arguments |
| P1 | Bare exceptions and global warning suppression | Corrupt/missing data can disappear silently | Catch specific errors and emit contextual failures |
| P1 | Greedy matching permits multiple objects to select the same partner | Incorrect many-to-one fusion associations | Use gated one-to-one assignment and test it |
| P1 | Random augmentation has no controlled generator and is followed by SMOTE | Runs are not reproducible; synthetic distribution is difficult to justify | Seed centrally and compare imbalance methods through ablation |
| P1 | The random-forest pickle omits its fitted scaler and class/threshold metadata | Served inference cannot reproduce evaluation preprocessing | Register a complete inference bundle |
| P1 | `DataGenerator` mutates slices of the source training array in place | Translation accumulates between epochs | Copy batches or use a stateless TensorFlow transform |
| P1 | “PointNet++” and “DGCNN” implementations are generic Conv1D networks | Architecture names and conclusions are misleading | Implement the defining sampling/graph operations or rename baselines |
| P2 | Accuracy and weighted F1 dominate evaluation of an imbalanced set | Minority-class failures can be hidden | Make macro F1 and per-class recall primary |
| P2 | Models, datasets, and plots are committed without run metadata | Artifacts are large and not traceable to a reproducible run | DVC/object storage plus MLflow registry |
| P2 | Requirements are a large workstation/Conda export including transitive and non-Python packages | Installation is fragile and security updates are unclear | Maintain small direct dependency groups in `pyproject.toml` |
| P2 | No tests, CI, service, container, IaC, or monitoring | No automated delivery or operational story | Add incrementally behind phase gates |

## Foundation changes completed

- Added an installable `radar_edge_net` package supporting Python 3.11 and 3.12.
- Reduced dependencies to explicit core and optional groups.
- Added a strict canonical HDF5 loader and writer.
- Added deterministic sequence-grouped splitting that selects a class-balanced candidate split.
- Stored `sequence_ids` and schema version in new processed outputs.
- Added unit tests, Ruff linting, and a GitHub Actions matrix.
- Added the project README and staged delivery/learning roadmap.

The grouped split was exercised against all current per-sequence artifacts into a temporary output:
1,367 training samples from 104 sequences and 333 validation samples from 27 disjoint sequences,
with both labels present in each split. This validates the new infrastructure; it does **not** make
the synthetic source views scientifically valid.

