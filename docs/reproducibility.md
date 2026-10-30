# Reproducibility Guide

This guide is for teammates who need to reproduce, inspect, or independently verify the SafeSense results without using historical assumptions.

## 1. Scope

The current selected target is:

```text
SafeSenseFactorized
seed 22
60 epochs
SWA epochs 45-60 inclusive
skeleton + IMU
```

The goal of reproduction is to confirm the frozen evidence, not to use the even-subject result for further tuning.

## 2. Software requirements

Project Python requirement:

```text
>=3.10,<3.11
```

Pinned core requirements in `requirements.txt`:

```text
numpy==2.2.6
scipy==1.15.3
opencv-python-headless==4.12.0.88
torch==2.10.0
modal==1.5.4
```

The frozen final artifact recorded:

```text
Python 3.10.17
PyTorch 2.10.0+cu128
CUDA 12.8
NumPy 2.2.6
SciPy 1.15.3
GPU Tesla T4
```

The Modal image adds plotting/runtime packages and clones the external Fusion-GCN repository at image build time.

Paper/reporting utilities use an optional dependency group rather than changing the core experiment requirements:

```bash
.venv/bin/pip install -e '.[paper]'
```

This installs the pinned Matplotlib reporting dependency plus SciPy needed by the statistical scripts.

## 3. Local environment

The project already uses a `.venv` locally. A clean equivalent setup can be created with Python 3.10 and the repository requirements.

For a simple source/test environment:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt -r requirements-local.txt
.venv/bin/python -m pip install -e .
```

Do not commit `.venv`.

## 4. Dataset requirements

Raw UTD-MHAD files are not redistributed by this repository.

The selected SafeSense path requires skeleton and inertial MAT files sufficient to build the Fusion-GCN-compatible arrays.

The expected complete paired sample count is:

```text
861
```

Known absent trial IDs are:

```text
a8_s1_t4
a23_s6_t4
a27_s8_t4
```

## 5. Prepare the matched skeleton+IMU data

The selected model consumes the Fusion-GCN-compatible prepared data, not the older 32-frame tri-modal cache.

Example local preparation:

```bash
.venv/bin/python -m safesense prepare-fusion \
  --data-dir cache/extracted \
  --output-dir cache/fusion-gcn \
  --expected-count 861
```

Depending on where the extracted `Skeleton` and `Inertial` MAT files live, adjust `--data-dir` accordingly.

Successful preparation creates files including:

```text
cache/fusion-gcn/manifest.csv
cache/fusion-gcn/skeleton_train_features.npy
cache/fusion-gcn/train_labels.npy
cache/fusion-gcn/skeleton_val_features.npy
cache/fusion-gcn/val_labels.npy
cache/fusion-gcn/fusion_gcn_audit.json
```

Expected per-sample feature shape in the audit:

```text
[1,128,22,3]
```

## 6. Legacy tri-modal preparation

The generic command:

```bash
.venv/bin/python -m safesense prepare --raw-dir DATA --cache-dir CACHE
```

builds the earlier depth/skeleton/IMU 32-frame feature cache used for the Houssein reconstruction.

It is valid historical infrastructure but **is not the selected SafeSense training input**.

## 7. Validate locally before GPU runs

Run:

```bash
.venv/bin/python -m py_compile \
  src/safesense/safesense.py \
  src/safesense/efficiency.py \
  modal_app.py

.venv/bin/pytest -q
```

The latest validated state before this documentation task had:

```text
92 passed
```

For JSON evidence:

```bash
.venv/bin/python -m json.tool research/factorized-swa-final-results.json >/dev/null
.venv/bin/python -m json.tool research/factorized-swa-latency-optimization-results.json >/dev/null
```

Before committing changes:

```bash
git diff --check
```

## 8. Modal authentication and result volume

The project uses Modal app name:

```text
safesense
```

and private result volume:

```text
safesense-baseline-results
```

The global Modal installation was previously authenticated under the project environment.

Do not create a second temporary Modal environment simply to run these functions unless the project environment has genuinely changed.

## 9. Current factorized-SWA smoke

Before an expensive training reproduction:

```bash
modal run modal_app.py::safesense_factorized_swa_smoke
```

The smoke uses:

- one seed;
- validation subject 7;
- two epochs;
- SWA starting at epoch 2;
- approximately 5-second progress heartbeat.

It verifies that the training/SWA/progress path executes on a T4. Its accuracy is diagnostic only.

## 10. Reproduce one odd development fold

Example:

```bash
modal run modal_app.py::safesense_factorized_swa_odd_fold \
  --validation-subject 3 \
  --seed 22
```

Supported validation subjects:

```text
1, 3, 5, 7
```

Supported seeds:

```text
11, 22, 33
```

Output path pattern in the Modal volume:

```text
/results/safesense-factorized-swa-odd-fold/fold-{subject}/seed-{seed}/
```

Each long run writes `progress.log`.

The full canonical development matrix was already completed and recorded in `research/factorized-swa-final-results.json`.

## 11. Reproduce the frozen final model

```bash
modal run modal_app.py::safesense_factorized_swa_final
```

The function hard-codes the frozen specification:

```text
seed 22
60 epochs
SWA start 45
batch size 32
patience 60 (full budget still run)
GPU cuda/T4
```

Output path:

```text
/results/safesense-factorized-swa-final/
```

Important artifacts:

```text
/results/safesense-factorized-swa-final/result.json
/results/safesense-factorized-swa-final/seed-22/checkpoint.pt
/results/safesense-factorized-swa-final/seed-22/stats.npz
/results/safesense-factorized-swa-final/progress.log
```

Do not alter parameters based on this post-hoc even-subject result.

## 12. Reproduce the final efficiency benchmark

```bash
modal run modal_app.py::safesense_factorized_swa_final_efficiency
```

The benchmark constructs:

- SafeSense eager;
- SafeSense CUDA Graph;
- Fusion-GCN eager;
- Fusion-GCN CUDA Graph.

Protocol:

```text
Tesla T4
FP32
batch size 1
200 warmups
1000 synchronized iterations
model-only
```

Output:

```text
/results/safesense-factorized-swa-final-efficiency/result.json
```

The benchmark requires the frozen SafeSense checkpoint and a preserved Fusion-GCN seed11 checkpoint available to the Modal image.

## 13. CUDA Graph deployment constraints

`CudaGraphBatch1Inference` captures a fixed set of input shapes on CUDA.

Current validated use is batch size 1 with the fixed prepared model shapes.

If a deployment needs multiple batch sizes or sequence/input shapes, capture a separate graph for each required shape or use the eager model for unsupported shapes.

The graph path includes D2D static-buffer input copies in its measured latency but excludes host-to-device transfer.

## 14. Reproduce the development-only depth complementarity ablation

This experiment is diagnostic only. It does not add depth to SafeSense and must not evaluate the even-subject partition.

The probe requires the four preserved seed-22 SafeSense odd-fold checkpoints. Download the exact `checkpoint.pt` file for each validation subject from the private Modal volume into ignored scratch storage. For example:

```bash
mkdir -p scratch/depth_probe/checkpoints
modal volume get safesense-baseline-results \
  safesense-factorized-swa-odd-fold/fold-1/seed-22/artifacts/seed-22/checkpoint.pt \
  scratch/depth_probe/checkpoints/fold-1.pt
```

Repeat for folds 3, 5, and 7, changing both fold numbers in the command.

Then run:

```bash
PYTHONPATH=src .venv/bin/python -m safesense.depth_probe \
  --cache-dir cache \
  --prepared-dir cache/fusion-gcn \
  --checkpoint-root scratch/depth_probe/checkpoints \
  --output research/depth-complementarity-results.json \
  --reference-results research/factorized-swa-final-results.json \
  --device cpu
```

The workflow reconstructs HOG depth features by opening only the 431 odd-subject raw MAT samples. It deliberately does not load the existing all-subject `features.npz` depth array. It verifies that every replayed SafeSense fold accuracy exactly matches the recorded seed-22 development result before accepting the paired comparison.

Frozen probe definition:

```text
Depth: 32 x 92-D HOG
Classifier: LSTM128 -> masked mean -> LayerNorm -> dropout 0.12 -> Linear27
Seeds: 11,22,33
Epochs: 60
SWA: epochs 45-60
AdamW: lr 1e-3, weight decay 1e-4
```

Expected canonical evidence is recorded in `research/depth-complementarity-results.json`; do not select a depth model or run an even-subject depth evaluation from these ablation results.

## 15. Fusion-GCN reproduction

The upstream Fusion-GCN source is not vendored.

`modal_app.py` clones:

```text
https://github.com/mduhme/fusion-gcn
```

and `src/safesense/fusion_gcn.py` applies runtime compatibility behavior needed by the current environment.

Do not call the project result an exact reproduction of the paper. The project has:

- paper-reported accuracy: 94.42%;
- local final-weight three-seed mean: approximately 93.49%.

The baseline is still the closest public-code skeleton+IMU comparison used by this project.

## 15.1 Expanded matched standard-baseline benchmark

The paper's corruption comparison is not restricted to Fusion-GCN. Four fixed conventional baselines are implemented by `scripts/run_matched_comparison_baselines.py`:

- Skeleton-TCN;
- IMU-TCN;
- EarlyFusion-TCN;
- LateFusion-TCN.

The reporting recipe is frozen in the script: train subjects 1/3/5/7, train-only normalization, clean cross-entropy only, 60 epochs, seeds 11/22/33, AdamW `1e-3`, weight decay `1e-4`, batch 32, and equal SWA epochs 45–60. No even-subject checkpoint/hyperparameter selection is performed.

Established T4 reproduction:

```bash
modal run modal_app.py::matched_comparison_baselines
```

The completed run is recorded as Modal app `ap-dgL9lHhPkqJ31lSniTgFVe`. The results volume persists:

```text
/results/matched-standard-baselines/result.json
/results/matched-standard-baselines/predictions.json
/results/matched-standard-baselines/progress.log
```

Canonical committed copies:

```text
research/matched-standard-baseline-results.json
research/matched-standard-baseline-predictions.json
```

These controls were added for reporting after SafeSense was frozen. They must not be used retrospectively to modify SafeSense.

## 16. Houssein reconstruction

Before reproducing or citing the Houssein baseline, read:

```text
research/houssein-reproducibility-audit.md
```

The correct description is **audited reconstruction**, not exact reproduction.

## 17. Reproduce paper statistics, tables and figures

After downloading the frozen selected checkpoint/stats as described in `paper-assets/evidence-map.md`, export its per-sample predictions:

```bash
.venv/bin/python scripts/export_current_predictions.py \
  --prepared-dir cache/fusion-gcn \
  --checkpoint scratch/paper_stats/current/checkpoint.pt \
  --stats scratch/paper_stats/current/stats.npz \
  --output research/safesense-factorized-swa-final-predictions.json \
  --metrics-output scratch/paper_stats/current/current-robustness-metrics.json
```

Then reproduce the committed statistical analysis:

```bash
.venv/bin/python scripts/paper_statistics.py \
  --output research/statistical-analysis-results.json \
  --bootstrap-replicates 10000 \
  --bootstrap-seed 20260816
```

Generate manuscript tables and figures:

```bash
.venv/bin/python scripts/generate_paper_tables.py
.venv/bin/python scripts/generate_paper_figures.py
```

The generated package is indexed by `paper-assets/README.md`.

## 17. Evidence files that should be archived with a paper release

At minimum:

```text
research/factorized-swa-final-results.json
research/factorized-swa-latency-optimization-results.json
research/statistical-analysis-results.json
research/safesense-factorized-swa-final-predictions.json
research/fusion-gcn-final-weight-predictions.json
research/matched-standard-baseline-results.json
research/matched-standard-baseline-predictions.json
research/robustness-efficiency-results.json
research/paired-final-results.json
research/houssein-reproducibility-audit.md
```

Historical negative-result artifacts should remain available because they document model-selection decisions and prevent retrospective cherry-picking.

## 19. Independent verification checklist

A teammate reproducing the final result should verify all of the following:

- prepared sample count = 861;
- split counts = 431 odd / 430 even;
- feature shape = `[N,1,128,22,3]` before SafeSense engineering;
- SafeSense parameter count = 440,562;
- odd-fold run never opens official even arrays;
- final run uses seed 22;
- full 60 epochs execute;
- SWA includes exactly 16 states, epochs 45-60;
- checkpoint and stats are saved;
- final clean metrics reproduce within deterministic/runtime expectations;
- matched standard-baseline artifact contains exactly 4 models × 3 seeds × 25 scenarios × 430 predictions;
- LateFusion-TCN three-seed mean reproduces approximately 92.0930% clean / 68.5478% equal-category corrupted accuracy;
- robustness evaluator uses corruption seed 20260814;
- efficiency protocol is labelled eager or CUDA Graph correctly;
- CUDA Graph prediction-equivalence test remains zero-mismatch;
- no raw UTD-MHAD files are committed.

## 20. What not to do during reproduction

Do not:

- select the best even-subject checkpoint;
- change seed after viewing even-subject accuracy;
- tune SWA start from the final result;
- replace final weights with a test-selected Fusion-GCN checkpoint;
- combine latency from different hardware/scopes in one “fair” ratio;
- silently use the old 32-frame tri-modal feature cache for the current model;
- overwrite old research JSON files with new reruns.

A rerun with a changed protocol should produce a new clearly named artifact.
