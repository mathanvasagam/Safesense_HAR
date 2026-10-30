# Data and Input Protocol

This document describes the dataset and the **actual input path used by the current selected SafeSense model**.

## 1. Active dataset

SafeSense currently uses UTD-MHAD only.

Repository-verified properties:

- 27 action classes;
- 8 subjects;
- nominally 4 trials per action/subject;
- 861 complete aligned samples in the local dataset;
- three known absent/corrupt trial IDs: `a8_s1_t4`, `a23_s6_t4`, `a27_s8_t4`.

The raw local archives/files are kept under `data/` and extracted/generated files under `cache/`. Raw UTD-MHAD data must **not** be committed or redistributed with the paper repository.

## 2. Subject protocol

### Official cross-subject partition

The project follows the standard odd/even subject split encoded in the repository:

- training-side odd subjects: 1, 3, 5, 7;
- official even subjects: 2, 4, 6, 8.

Counts:

- odd: 431 samples;
- even: 430 samples.

### Development protocol for the selected model

The selected factorized-SWA design was audited with **leave-one-odd-subject-out** development.

For each validation subject in `{1, 3, 5, 7}`:

- that odd subject is held out as development validation;
- the other three odd subjects are used for development training;
- even-subject arrays are not loaded by the development workflow.

This produces four subject-held-out development folds and is substantially stricter than the older project convention of training on subjects 1,3,5 and validating only on subject 7.

The selected seed 22 was frozen from the complete odd-fold development evidence before the current full-odd replay.

### Important post-hoc caveat

The even-subject partition was observed by older project model cycles. Consequently, although the current factorized-SWA model was frozen using odd-fold development only, its even-subject outcome must be described as **post-hoc frozen evaluation**, not as an untouched never-before-seen blind test.

This distinction belongs in the paper methodology/limitations section.

## 3. Two preprocessing paths exist in the repository

This is the most important data-pipeline distinction for paper authors.

### 3.1 Earlier tri-modal reconstruction path

`src/safesense/preprocessing.py` implements a 32-frame path containing:

- depth HOG features;
- flattened 20-joint skeleton coordinates;
- 6-channel IMU;
- visual motion-based frame selection;
- proportional visual-to-IMU mapping.

This path was created for the Houssein-style audited reconstruction.

**It is not the input path of the current selected SafeSense model.**

### 3.2 Current selected skeleton+IMU path

The current selected SafeSense and Fusion-GCN comparison starts from Fusion-GCN-compatible prepared arrays created by `src/safesense/fusion_gcn.py`.

The prepared shape is:

```text
[N, 1, 128, 22, 3]
```

where:

- 20 entries are skeleton joints;
- 2 entries are IMU pseudo-joints;
- each entry has 3 channels;
- temporal length is 128.

The prepared-data audit is stored at:

`cache/fusion-gcn/fusion_gcn_audit.json`

It records the shape `[1, 128, 22, 3]` per sample and excludes RGB/depth from the Fusion-GCN comparison path.

## 4. Fusion-GCN-compatible preparation

`src/safesense/fusion_gcn.py` reconstructs the upstream UTD-MHAD skeleton+IMU representation without vendoring the upstream repository.

For skeleton:

1. load `d_skel` from the MAT file;
2. transpose to temporal layout `[T,20,3]`;
3. pad/extend to 128 frames using the upstream-compatible behavior;
4. normalize around the reference joint;
5. rotate the sequence into a body-oriented coordinate system consistent with the upstream implementation.

For IMU:

1. load `d_iner`, shape `[T_imu,6]`;
2. resample to the actual skeleton duration using nearest proportional indices;
3. reshape 6 channels into two 3-D pseudo-joints;
4. place those pseudo-joints after the 20 skeleton joints.

The resulting fused sample is `[1,128,22,3]`.

The dataset files contain no usable hardware timestamps. Therefore synchronization/alignment claims must not be phrased as recovery of unknown real sensor-clock lag.

## 5. SafeSense feature construction

The current SafeSense model does not feed the raw 22-node tensor directly into its classifier. `_feature_arrays()` in `src/safesense/safesense.py` derives separate skeleton and IMU tensors.

### 5.1 Skeleton channels

From the first 20 joints:

#### Position

```text
20 joints x 3 coordinates = 60 channels
```

#### Temporal first difference

For frame `t > 0`:

```text
Delta p_t = p_t - p_(t-1)
```

The first frame receives zero differences.

This gives another 60 channels.

#### Bone vectors

The code defines 19 Kinect skeleton body edges in `BONE_EDGES`. For each edge:

```text
bone(child,parent) = joint(child) - joint(parent)
```

```text
19 edges x 3 coordinates = 57 channels
```

Total skeleton representation:

```text
60 position + 60 temporal difference + 57 bone = 177 channels/frame
```

Shape:

```text
[B,128,177]
```

### 5.2 IMU channels

The two prepared IMU pseudo-joints are flattened back to six physical channels:

```text
6 raw channels
```

A six-channel temporal first difference is added:

```text
6 raw + 6 difference = 12 channels/frame
```

Shape:

```text
[B,128,12]
```

## 6. Masks and availability

A temporal mask is inferred from the prepared IMU raw channels:

```text
mask_t = 1 if sum(abs(imu_raw_t)) > 0 else 0
```

The model/evaluator can also use paired masks with independent skeleton and IMU validity:

```text
[B,T,2]
```

The final dimension is:

- index 0: skeleton validity;
- index 1: IMU validity.

`SafeSenseDataset` computes a two-value modality-availability vector by checking whether each modality has at least one valid frame.

Masked values are explicitly zeroed after normalization.

## 7. Training-only normalization

Normalization statistics are never fitted on a held-out subject for development and are not fitted using the official even partition.

For each feature channel:

```text
z = (x - mean_train) / std_train
```

The code uses only frames selected as valid by the training mask. Very small standard deviations are replaced with 1.0 to avoid numerical instability.

The stored final model statistics are part of the frozen deployment artifact:

`/results/safesense-factorized-swa-final/seed-22/stats.npz`

The checkpoint and those statistics must be treated as a pair.

## 8. Derivative sanitization and gap repair

The current model has explicit safeguards for corrupted/missing frames.

### 8.1 Derivative sanitization

If either side of a temporal transition is invalid, the engineered derivative channels are zeroed for that transition. This avoids creating a false large velocity merely because a frame disappeared.

Affected channels:

- skeleton first differences: channels 60:120;
- IMU first differences: channels 6:12.

### 8.2 Internal gap repair

`_repair_internal_gaps` fills only short internal gaps when there are valid frames on both sides within radius 3. It uses a local valid-value average and preserves missing boundaries.

This operation is part of the model's input handling and is distinct from the external robustness corruption generator.

## 9. Labels

Fusion preparation writes zero-based labels using:

```text
label = action_id - 1
```

Therefore:

```text
a1  -> class 0
...
a27 -> class 26
```

The model outputs 27 logits and evaluation uses `argmax`.

## 10. What the selected model does not consume

The selected `SafeSenseFactorized` model does not consume:

- RGB frames;
- depth frames;
- HOG92 depth descriptors;
- the old 32-frame `features.npz` tri-modal representation;
- action text labels;
- corruption labels at inference;
- subject identity as an inference input.

## 11. Files that define the protocol

Primary source files:

- `src/safesense/dataset.py` - UTD-MHAD indexing and subject split labels;
- `src/safesense/fusion_gcn.py` - matched 128-frame skeleton+IMU preparation;
- `src/safesense/safesense.py::_feature_arrays` - 177/12-dimensional feature construction;
- `src/safesense/safesense.py::SafeSenseDataset` - normalization/masking/availability;
- `src/safesense/robustness.py` - evaluation-time corruption generation.

Evidence files:

- `cache/fusion-gcn/fusion_gcn_audit.json`;
- `research/factorized-swa-final-results.json`;
- `research/robustness-efficiency-results.json`.

## 12. Paper wording guidance

Recommended:

> We evaluate isolated 27-class action recognition on UTD-MHAD using skeleton and inertial modalities only. Samples are prepared into a 128-step representation compatible with the Fusion-GCN skeleton-inertial pipeline. SafeSense derives per-frame joint position, motion, bone-vector, raw inertial, and inertial-motion features. All normalization statistics are fitted on the corresponding training subjects only.

Avoid:

- “SafeSense uses depth and IMU”;
- “SafeSense is tri-modal”;
- “32 motion-selected frames are fed to SafeSense”;
- “real hardware lag is corrected”;
- “the even-subject split was never observed before.”
