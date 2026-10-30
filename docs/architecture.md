# SafeSense Architecture

This document describes the **current selected SafeSense model**, implemented as `SafeSenseFactorized` in `src/safesense/safesense.py`.

Historical SafeSense architectures remain in the same file for experiment reproducibility. They are not the selected architecture and should not be merged into the paper's method description.

## 1. Design objective

The selected model is designed for three simultaneous goals:

1. high cross-subject clean classification accuracy;
2. robustness when skeleton/IMU timing or availability is imperfect;
3. compact, low-latency deployment.

The architecture uses explicit body topology for skeleton reasoning, temporal convolutions for local/multi-scale motion, and factorized attention for longer temporal and cross-sensor interaction.

## 2. Model identity

Canonical selected model:

```text
Class:             SafeSenseFactorized
Width:             96
Factorized blocks: 2
Classes:           27
Parameters:        440,562
Modalities:        skeleton + IMU
RGB:               no
Depth:             no
```

The selected checkpoint is the seed-22 late-SWA checkpoint produced by 60-epoch training and equal averaging of epochs 45-60.

The architecture itself is unchanged by SWA and unchanged by CUDA Graph deployment.

## 3. Input tensors

After preparation and normalization:

```text
skeleton: [B,128,177]
imu:      [B,128,12]
mask:     [B,128] or [B,128,2]
availability: [B,2] (optional; inferred if omitted)
```

The 177 skeleton channels are:

- 60 joint coordinates;
- 60 first-difference coordinates;
- 57 bone-vector coordinates.

The 12 IMU channels are:

- 6 prepared raw inertial channels;
- 6 temporal first differences.

See `data.md` for construction details.

## 4. Input sanitation

Before learned encoding, the forward pass:

1. resolves independent skeleton/IMU masks;
2. infers modality availability if needed;
3. zeros invalid data;
4. suppresses first-difference channels across invalid transitions;
5. repairs short internal missing runs with nearby valid information.

This prevents missing data from masquerading as motion.

## 5. Skeleton stream

The skeleton is intentionally processed by **two complementary branches**.

### 5.1 Flat temporal branch

`SafeSenseTCNBranch(SKELETON_INPUTS=177, width=64)` projects the complete engineered feature vector to 64 channels with a 1x1 Conv1d.

It then applies residual depthwise-separable temporal blocks with dilation sequence:

```text
1, 2, 4
```

Each residual block contains:

```text
depthwise Conv1d(kernel=5, dilation=d)
-> pointwise Conv1d(1x1)
-> GroupNorm
-> GELU
-> Dropout
-> residual add
```

The flat branch captures temporal patterns across position, motion, and bone-vector channels without imposing a graph structure on every operation.

Output:

```text
[B,64,128]
```

### 5.2 Explicit skeleton-topology branch

`SkeletonTopologyBranch` uses the fixed 20-joint UTD-MHAD Kinect topology defined by 19 body edges.

For each joint at each frame, it extracts:

- 3 position channels;
- 3 first-difference motion channels.

The per-joint six-dimensional vector is projected as:

```text
Linear(6,96) -> LayerNorm(96) -> GELU
```

A normalized adjacency matrix is built once from:

- self-connections;
- the 19 bidirectional skeleton edges.

The adjacency is symmetrically degree-normalized.

### 5.3 Graph message passing

Three `TopologyGraphBlock(96)` layers are applied.

For node feature `h_i`, each block constructs a neighbor aggregate using the fixed adjacency, then combines:

```text
self_linear(h_i) + neighbor_linear(sum_j A_ij h_j)
```

The update is normalized, GELU-activated, dropped out, and added residually to the node representation.

### 5.4 Joint attention pooling

After graph propagation, each joint receives a learned scalar score from `Linear(96,1)`.

A softmax over the 20 joints produces frame-wise joint weights. The weighted joint representation becomes one 96-D skeleton token per frame.

The mask is applied, then another 96-wide temporal TCN with dilations 1,2,4 is used.

Topology-branch output:

```text
[B,96,128]
```

### 5.5 Skeleton branch fusion

The two skeleton outputs are concatenated:

```text
64 + 96 = 160 channels
```

Then projected:

```text
Conv1d(160,96,1)
-> GroupNorm(16,96)
-> GELU
```

Final pre-reduction skeleton stream:

```text
[B,96,128]
```

## 6. IMU stream

The IMU encoder is deliberately smaller because the input has only 12 channels.

```text
SafeSenseTCNBranch(12,64)
```

uses the same residual depthwise-separable temporal structure with dilations 1,2,4.

It is then projected to the shared fusion width:

```text
Conv1d(64,96,1)
-> GroupNorm(16,96)
-> GELU
```

Output:

```text
[B,96,128]
```

## 7. Efficient temporal reduction

Full attention over 128 steps would add unnecessary batch-1 runtime. Each modality therefore passes through `EfficientTemporalReducer(96)`.

It has two reduction stages. Each stage is:

```text
depthwise Conv1d(kernel=5, stride=2, padding=2)
-> pointwise Conv1d(1x1)
-> GroupNorm
-> GELU
```

Masks are reduced using max pooling so a reduced token is valid when its receptive field contains valid source information.

Temporal length becomes:

```text
128 -> 64 -> 32
```

Both modality streams are now:

```text
[B,96,32]
```

This reduction is an important part of the efficiency design: the attention stage operates on 32 temporal positions rather than 128.

## 8. Factorized time/modality fusion

The model stacks skeleton and IMU as:

```text
[B,2,32,96]
```

with a corresponding validity tensor:

```text
[B,2,32]
```

Two identical `FactorizedTimeModalityBlock`s are applied.

Each block separates temporal interaction from modality interaction.

### 8.1 Temporal attention

Tokens are reshaped so each modality receives its own 32-step temporal sequence:

```text
[B*2,32,96]
```

The sequence is LayerNorm-normalized and passed through 4-head `MultiheadAttention(96, heads=4)` with invalid tokens key-masked.

Conceptually:

```text
skeleton t1 <-> ... <-> t32
imu      t1 <-> ... <-> t32
```

The attention update is residual and re-masked.

The code explicitly uses PyTorch's MATH SDPA backend for this attention implementation.

### 8.2 Modality attention

The tensor is then rearranged to make the two modalities the attention axis independently at every time step:

```text
[B*32,2,96]
```

Another 4-head multihead-attention operation lets skeleton and IMU exchange information at the same reduced temporal position.

Conceptually:

```text
time 1: skeleton <-> IMU
time 2: skeleton <-> IMU
...
time 32: skeleton <-> IMU
```

Again, invalid modalities are key-masked.

### 8.3 Feed-forward network

After temporal and modality attention, a residual feed-forward block is applied:

```text
LayerNorm(96)
-> Linear(96,192)
-> GELU
-> Dropout
-> Linear(192,96)
-> Dropout
-> residual add
```

The final tokens are re-masked.

## 9. Expert branches used during training

After factorized attention, skeleton and IMU tokens are separated.

Each modality is pooled using `SafeSense._pool`, which concatenates:

- masked temporal mean;
- masked temporal maximum.

For width 96, each modality embedding is 192-D.

Two auxiliary classifier heads exist:

```text
skeleton_expert: LayerNorm(192) -> Linear(192,27)
imu_expert:      LayerNorm(192) -> Linear(192,27)
```

These expert logits support the auxiliary training objective. They are also useful for analysis, but the deployed activity prediction is produced by the fused classifier.

The model also stores mean pooled skeleton/IMU alignment embeddings for the cross-modal alignment loss.

## 10. Final fused representation

For each reduced time token, available modality tokens are averaged:

```text
fused_t = sum(valid_m * token_m) / number_of_valid_modalities
```

This naturally handles one missing modality.

The fused 32-step sequence is again mean+max pooled, yielding:

```text
192-D final embedding
```

Classifier:

```text
LayerNorm(192)
-> Linear(192,27)
```

Main output:

```text
logits: [B,27]
```

Prediction:

```text
argmax(logits, dim=1)
```

## 11. Returned modality weights

The model also returns a two-value weight vector computed from modality availability:

```text
weights = availability / sum(availability)
```

Examples:

```text
both sensors present -> [0.5, 0.5]
only skeleton        -> [1.0, 0.0]
only IMU             -> [0.0, 1.0]
```

If both are absent, the fallback returned weights are `[0.5,0.5]`; no valid evidence exists in that case, so classification performance is not expected to be meaningful.

These are availability weights, not softmax class confidence and not a learned reliability gate in the current selected architecture.

## 12. Parameter count

The canonical parameter count is:

```text
440,562 trainable parameters
```

Fusion-GCN reproduction reference:

```text
3,456,631 parameters
```

Reduction:

```text
87.2546%
```

The selected SafeSense FP32 checkpoint is 1,814,707 bytes versus Fusion-GCN's preserved 14,066,827-byte checkpoint, a reduction of approximately 87.10%.

## 13. CUDA Graph deployment wrapper

The CUDA Graph path in `src/safesense/efficiency.py` is **not an architectural change**.

`CudaGraphBatch1Inference`:

- requires CUDA;
- supports the captured fixed input shapes;
- copies new device tensors into static input buffers;
- replays a captured CUDA graph;
- returns the static graph output.

During graph capture only, two input-mask helpers are temporarily replaced by mathematically equivalent branch-free implementations so capture does not encounter host-side data-dependent synchronization. The original Python functions are restored immediately after capture.

No weights, layers, logits, or training rules are changed.

Evidence of equivalence:

- 430 even-subject samples;
- 25 scenarios;
- 10,750 predictions;
- 0 eager/CUDA-Graph prediction mismatches;
- representative maximum absolute logit difference 0.0;
- exact aggregate metric preservation.

## 14. Why parameter reduction did not initially give a proportional latency reduction

Profiling showed the eager model launched approximately **427 CUDA kernels per batch-1 forward**. The CPU spent approximately 2.875 ms per forward in `cudaLaunchKernel` self time in the profiling run.

The bottleneck was therefore not simply matrix FLOPs or parameter count. It was a large number of small operations:

- matrix multiplications;
- copies;
- normalization kernels;
- depthwise/pointwise convolutions;
- pointwise activation/arithmetic kernels.

CUDA Graph replay reduces host launch overhead and changes SafeSense latency from roughly 9.65 ms eager to 1.732 ms optimized on the matched T4 benchmark.

## 15. Method-description boundaries

The paper can accurately describe the current architecture as:

> A graph-aware dual-stream skeleton-inertial network that combines residual depthwise temporal convolution, fixed human-body topology encoding, efficient temporal token reduction, and factorized temporal/modality attention.

Do not describe it as:

- a depth model;
- an RGB model;
- the historical temporal33+adapter27 model;
- the historical early-exit ensemble;
- an adaptive learned modality gate (the current returned fusion weights are availability-based);
- a newly invented generic factorized-attention concept.

## 16. Source-of-truth code

The principal definitions are in:

- `src/safesense/safesense.py::ResidualDepthwiseTCNBlock`
- `src/safesense/safesense.py::SafeSenseTCNBranch`
- `src/safesense/safesense.py::TopologyGraphBlock`
- `src/safesense/safesense.py::SkeletonTopologyBranch`
- `src/safesense/safesense.py::EfficientTemporalReducer`
- `src/safesense/safesense.py::FactorizedTimeModalityBlock`
- `src/safesense/safesense.py::SafeSenseFactorized`
- `src/safesense/efficiency.py::CudaGraphBatch1Inference`
