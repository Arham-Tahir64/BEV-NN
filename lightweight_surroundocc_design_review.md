# Lightweight Camera-Only 3D Occupancy: Implementation Guide

Camera-only 3D semantic occupancy prediction | PyTorch | nuScenes | RTX 4060, 8 GB

Prepared 13 October 2026. Engineering recommendations and memory estimates are not validated measurements on the target GPU.

This guide specifies a lightweight, single-frame camera-only network that predicts semantic occupancy from six surround-view images. The implementation targets PyTorch, nuScenes with Occ3D annotations, and an RTX 4060 with 8 GB VRAM.

The first version uses a shared pretrained image encoder, geometry-based voxel sampling, masked camera fusion, and a compact 3D decoder. A second version introduces local image-to-voxel cross-attention after the baseline is verified.

## 1. Data and geometry contract

Define the following conventions before implementing the network. They form the interface shared by data loading, projection, prediction, and evaluation.

| Contract | Required decision |
|---|---|
| Reference frame | Which coordinate frame contains the occupancy grid? |
| Axes | What do positive X, Y, and Z mean? |
| Tensor order | How do physical XYZ axes map to tensor dimensions? |
| Bounds | Exact metric extent and voxel size |
| Labels | Class IDs, free-space ID, and supervision masks |
| Cameras | Fixed ordering, intrinsics, extrinsics, and timestamps |
| Augmentation | How resizing/cropping changes projection |
| Evaluation | Class set, visibility policy, spatial extent, and split |

Put these decisions in the configuration and README, and enforce them with assertions and tests.

## 2. Architecture scope and development stages

**Baseline:** six camera images are encoded by one shared ResNet-18. Fixed voxel centres sample the image features through calibrated projection. Valid camera features are averaged into a coarse 3D volume, refined with a small 3D CNN, and classified into semantic occupancy logits. The logits are upsampled to the target label grid.

**Attention extension:** coarse voxel features and position encodings generate queries for local image cross-attention. This is the planned lightweight SurroundOcc-inspired stage. SurroundOcc supplies the architectural inspiration of image-to-volume attention and volumetric decoding. [1]

The baseline uses one time step, one image feature scale, 32 lifted feature channels, and a cropped spatial range. Temporal fusion, global voxel attention, a large 3D U-Net, custom CUDA kernels, and multiple auxiliary losses are outside the initial scope.

Inference requires camera images and calibration/coordinate metadata. LiDAR-derived labels are permitted for training; LiDAR measurements and annotation visibility masks are not model inputs at inference.

## 3. Dataset, labels, and spatial extent

Use **Occ3D-nuScenes** initially. It supplies a native `200 x 200 x 16` grid, X/Y bounds of `[-40, 40]` metres, Z bounds of `[-1, 5.4]` metres, and `0.4 m` voxels. There are 18 label IDs, including `17` for free space. Each label file contains `semantics`, `mask_lidar`, and `mask_camera`. Its documented evaluation uses `mask_camera` to exclude unobserved voxels. Training mask choices are configurable; initially use the camera mask for supervision too. [2]

**Do not treat unknown or unsupervised voxels as free space.** Do not assume occupancy class IDs equal raw nuScenes lidarseg IDs; explicitly validate the annotation mapping.

For the initial GPU configuration, use a central crop of native labels:

| Property | Initial choice |
|---|---|
| X/Y bounds | `[-20, 20] m` |
| Z bounds | `[-1, 5.4] m` |
| Output resolution | Native `0.4 m` |
| Output grid in XYZ order | `100 x 100 x 16` |
| Internal feature grid in XYZ order | `50 x 50 x 8` |
| Internal feature spacing | `0.8 m` |

This avoids inventing a coarse-label aggregation policy immediately. Train against native labels in the cropped region while keeping expensive features at coarser resolution.

**This is a custom cropped task. Its scores are not directly comparable to full-range Occ3D results.** Expand to the official range later for benchmark comparisons.

## 4. Module responsibilities

| Component | Responsibility |
|---|---|
| `configs/baseline.yaml` | Baseline data, geometry, model, training, and evaluation settings |
| `configs/debug.yaml` | Small deterministic configuration for smoke tests and frame overfitting |
| `config.py` | Load configuration and validate grid dimensions, bounds, and class settings |
| `geometry/voxel_grid.py` | Metric voxel centres, grid bounds, flattening, and tensor-axis conversion |
| `geometry/projection.py` | Coordinate transforms, camera projection, validity masks, and sampling coordinates |
| `data/nuscenes_dataset.py` | Match six images, calibrations, annotations, and sample metadata |
| `data/occupancy_labels.py` | Load class labels and masks; validate IDs, crop, and convert axis order |
| `data/transforms.py` | Resize and normalize images while recording geometry changes |
| `models/image_encoder.py` | Shared pretrained encoder and trainable channel projection |
| `models/voxel_lifter.py` | Sample camera features at projected voxel centres and fuse valid samples |
| `models/voxel_decoder.py` | Coarse residual 3D refinement, class prediction, and logit upsampling |
| `models/occupancy_net.py` | Connect encoder, lifting, decoder, and output shape contracts |
| `losses.py` | Masked cross-entropy and optional controlled class weighting |
| `metrics.py` | Dataset-level confusion counts and documented IoU calculations |
| `visualization.py` | Projection overlays, occupancy rendering, and error views |
| `scripts/inspect_sample.py` | Inspect one sample and verify geometry before training |
| `scripts/train.py` | Optimization, validation, checkpointing, and reproducibility |
| `scripts/evaluate.py` | Run the fixed held-out evaluation protocol |
| `scripts/visualize.py` | Export or display predictions alongside ground truth |
| `scripts/profile_model.py` | Measure training memory, throughput, and inference latency |
| `tests/` | Known-answer checks for geometry, labels, sampling, masks, and metrics |

Keep deterministic geometry separate from dataset adapters and trainable modules. Introduce a separate attention module when that extension is implemented.

## 5. Baseline model components

Use one shared encoder, explicit geometric sampling, a small 3D CNN, and one semantic classifier.

### Image encoder

Start with an ImageNet-pretrained ResNet-18, stopping at its stride-16 stage. Project its 256 channels to 32 using a trainable `1 x 1` convolution.

- Pretraining supplies useful visual features while you learn the 3D components.
- A familiar backbone is easier to inspect than a custom lightweight encoder.
- Stride 16 retains more spatial detail than using only stride 32.
- Thirty-two feature channels keep the volumetric stages small.

Freeze the backbone first, including BatchNorm running statistics. Unfreeze its last used stage after the rest of the model trains correctly. Training the image encoder from scratch adds substantial optimization difficulty without helping you understand camera geometry. Implement and train the lifting projection, decoder, classifier, and later attention components yourself.

### Voxel representation

Generate fixed metric voxel centres and normalized XYZ positional features. These are spatial locations, not automatically learned queries. Independent learned embeddings are unnecessary for baseline sampling and tie parameters to a particular grid. Later, generate attention queries from positional encodings and existing voxel features.

### Image-to-voxel lifting

Project each coarse voxel centre into every camera and bilinearly sample the corresponding image feature. For voxel v:

```text
f_v = sum_c [m_vc * F_c(pi_c(p_v))] / max(1, sum_c m_vc)
```

- `p_v`: voxel centre in the occupancy reference frame.
- `pi_c`: projection into camera c.
- `F_c`: camera feature map.
- `m_vc`: positive-depth and image-bounds validity mask.

Include normalized XYZ and a valid-camera count alongside fused features. Zero camera coverage should be explicit rather than indistinguishable from a valid zero feature.

**Critical limitation:** projection into an image does not establish that the image observes that voxel. Multiple positions along a ray sample similar evidence, including positions behind objects. Projection validity is not an occlusion test.

Depth-aware backward projection addresses this ambiguity by weighting features according to predicted depth consistency; this is a useful later idea from FB-BEV. [3] Masked averaging is a debuggable baseline, not a final solution to depth reasoning.

### 3D decoder

Use two small residual blocks with 32 channels and GroupNorm, operating on the coarse grid. Their role is to combine neighbouring evidence and learn local scene structure.

1. Refine coarse voxel features.
2. Produce 18 coarse semantic logits per voxel.
3. Trilinearly upsample logits to the target grid.
4. Apply cross-entropy against fine labels.

Interpolation cannot recover arbitrary fine geometry. Thin structures and object boundaries will suffer. Evaluate a later learned upsampling stage against this baseline.

### Semantic head and loss

One 18-class classifier is sufficient. Separate binary occupancy and semantic heads initially add loss balancing and potentially contradictory outputs. With an explicit free class:

```text
P(occupied) = 1 - P(free)
```

Start with masked cross-entropy and inspect class frequencies. If free-space prediction dominates, introduce moderate, capped class weights derived from the training split. Defer Lovasz, geometric scaling, semantic scaling, and auxiliary losses until a specific failure justifies them. A loss stack can hide a geometry bug rather than fix it.

## 6. Camera geometry and visibility masks

nuScenes keyframe sensors have nearby timestamps, rather than necessarily identical capture times. The official devkit transforms points through the relevant sensor calibration and ego poses to account for different sensor timestamps. [4, 5]

Let `T[a <- b]` mean a transform mapping coordinates from frame b into frame a:

```text
p_cam,c = T[cam,c <- ego(t_c)]
        * T[ego(t_c) <- global]
        * T[global <- ref]
        * p_ref
```

Then apply camera intrinsics and the image resize/crop transform. The annotation adapter must establish what `ref` means for the actual label source. **Do not assume ego frame and LiDAR frame are interchangeable.**

- Calculate voxel centres with the half-voxel offset.
- Reject negative or near-zero camera depth before division.
- Track resizing/cropping explicitly.
- Keep projection calculations in FP32.
- Define one feature-coordinate and pixel-centre convention.
- Test normalization into `grid_sample` coordinates.

PyTorch's `grid_sample` uses normalized coordinates and has explicit `align_corners` semantics. A visually small coordinate offset can consistently corrupt lifted features. [6]

| Mask | Purpose |
|---|---|
| Projection validity | Can this voxel's image feature be sampled from this camera? |
| Supervision validity | Should this voxel contribute to training loss? |
| Evaluation validity | Should this voxel contribute to reported metrics? |

Neither annotation mask should become an inference input.

## 7. Exact baseline tensor flow

Use image tensors `[B, N, C, H, W]`, 3D feature tensors `[B, C, Z, Y, X]`, and target tensors `[B, Z, Y, X]`. Explicitly convert annotation arrays into this order.

| Stage | Tensor shape for batch size one |
|---|---|
| Six resized RGB images | `[1, 6, 3, 288, 512]` |
| Flatten cameras into batch | `[6, 3, 288, 512]` |
| Shared stride-16 encoder | `[6, 256, 18, 32]` |
| Trainable channel projection | `[6, 32, 18, 32]` |
| Restore camera axis | `[1, 6, 32, 18, 32]` |
| Coarse metric voxel centres | `[20000, 3]` |
| Projected sampling coordinates | `[1, 6, 20000, 2]` |
| Projection validity | `[1, 6, 20000]` |
| Sampled camera features, conceptually | `[1, 6, 20000, 32]` |
| Masked camera fusion | `[1, 20000, 32]` |
| Reshape to volume | `[1, 32, 8, 50, 50]` |
| Append XYZ and camera-count channel | `[1, 36, 8, 50, 50]` |
| Channel projection and 3D refinement | `[1, 32, 8, 50, 50]` |
| Coarse class logits | `[1, 18, 8, 50, 50]` |
| Upsampled class logits | `[1, 18, 16, 100, 100]` |
| Labels and supervision mask | `[1, 16, 100, 100]` |
| Predicted class IDs | `[1, 16, 100, 100]` |

Camera features need not be materialized for all cameras simultaneously; accumulate sampled features and valid counts if useful.

The network predicts a dense categorical volume. A visualization renders selected occupied voxels, typically coloured by semantic class. It is not inherently a watertight mesh or an exact geometric reconstruction.

## 8. Feasibility on 8 GB

The coarse feature volume, 32-channel decoder, frozen initial backbone, and batch size of one make this baseline a plausible 8 GB target. Validate the complete training step on the target GPU before increasing spatial range, image resolution, or model width.

| Tensor or operation | Approximate FP16 storage |
|---|---|
| Coarse `32 x 8 x 50 x 50` feature tensor | 1.22 MiB |
| Cropped fine `32 x 16 x 100 x 100` feature tensor | 9.77 MiB |
| Full-range `64 x 16 x 200 x 200` feature tensor | 78.1 MiB |
| Dense attention matrix over 20,000 voxels, one head | 763 MiB |
| Dense attention matrix over 160,000 voxels, one head | 47.7 GiB |

These are individual tensor-storage calculations, excluding saved activations, gradients, optimizer states, workspaces, and other layers. They are not total model-memory measurements.

Likely bottlenecks:

- **Image backbone activations:** six views multiply processing and stored activations.
- **Fine-resolution 3D convolutions:** many intermediate volumes and substantial arithmetic.
- **Attention:** global voxel attention grows quadratically.
- **Sampling expansion:** multiple levels, heads, and points create large intermediates.
- **Data throughput:** decoding six images and loading labels can leave the GPU idle.

### Initial configuration

| Setting | Starting value |
|---|---|
| Image size | `288 x 512`, preserving the standard image aspect ratio |
| Camera count / temporal frames | 6 / 1 |
| Backbone | ImageNet-pretrained ResNet-18 through stride 16 |
| Backbone training | Frozen initially; frozen BatchNorm statistics |
| Lifted feature channels | 32 |
| Internal grid in XYZ | `50 x 50 x 8` |
| Output grid in XYZ | `100 x 100 x 16` |
| Fusion | Masked mean |
| Decoder | Two small residual 3D blocks |
| Output classes | 18 |
| Precision | Mixed precision; FP32 geometry |
| Physical batch size | 1 |
| Gradient accumulation | 4 steps |
| Optimizer | AdamW |
| Initial learning rate | `3e-4` for new modules |
| Later backbone learning rate | Start around `3e-5` for the unfrozen stage |
| Loss | Masked cross-entropy |
| Initial augmentation | Deterministic resize and normalization |

Learning rates are starting hypotheses, not tuned values. Gradient accumulation increases effective batch size but does not make an oversized forward pass fit.

Before long training, measure several warm-up steps, a complete forward/backward/optimizer step, peak allocated and reserved CUDA memory, training throughput, and inference latency separately. Allow for optimizer-state allocation on the first update. Do not promise a training duration or FPS before these measurements.

## 9. Project directory structure

Create files as milestones require them.

```text
lightweight-surroundocc/
├── README.md
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── baseline.yaml
│   └── debug.yaml
├── src/
│   └── lightocc/
│       ├── __init__.py
│       ├── config.py
│       ├── geometry/
│       │   ├── __init__.py
│       │   ├── voxel_grid.py
│       │   └── projection.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── nuscenes_dataset.py
│       │   ├── occupancy_labels.py
│       │   └── transforms.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── image_encoder.py
│       │   ├── voxel_lifter.py
│       │   ├── voxel_decoder.py
│       │   └── occupancy_net.py
│       ├── losses.py
│       ├── metrics.py
│       └── visualization.py
├── scripts/
│   ├── inspect_sample.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── profile_model.py
└── tests/
    ├── test_voxel_grid.py
    ├── test_projection.py
    ├── test_labels.py
    ├── test_voxel_lifter.py
    └── test_loss_metrics.py
```

- **Geometry:** deterministic coordinate operations.
- **Data:** dataset-specific loading and label interpretation.
- **Models:** differentiable inference components.
- **Scripts:** runnable entry points.
- **Metrics:** independent of the training loop.

`voxel_lifter.py` initially contains sampling and mean fusion. Add a separate attention module only when implementing a genuinely different fusion strategy. Avoid registries, generic trainer frameworks, and elaborate plugin systems initially.

## 10. Implementation milestones and completion gates

### Milestone 1: Geometry contract

**Files:** `voxel_grid.py`, `projection.py`, and geometry tests.

Implement XYZ centres to camera coordinates to pixels. Verify synthetic known-answer projection, transform round trips, and correct boundary centres. Defer neural networks.

### Milestone 2: One real sample

**Files:** `occupancy_labels.py`, `nuscenes_dataset.py`, `transforms.py`, and `inspect_sample.py`.

Load images, calibrations, labels, and masks. Render GT and overlay projected occupied centres in all cameras; confirm resize alignment. Projected GT behind surfaces can be confusing because of occlusion, so inspect selected slices or near-surface points. Defer random augmentation.

### Milestone 3: Feature lifting

**Files:** `image_encoder.py` and `voxel_lifter.py`.

Convert six feature maps into `[B, 32, 8, 50, 50]`. Verify synthetic image ramps give expected sampled values, invalid cameras have no effect, and gradients reach the trainable projection. Defer attention.

### Milestone 4: Complete network

**Files:** `voxel_decoder.py`, `occupancy_net.py`, `losses.py`, and `train.py`.

Convert coarse features into fine logits and masked loss. Overfit one frame, then 4-8 frames, and inspect per-class predictions. Tiny-frame overfitting is a wiring test, not proof of generalization. Failure may indicate geometry errors, incorrect labels, excessive regularization, or insufficient decoder capacity. Defer long training.

### Milestone 5: Trustworthy evaluation

**Files:** `metrics.py`, `evaluate.py`, and `profile_model.py`.

Produce a confusion matrix and performance report. Perfect predictions should score correctly, masked errors should not count, and a complete training step must fit the GPU. Defer paper comparisons.

### Milestone 6: Small generalization experiment

**Work:** training configuration and visualization improvements.

Use scene-disjoint training and validation. Beat an all-free baseline on occupied IoU and inspect validation errors. Split by scene rather than mixing adjacent frames randomly; preserve the official split relationship when using subsets. Defer architectural expansion.

### Milestone 7: Local attention experiment

**Work:** add a local attention implementation and configuration.

Coarse queries sample local projected image features to refine the volume. Require improvement versus mean fusion at a measured memory/runtime cost. Defer temporal fusion.

### Milestone 8: Expand scope

**Work:** learned upsampling, an optional feature pyramid, and larger spatial range.

Produce more detailed or broader occupancy. Controlled ablations must justify each addition. Defer anything without evidence.

### First three implementation files

1. **`geometry/voxel_grid.py`:** bounds, centres, flattening order, XYZ-to-tensor conversion.
2. **`geometry/projection.py`:** reference-to-camera transforms, image transforms, validity, sampling coordinates.
3. **`data/occupancy_labels.py`:** annotation loading, class validation, crop, masks, tensor ordering.

Write small tests alongside them. Then implement the dataset loader and inspect a real sample. Implement the attention extension after the sampling baseline passes its verification gates.

## 11. Local attention extension and experiments

Once the baseline works, add one local cross-attention block at coarse voxel resolution:

- Queries derived from voxel position and current voxel features.
- Reference points given by geometric projection.
- Initially one image feature scale.
- Four learned sample offsets per camera.
- One or two heads.
- Masked normalization over valid samples.
- Residual connection and a small feed-forward network.

Handle the all-invalid case explicitly to avoid softmax over entirely masked entries. This lets the network choose nearby image evidence instead of averaging one sample from every valid camera. It does not automatically solve depth ambiguity.

| Experiment | Question answered |
|---|---|
| Mean fusion to learned camera weights | Does choosing among camera views help? |
| Learned weights to local offsets | Does adapting the sampled image location help? |
| Add a second feature scale | Does more spatial detail improve small objects? |
| Add depth-aware weighting | Does explicit depth consistency reduce ray ambiguity? |
| Add learned upsampling | Can the model recover sharper occupancy boundaries? |

Depth supervision derived from training LiDAR is compatible with camera-only inference, provided inference uses predicted depth and does not require LiDAR.

## 12. Technical risks and evaluation requirements

1. **Incorrect geometry.** Wrong transform direction, frame, timestamp handling, or axis order can produce plausible-looking outputs while preventing meaningful learning.
2. **Incorrect supervision and evaluation.** Unknown/free confusion, wrong class mapping, or inconsistent masks can invalidate losses and scores.
3. **Attention and resolution chosen without profiling.** Global attention or broad fine-resolution 3D processing can consume the memory budget immediately.
4. **Mistaking projected features for resolved 3D evidence.** Sampling at a voxel projection does not prove occupancy. Expect ray ambiguity and occlusion errors.
5. **Trusting visualizations or voxel accuracy alone.** An all-free model can appear numerically successful under inappropriate metrics; a model can also memorize nearby training frames.

Report semantic mIoU with the selected evaluator's class policy, per-class IoU, occupied/free IoU, and the exact mask and spatial range. Aggregate confusion counts over the dataset instead of averaging arbitrary batch IoUs.

For a strong portfolio result, prioritize **verified geometry, reproducible training, held-out evaluation, and a measured mean-fusion-versus-attention ablation**. Document each design decision, its measured effect, and the remaining limitations so the project is reproducible and easy to explain.

## References

[1] Wei et al. SurroundOcc: Multi-camera 3D Occupancy Prediction for Autonomous Driving. ICCV 2023. https://openaccess.thecvf.com/content/ICCV2023/html/Wei_SurroundOcc_Multi-camera_3D_Occupancy_Prediction_for_Autonomous_Driving_ICCV_2023_paper.html

[2] Occ3D official repository and annotation documentation. https://github.com/Tsinghua-MARS-Lab/Occ3D

[3] Li et al. FB-BEV: BEV Representation from Forward-Backward View Transformations. ICCV 2023. https://openaccess.thecvf.com/content/ICCV2023/papers/Li_FB-BEV_BEV_Representation_from_Forward-Backward_View_Transformations_ICCV_2023_paper.pdf

[4] nuScenes official tutorial. https://www.nuscenes.org/tutorials/nuscenes_tutorial.html

[5] nuScenes official devkit: sensor-to-image transform implementation. https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py

[6] PyTorch grid_sample documentation. https://docs.pytorch.org/docs/main/generated/torch.nn.functional.grid_sample.html

