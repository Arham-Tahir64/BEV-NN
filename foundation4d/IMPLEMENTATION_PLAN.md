# Foundation4D-inspired local model: implementation plan

Implementation status: M0–M2's functional pipeline is complete, including geometry
tests, real-data single-frame fitting, measured training profiles, CLI train/resume,
evaluation and viewer loading. Full eight-epoch validation remains to be run. M3's
learned-offset and auxiliary-depth switches are implemented and smoke-tested, but
their full ablations are not trained. M4 temporal history remains optional future work.
See RUNNING.md for the implemented commands and VERIFICATION.md for measurements.

## Objective and scope

Improve occupancy IoU, semantic mIoU, boundaries and distant localization per unit
of training compute on a single RTX 4060. This is a new model family inside this
folder. It should replace dense hidden 3D volumes and ray-distributed appearance
with calibration-grounded BEV queries. It is not a recreation of the full NVIDIA
system or its reported performance. Known camera calibration is used directly.

Depth and semantic occupancy are the initial tasks. Detection, learned poses,
Gaussian reconstruction, offline tracking and global pose optimization are outside
the initial training budget. Add a task only after a measured occupancy benefit.
No teacher distillation should depend on unavailable NVIDIA weights.

The design borrows the compact BEV/height-at-output principle from Foundation4D.
All concrete widths, resolutions, pooling rules and schedules below are our own
engineering choices, not copied settings or verified performance claims.

## Planned forward path

```text
Six calibrated RGB images
  -> shared ImageNet-pretrained ResNet18 multi-scale features
  -> 64-channel stride-16 feature maps
  -> project BEV pillar references into valid cameras
  -> chunked local sampling + learned camera/height aggregation
  -> two lightweight BEV residual blocks
  -> 2D upsampling / refinement to 200x200
  -> separate binary and semantic channels
  -> reshape channels into 16 height layers
  -> binary decision + occupied-class assignment
```

1. **Image encoder.** Begin with 320x576, six views, batch 1; width 64. Reuse the
   validated feature-centre convention (pixel stride*i) and resize-intrinsics rule.
   Keep BatchNorm running statistics fixed for the correlated/small batch. Use
   trainable residual stages after an initial smoke baseline; no ViT-L dependency.
   Copy/adapt a minimal encoder into the new package when implemented, preserving
   provenance; do not import an evolving old model implicitly at runtime.
2. **Calibrated BEV projector.** Use 100x100 BEV cells over the existing 80x80 m
   extent, i.e. 0.8 m hidden XY spacing. Place four uniformly spaced pillar-height
   references in [-1,5.4] m. Their cell-centre heights are -0.2,1.4,3.0,4.6 m.
   Project using the existing exact sample-time-to-camera transform. Start with
   fixed samples, then a bounded learned 2D offset variant (four samples per
   height/camera). Mask behind-camera, out-of-image and invalid samples BEFORE
   normalization. All-invalid queries return zero image evidence, not NaNs or an
   arbitrary camera average. Preserve separate height encodings until merging;
   averaging unencoded heights would discard essential vertical structure.
3. **Aggregation.** Process cameras and query chunks incrementally. Normalize
   learned camera/sample weights over valid support. Start with weighted sums plus
   small MLPs/convolutions, not global all-to-all BEV self-attention. Do not create
   a full CxDxHxW image frustum. This changes the ambiguity pattern, but projection
   alone does not solve monocular depth or occlusion. Inspect false occupancy and
   add depth-consistency evidence only in a separate ablation.
4. **Decoder.** Two BEV residual blocks, bilinear 2x upsampling plus 2D convolutions.
   Avoid transposed-convolution checkerboard effects in the first prototype. Output
   16 binary channels and 16*17 semantic channels, reshape using explicit XYZ tests.
   Class 0 is occupied, class 17 free, unknown 255 excluded. Preserve all target
   height layers; output-grid resolution remains 0.4 m for comparable evaluation.
   Height-from-channels may lose detail or entangle surfaces: inspect height slices
   and compare a cheap learned height embedding before adding expensive 3D refinement.
5. **Depth auxiliary head, staged.** Train supervised depth on the image features
   only after the projector baseline works. Compare off/on, using the existing
   pseudo-targets and their known limitations. Do not initially multiply context
   along every depth bin again. Optional range balancing is a distinct switch;
   do not bundle the prior two experimental changes into the baseline by default.

## Supervision and inference contracts

Binary BCE is evaluated on known voxels (occupied 0..16 versus free 17). Semantic
CE is evaluated on known OCCUPIED voxels only. An all-free batch must return a
connected zero semantic loss while still training binary occupancy. Unknown space
is not treated as free. Start both loss weights at 1 as a tuning baseline, with
the same existing class weights for occupied CE restricted to classes 0..16.
Retain unweighted evaluation metrics regardless of training weights.

At inference, sigmoid(binary) above a calibrated threshold selects occupied;
semantic argmax over 17 classes assigns its class, otherwise assign free=17.
Choose the threshold on a separate calibration subset, then freeze it for reported
validation. Also report a fixed 0.5 baseline. Do not tune on the particular frames
used to claim a visual gain. Report precision and recall so threshold effects are clear.
Separate binary/semantic heads can improve decision control but do not guarantee
higher IoU. Dice, Lovasz, boundary BCE or BEV supervision are independent later
experiments rather than all-on defaults. The earlier BEV-target coverage was sparse.

## Milestones and acceptance gates

### M0 — data/geometry contracts

- Adapter reuses the existing train100/val20 scene list, images and labels by path.
  It preserves camera order, calibrated poses, label mapping, masks and split isolation.
- Unit checks: projection round trips, resize/feature centre alignment, camera
  permutation consistency, invalid projections, missing-camera masks, exact XYZ and
  channel-to-height layout. Synthetic known-depth points land in intended queries.
- Use existing annotations/calibration. Optional point-cloud supervision is a training
  data choice; model inference inputs remain RGB plus known calibration.

### M1 — minimal forward/backward and resource gate

- Implement the image encoder, fixed four-height sampler, 2D BEV neck and two heads.
- Overfit one training frame before a full split. Test disconnected surfaces and
  high/low height layers, not just ground-plane appearance.
- Measure at least 20 steady-state optimizer steps after warmup; initialize Adam
  states before recording peaks. Report parameters, peak allocated AND reserved
  memory, preprocessing, forward/backward/step times separately and AMP overflows.
- Profile with other GPU workloads stopped. Target <=6 GiB allocated / <=7 GiB
  reserved. Record actual device and image size. A tiny toy pass is not sufficient.
- If the target fails: query chunk 512, 50x50 queries for feasibility, feature width
  32, fewer offsets; then retest. Always retain the output grid/mask when comparing
  accuracy. Query coarsening risks worse boundaries and must be labeled accordingly.

### M2 — single-frame controlled baseline

- Train 8 epochs at 320x576, seed 0, AdamW lr 1e-4, batch 1, mixed precision. Start
  with a constant LR for comparability; any cosine-decay experiment is separate.
- Compare against an old-model 320x576 control and existing best checkpoints on
  the exact evaluation protocol. The cheaper resolution is not a fair stand-in for
  a 480x864 architecture comparison; record accuracy and compute jointly.
- Then profile 480x864, and only train it if measured memory fits. This is the
  resolution-matched comparison against the old boundary run.
- Retain latest, best binary IoU, best semantic mIoU, best boundary F1 and EVERY
  epoch checkpoint plus metadata. Preflight disk budget; optimizer checkpoints can
  dominate storage. No automatic deletion of user checkpoints.
- Log near/middle/far, front/rear, per-class IoU, exact boundary F1, precision/recall,
  known-free false positives and height slices. The output must support existing
  viewers through a model registry key; do not silently change old architecture loading.

### M3 — resolve ambiguity with minimal added training cost

- Compare fixed vs bounded learned sample offsets, depth auxiliary off/on and
  optional range weighting one at a time. Use the same seeds and epoch budgets.
- Add depth-to-reference compatibility as a soft sampled feature, rather than
  unconditional ray multiplication. Never use labels as inference inputs. Avoid
  hard empty-space carving behind a surface or in unknown regions.
- Run matched-frame ray/false-positive analysis; require stable aggregate occupancy
  and boundary gains before calling a qualitative change an improvement.

### M4 — streaming temporal evidence, optional

- Cache one detached past 100x100x64 BEV, warp it with provided relative ego poses,
  fuse using a confidence/age gate, and retain the current path. About 1.22 MiB per
  FP16 feature state alone, plus masks/workspaces. No backprop through old encoders
  initially. This is NOT equivalent to the paper's full spatiotemporal transformer.
- Train with realistic history and scene-boundary resets; randomly drop history
  so the current-frame path remains valid. Use past frames only for causal inference.
- Planar warping is an approximation: test turns, pitch/roll, slopes, disocclusion,
  invalid overlap and moving objects. Suppress inconsistent dynamic history. A
  persistent wrong prediction must not become a new ghost trail. An exact SE(3)
  geometry-aware state may be needed later if planar errors dominate.
- Using supplied ego poses is an explicit assumption, not a learned pose capability.

## What would count as success

The current comparison references are old boundary epoch 7 (50.84% binary IoU,
16.40% semantic mIoU, 22.43% boundary F1) and new supported/range epoch 2 (52.40%,
15.04%,21.38%). A useful new model should improve the accuracy/compute tradeoff,
not merely trade more false occupancy for recall or reduce coverage/mask difficulty.
Use the full unchanged validation set for those references and report the resolution.
For a matched-resolution run, beating roughly 52.4% occupancy IoU while retaining
at least ~16.4% semantic mIoU and ~22.4% boundary F1 would be a strong first goal,
not a guarantee. More distant accuracy at substantially lower compute is also useful.

Foundation4D's OpenOccupancy numbers cannot be compared directly with this project's
Occ3D visibility-masked, 0.4 m, 17-occupied-class scores. Keep dataset, taxonomy,
spatial extent and mask protocols explicit. Repeat promising comparisons over
additional seeds before attributing a small gain to the architecture.
