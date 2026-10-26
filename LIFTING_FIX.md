# Probability-weighted lifting

New `train.py --depth` runs default to `--depth-fusion probability`.
This corrects three mechanisms in the earlier residual lifting:

1. Every camera appearance feature is depth-weighted. There is no raw image
   appearance branch alongside the weighted features.
2. Weighting is `D * p(depth)` with no ceiling, where D is the fixed bin count
   (64 here). Multiplying by one global constant preserves probability ratios
   and keeps a uniform distribution neutral in feature scale. For example,
   90% versus 6% produces 57.6 versus 3.84: a 15-to-1 feature ratio. The old
   `min(64*p, 4)` produced 4 versus 3.84.
3. Image context is multiplied by probability at each image feature pixel
   BEFORE interpolation into voxels. This preserves each pixel's appearance
   and depth pairing. Separately interpolating context and depth can produce
   false foreground features at a neighbouring background pixel's depth.

The implementation creates a small `[B*6,C+1,D,h,w]` image frustum once,
then samples voxel chunks with trilinear interpolation. It never allocates
`[B,6,C,D,V]`. One channel carries depth support; the other C carry weighted
appearance. Valid views are averaged by their geometric camera count. There
is no division by depth support that could cancel attenuation in one view.

Depth-bin ends are extended only within their valid range. Out-of-range depth,
out-of-image projections, and invalid camera views contribute zero. Sampling
and depth weighting use FP32 with CUDA AMP; the fused voxel features return
to the image context's dtype before decoding.

## Compatibility and scope

The model constructor still defaults to `legacy` for old configurations.
Saved checkpoints retain their exact `legacy`, `residual`, or `gated` mode;
resume and the viewer reconstruct that mode from checkpoint configuration.
The current epoch-5 viewer model therefore remains the trained reference.
Do not change its configuration string to relabel it as the corrected model.

The `probability` path uses the existing C+1-to-C projection (1,088 parameters
at C=32), with identity context initialization and initially zero support
weight/bias. It introduces no additional prediction head. Its trainable
parameter count is 123,026 versus 124,050 for residual fusion.

The decoder, frozen image backbone, image resolution, depth bins/targets,
losses, and XYZ features are unchanged. XYZ, learned biases, and decoder
context can still produce occupancy priors. Mandatory depth weighting of
camera appearance is not a hard constraint that occupied predictions must
sit at a depth peak. Broad or wrong learned depth can still cause smearing.

This is a lifting correction, not a reproduction of the full BEVDepth detector.

## Fresh five-epoch comparison on 100 scenes

Run from `C:/Users/Arham/Desktop/BEV-NN`:

```powershell
.\.venv\Scripts\python.exe train.py --dataset-version v1.0-trainval --dataroot data/occ3d_trainval --annotations-json data/occ3d_trainval/downloads/annotations.json --occupancy-root data/occ3d_trainval/gts --train-scenes outputs/splits/train100_val20/train_scenes.json --val-scenes outputs/splits/train100_val20/val_scenes.json --epochs 5 --overfit-samples 0 --class-balance sqrt-inverse --xyz --depth --depth-fusion probability --geometry-loss-weight 0 --output-dir outputs/train_xyz_depth_probability_train100
```

This uses the same 4,021 training and 803 validation frames, seed, optimizer,
class weighting, and five-epoch budget as the larger residual run. Start fresh:
`resume_train.py` deliberately restores the OLD lifting for OLD checkpoints.
The new directory preserves both the epoch-20 mini baseline and the trained
100-scene epoch-5 model. Class weights are estimated from training data only;
the existing content-keyed depth-target cache can be reused.

This combines three related lifting changes. It compares the resulting fix
against the previous path, but does not isolate each change's contribution.
Evaluate occupied semantic mIoU, car/manmade IoU, occupied precision/recall,
above-2 m precision/recall, and identical scene/frame views. Cleaner pictures
alone do not justify promotion if real structures disappear.

Implementation and smoke verification are complete. The five-epoch quality
comparison above has not been started by this change.

## Verification

- All 93 project tests pass, including 10 new lifting tests.
- Deterministic full-model outputs for depth-off, legacy, residual, and gated
  paths match captured pre-change outputs bit for bit.
- The old interpolation gives false foreground feature 5 in the synthetic
  foreground/background test; joint lifting gives exactly 0 at that depth.
- A 90%/6% depth preference retains a 15-to-1 feature ratio; the high-probability
  path also retains a nonzero gradient above the old clipping threshold.
- Uniform feature scale, depth-edge masking, invalid views, chunked outputs
  and gradients, occupancy-to-depth gradients, checkpoint round trips, and
  CLI mode selection are covered.
- Actual 6-camera 320x576 inputs and a 200x200x16 grid completed three CUDA AMP
  optimizer updates per path. All three updates executed; depth gradients
  were finite and nonzero. Backbone parameters remain frozen.

Short RTX 4060 smoke measurements (not a full benchmark):

| Measurement | Residual reference | Probability lifting |
|---|---:|---:|
| Median inference, five measured forwards | 45.79 ms | 47.68 ms |
| Peak PyTorch allocated memory during three training batches | 771.51 MiB | 456.77 MiB |

Timing excludes image loading and rendering. Memory reports this process's
PyTorch allocations, excluding other apps and CUDA driver overhead. Three
training batches include startup effects and cannot establish epoch throughput.
No claim about trained accuracy follows from this smoke check.

Full smoke evidence: `outputs/lifting_probability_smoke/validation.json`.
Original source backup: `outputs/lifting_fix_backup_20261025_180214/`.
