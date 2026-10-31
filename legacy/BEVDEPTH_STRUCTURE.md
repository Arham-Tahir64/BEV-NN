# BEVDepth structure adapted to camera-only occupancy

Implemented in the Desktop BEV-NN project on 2026-10-25. Enable with
`--architecture bevdepth --depth`. Existing checkpoints retain their architecture.

## Camera-only contract

The model receives six RGB images, intrinsics and ego-to-camera calibration.
It never receives LiDAR points, ground-truth occupancy or depth targets as input.
Occupancy/depth annotations are training supervision only. The recommended
100-scene command below uses the existing occupancy annotations for approximate
depth supervision and requires no raw LiDAR, including during training.

An optional `--depth-target-mode lidar` implements measured training targets
for comparison with the paper. It is not the recommended command below and is
not a sensor-fusion branch. It was smoke-tested with the available mini files.
The larger Occ3D package lacks raw LiDAR and its metadata; this option fails
explicitly there rather than substituting pseudo-depth. Camera-only inference
works with either kind of training supervision.

## Why the previous depth head was insufficient

The previous frozen image backbone, small depth classifier and inverse voxel
sampling were not a reproduction of BEVDepth. Correcting probability scaling
removed a lifting issue, but did not provide the paper's camera conditioning,
trainable multiscale features or frustum refinement. A broad depth distribution
still sends appearance along multiple depths. Our annotation-derived depth
targets are also imperfect: the nearest annotated voxel is not necessarily the
nearest physical surface.

The screenshot establishes remaining artifacts; it cannot identify their
individual causes. This implementation is a new architecture experiment, not
evidence that the artifacts are already fixed.

## Implemented flow

1. Trainable ResNet18 residual stages produce four image scales; a frozen stem
   and fixed ImageNet BatchNorm statistics stabilize the small camera batch.
   Aligned multiscale features are fused into 128 channels at stride 16.
2. Intrinsics, camera-to-ego pose and augmentation descriptors condition
   separate depth/context branches through MLPs and channel gates.
3. The depth branch uses three residual blocks, ASPP, a deformable convolution
   and a 64-bin classifier. Softmax produces a depth distribution per pixel.
   The context branch produces 32 appearance channels.
4. Lift using `context * depth_probability`. No unrestricted appearance
   bypass, probability clipping, bin-count multiplier or support normalization.
5. Refine the lifted frustum with residual convolutions over depth and image
   width, independently for each image row. Checkpointed chunks reduce memory.
6. Unproject each frustum sample through calibration and sum it into its
   destination voxel. Out-of-bounds samples are discarded. Height is retained.
7. A multiscale 3D residual decoder predicts 18 voxel classes. XY is downsampled
   twice and restored with skips; all 16 Z layers remain throughout.

Depth supervision defaults to one-hot BCE on softmax probabilities, summed
over bins and averaged over valid pixels, weighted by 3. Unknown depths are
masked. Occupancy supervision and the selected scene split remain in place.

## Deliberate differences from the reference

This is an occupancy adaptation, not a full BEVDepth benchmark reproduction.
The reference is a 3D detector; it ordinarily pools into BEV and uses a
detection head. We retain Z and use a 3D occupancy decoder.

The implementation uses ResNet18 instead of the reference ResNet50, 128 depth
channels instead of 512, and 32 context channels instead of 80. The image neck
uses aligned lateral projections instead of SECONDFPN. New convolutional
blocks use GroupNorm; camera conditioning uses normalized calibration and
LayerNorm rather than the reference BatchNorm descriptor. These choices fit
the local GPU and batch size. The default bins remain 64 one-metre intervals
over [0.1,64.1), rather than the reference configuration's half-metre intervals.

Feature rays are centred at pixel `16*i`, so nearest-cell target assignment
matches that convention rather than the reference's block-floor min pooling.
Image resize is included in K; other augmentation descriptors are identities.
This version is single-frame, with no new temporal fusion or augmentation
pipeline. Forward pooling uses native PyTorch index-add, not the official
optimized pooling kernel. Projected occupancy targets in the recommended
experiment differ materially from the paper's measured LiDAR targets.

## Fresh five-epoch pilot

From `C:\Users\Arham\Desktop\BEV-NN`:

```powershell
.\.venv\Scripts\python.exe train.py --architecture bevdepth --depth --depth-target-mode projected --depth-loss-mode bce --depth-loss-weight 3 --dataset-version v1.0-trainval --dataroot data/occ3d_trainval --annotations-json data/occ3d_trainval/downloads/annotations.json --occupancy-root data/occ3d_trainval/gts --train-scenes outputs/splits/train100_val20/train_scenes.json --val-scenes outputs/splits/train100_val20/val_scenes.json --epochs 5 --overfit-samples 0 --class-balance sqrt-inverse --xyz --geometry-loss-weight 0 --output-dir outputs/train_bevdepth_camera_train100
```

Start fresh from ImageNet, not the mini epoch-20 weights. Two former mini
training scenes belong to the full validation split. The new architecture
also has incompatible task weights. Do not reuse an existing run directory.

Compare validation occupied mIoU, class IoUs and occupied precision/recall
against the probability-lifting run on the same 100/20 split. Inspect the same
frames and upper-height occupancy. Depth loss is now BCE, so its numeric value
is not directly comparable to the previous linear CE. Several components
change together: this tests the structure, not which component caused a gain.
Five epochs are an initial check, not a guaranteed convergence budget.

## Validation and cost

- 108 unit tests pass, including geometric unprojection, voxel ordering,
  height preservation, camera conditioning, refinement gradients, losses,
  optional measured targets and checkpoint reconstruction.
- All five prior model modes produce bit-for-bit identical reference outputs.
  Preserved epoch-20 and probability epoch-4 checkpoint hashes are unchanged.
- RTX 4060, batch 1, six 320x576 images, 200x200x16 voxels: median inference
  87.81 ms over five warmed timings; peak training allocation 2012 MiB,
  reserved 2218 MiB; 14,719,632 trainable parameters. Three AMP optimizer steps
  completed with finite gradients reaching depth and image stages.
- Earlier probability architecture measured 47.68 ms and 457 MiB allocated
  in its smoke test: roughly 1.8x inference time and 4.4x training allocation.
  These short runs are not an epoch-time estimate and exclude other apps'
  GPU allocations. Data loading and validation add time.
- Measured-target mini training, resume, and full-resolution viewer inference
  also passed. These smoke checkpoints are not quality candidates. The live
  viewer still shows the trained probability epoch-4 checkpoint.

Report: `outputs/bevdepth_structure_smoke/validation.json`.
Test log: `outputs/bevdepth_tests.log`.
Original source backup: `outputs/bevdepth_structure_backup_20261025_222733`.
No full pilot has been started by this implementation task.

## Sources

- [BEVDepth paper, section 4](https://arxiv.org/html/2206.10092v2)
- [Official depth network, refinement and lifting](https://github.com/Megvii-BaseDetection/BEVDepth/blob/main/bevdepth/layers/backbones/base_lss_fpn.py)
- [Official experiment configuration and depth objective](https://github.com/Megvii-BaseDetection/BEVDepth/blob/main/bevdepth/exps/nuscenes/base_exp.py)

## Boundary correction

The first pilot stopped in epoch 1 near batch 1740 with a CUDA index-add assertion. Float32 grid-coordinate rounding could send an interior upper-edge point to index 200. Metric bounds still control inclusion; calculated indices are now constrained per axis before flattening. A CPU/CUDA regression verifies edge assignment and gradients. All 109 tests pass. Failed-run logs are preserved; restarted output: outputs/train_bevdepth_camera_train100_boundaryfix.

