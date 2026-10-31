# Free-space boundaries and larger camera images

This experiment keeps the camera-only BEVDepth architecture and 0.4 m voxel
grid. It uses 480x864 RGB images and adds geometry supervision to the existing
semantic logits. There is no new prediction head or inference sensor input.

## Objective and masking

The proposed initial objective is:

    semantic CE + 3.0 * depth BCE + 0.2 * occupied/free BCE + 0.1 * boundary BCE

Semantic CE retains the existing inverse-square-root class weights. Global
occupied/free BCE weights all camera-visible, non-ignore voxels equally. Both
geometry terms use logsumexp of classes 0..16 minus the free-class (17) logit
as binary occupied log-odds. Their means are independent of semantic weights.

Boundary BCE is averaged over both endpoints of a six-neighbour occupied/free
transition in X, Y or Z. BOTH endpoints must pass the camera visibility mask
and have known labels. Unknown voxels (255), invisible voxels, and the exterior
of the voxel grid create no boundaries. Neighbouring different occupied
classes do not count as occupied/free boundaries. There is no dilation or
instance-identity supervision. With no valid boundary, its contribution is a
differentiable zero. Geometry 0.2 and boundary 0.1 are starting hyperparameters,
not established optimal weights.

## Resolution and checkpoints

`--image-size 480 864` yields 30x54 features at stride 16 rather than 20x36.
Camera intrinsics follow the existing pixel-centre resize convention; depth
targets and cache keys use the resized geometry. Input/frustum spatial area
increases 2.25x, while the output remains 200x200x16. Whole-model runtime and
memory do not scale by a fixed 2.25x factor.

Image size is stored in checkpoint data metadata. Resume and the viewer use
the saved size. Boundary weight is saved in training/loss configuration and
restored by resume. Legacy defaults remain 320x576, zero boundary weight and
zero geometry weight. Resuming an old checkpoint preserves its old settings;
the command below creates the intended new experiment.

## Launch the five-epoch combined experiment

From `C:\Users\Arham\Desktop\BEV-NN`:

```powershell
.\legacy\train_boundary_hires.ps1
```

Equivalent command:

```powershell
.\.venv\Scripts\python.exe train.py --architecture bevdepth --depth --xyz --image-size 480 864 --geometry-loss-weight 0.2 --boundary-loss-weight 0.1 --depth-target-mode projected --depth-loss-mode bce --depth-loss-weight 3 --dataset-version v1.0-trainval --dataroot data/occ3d_trainval --annotations-json data/occ3d_trainval/downloads/annotations.json --occupancy-root data/occ3d_trainval/gts --train-scenes outputs/splits/train100_val20/train_scenes.json --val-scenes outputs/splits/train100_val20/val_scenes.json --epochs 5 --overfit-samples 0 --class-balance sqrt-inverse --batch-size 1 --workers 2 --seed 0 --output-dir outputs/train_bevdepth_boundary_hires_train100
```

This initializes from ImageNet and retains the existing 100/20 scene split.
It does not warm-start from mini weights or overwrite the current best model.
The launcher refuses an existing output directory. After an interruption,
resume that run with:

```powershell
.\.venv\Scripts\python.exe resume_train.py --checkpoint outputs/train_bevdepth_boundary_hires_train100/latest.pt --epochs 5 --workers 2
```

No full experiment has been started by the implementation task.

## Evaluate the trade-off

Validation now logs occupied precision/recall/mIoU and exact binary boundary
precision, recall and F1. Boundary counts are aggregated across the whole
validation set before computing rates. Both predicted and labelled boundaries
are restricted to valid observed neighbour pairs. This is exact voxel overlap
at 0.4 m resolution, with zero matching tolerance; undefined rates are null.
It measures surface localization, not separate instances or semantic identity.

Look for higher precision and boundary F1 without losing substantial occupied
recall or class mIoU. Keep monitoring far-depth MAE and the same viewer frames.
Do not select a model on free-space accuracy alone. Existing best-semantic
checkpoint selection still uses occupied mIoU; boundary metrics are diagnostics.
Historical checkpoints have no saved boundary scores, so re-evaluate them with
the updated validation function for direct boundary comparisons. The baseline
best epoch-4 scores were mIoU 14.51%, precision 51.65%, recall 82.68%.

This combined run changes both resolution and objective. To attribute a gain,
run separate output directories for loss-only (320x576) and resolution-only
(both geometry coefficients zero), with the same split and seed.

## Verification

- 121 unit tests passed, including unknown/invisible masks, all neighbour
  directions, absence of exterior/batch wrapping, gradient direction, gap
  filling metrics, CLI validation and training/validation integration.
- RTX 4060 smoke with six real mini frames, batch size 1 and two workers:
  2419.5 MiB peak allocated and 2730 MiB reserved GPU memory. These are
  process allocator figures, not total device use or maximum over an epoch.
- Verified 480x864 inputs, 30x54 depth targets/logits, resized calibration,
  finite outputs, and unchanged 200x200x16 occupancy dimensions.
- Fresh AMP startup backed off once from scale 65536 to 32768, so five of six
  initial optimizer steps ran. This is recorded rather than counted as six
  updates. The smoke checkpoint is for engineering checks, not model quality.

Test log: `outputs/boundary_hires_tests.log`.
GPU report: `outputs/boundary_hires_smoke_verified/validation.json`.
Source backup: `outputs/boundary_hires_backup_20261026_110353`.

Implementation: `src/losses.py`, `src/metrics.py`, `train.py`, `resume_train.py`.

Resume verification: all six additional optimizer steps completed; image size, both loss weights and workers=2 survived checkpoint restoration. Viewer checkpoint loading and full-resolution frame generation passed.

