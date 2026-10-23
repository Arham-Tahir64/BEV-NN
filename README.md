# nuScenes mini loader

## Development blog

Follow the architecture decisions and experiments at
[BEV-NN — field notes](https://arham-tahir64.github.io/BEV-NN/).
The Astro site lives in [`blog/`](blog/README.md); new Markdown entries publish
through GitHub Actions when pushed to `main`.

Official mini dataset: https://www.nuscenes.org/data/v1.0-mini.tgz
Official devkit: https://github.com/nutonomy/nuscenes-devkit

The dataset lives in `data/nuscenes/`, containing `v1.0-mini/`, `samples/`,
`sweeps/`, and `maps/`. The compressed archive is also retained there.

## Run in PowerShell

```powershell
cd C:\Users\Arham\Desktop\BEV-NN
.\.venv\Scripts\python.exe load_nuscenes.py
```

By default, the script streams all six camera views with coloured ground-truth
voxel centres. The installed Occ3D annotations cover all 404 mini keyframes.
Playback uses the exact six camera tokens for each annotated keyframe (about
2 Hz) and follows the recorded timestamps. The GT voxel checkbox toggles the dots.

Use `--no-project-voxels` for the original full-rate camera playback, including
intermediate frames from `sweeps/`, aligned to the nearest front-camera timestamp.
Scenes play in order of start time; gaps between separate scenes are omitted.
Images are decoded and resized as playback advances, so video is not preloaded.
Playback can run slower on machines that cannot decode six images fast enough.

Use the buttons or Space to pause/play. Left/right arrows step one frame.
Choose a speed in the window, or pass `--speed`. Playback stops at the end;
`--loop` repeats it. Close the window to exit.

```powershell
.\.venv\Scripts\python.exe load_nuscenes.py --scene scene-0061
.\.venv\Scripts\python.exe load_nuscenes.py --speed 2 --loop
.\.venv\Scripts\python.exe load_nuscenes.py --single-frame --sample-index 10
.\.venv\Scripts\python.exe load_nuscenes.py --dataroot D:\datasets\nuscenes --version v1.0-trainval
```

`--single-frame`, `--no-show`, and `--save-camera-grid` use the sample-based
loader. `--sample-index` and `--camera` apply to that mode. Streaming uses
`--scene` to select a scene. GT playback starts at the first labelled keyframe;
full-rate playback starts at the first recorded camera frame.

Save the camera grid without opening a window:

```powershell
.\.venv\Scripts\python.exe load_nuscenes.py --no-show --save-camera-grid camera_frames.png
```

## Use in Python

```python
from data.nuscenes_dataset import load_nuscenes, load_sample
from load_nuscenes import show_camera_frames

nusc = load_nuscenes()
item = load_sample(nusc, sample_index=0)
rgb = item["image"]             # H x W x 3 uint8 RGB array
annotations = item["annotations"]
all_camera_images = item["images"]  # Camera name -> RGB array
show_camera_frames(item)
```

Annotation positions are in the nuScenes global frame. Apply devkit calibration
and ego poses before projecting annotations onto camera images.

## Data package and occupancy labels

```text
data/
  __init__.py
  nuscenes_dataset.py   # nuScenes metadata, camera images, annotations
  occupancy_labels.py   # voxel semantics and camera visibility mask
  voxel_projection.py   # ego voxel centres -> calibrated camera projections
  nuscenes/             # downloaded nuScenes dataset (ignored by Git)
  occupancy/            # external occupancy annotations (ignored by Git)
    gts/
      scene-0061/
        <sample_token>/
          labels.npz
```

`load_nuscenes.py` is the playback/sample command-line entry point. Its loading
functions now live in `data/nuscenes_dataset.py`; existing script imports remain
available for compatibility.

The occupancy loader supports the [Occ3D-nuScenes archive format](https://github.com/Tsinghua-MARS-Lab/Occ3D).
Each `labels.npz` must contain `semantics` and `mask_camera` with
the same nonempty 3D shape. Labels keep their original integer dtype and class
IDs, including ignore labels. The camera visibility mask is returned as a boolean array;
`True` means observed. Only the two required arrays are loaded. Arrays are not
resized, transposed, or remapped. Match grid bounds, resolution, axis order, and
class mapping to the source annotations when configuring the model.

The standard nuScenes mini download does not include dense voxel occupancy labels.
Occ3D labels have now been downloaded separately and installed for all 404 mini
samples under `data/occupancy/gts`. The full source archive is retained at
`data/occupancy/gts.tar.gz`; only mini labels were extracted. See
`data/occupancy/download_manifest.json` for source URLs and the verified SHA-256.
For other datasets, pass the location of a compatible `gts` directory.
Occupancy labels are associated with sample tokens (keyframes), not camera sweeps.

```python
from data import load_nuscenes, load_sample, load_occupancy_labels, load_sample_occupancy

nusc = load_nuscenes()
occupancy = load_sample_occupancy(nusc, nusc.sample[0]["token"])
labels = occupancy["semantics"]
camera_visible = occupancy["mask_camera"]

# Alternatively, attach occupancy to the loaded camera sample:
item = load_sample(nusc, sample_index=0, occupancy_root="data/occupancy/gts")
labels = item["occupancy"]["semantics"]

# Read any individual compatible archive directly:
occupancy = load_occupancy_labels("path/to/labels.npz")
```

Inspect one sample with its installed occupancy labels:

```powershell
.\.venv\Scripts\python.exe load_nuscenes.py --no-show --occupancy-root data/occupancy/gts
```

## Ground-truth voxel projection

```powershell
# Play all six camera views with ground-truth semantic voxel dots (default).
.\.venv\Scripts\python.exe load_nuscenes.py

# Inspect one annotated frame, or save the six-camera sanity-check image.
.\.venv\Scripts\python.exe load_nuscenes.py --single-frame --sample-index 0
.\.venv\Scripts\python.exe load_nuscenes.py --no-show --save-camera-grid outputs/voxel_projection.png

# Original camera-only playback with sweeps, without voxel overlays.
.\.venv\Scripts\python.exe load_nuscenes.py --no-project-voxels
```

Voxel centres use Occ3D-nuScenes's `[X,Y,Z]` grid of `200 x 200 x 16`, with
0.4 m voxels and bounds `[-40,-40,-1]` to `[40,40,5.4]` metres. The centre is
`minimum + (index + 0.5) * voxel_size`. The coordinate frame is the ego vehicle
at the sample timestamp. For each camera, centres are transformed through the
sample ego pose, global frame, camera acquisition ego pose, and camera extrinsics,
then projected using the native camera intrinsic matrix. Resizing is accounted for.
Only calibration/pose metadata is read; no raw point clouds are loaded.

Only occupied, camera-observed surface voxels are drawn. Free (17), ignore (255),
and interior voxels are excluded. Other class IDs are preserved. The colour
legend groups vehicle/cycle classes and barrier/cone classes for readability;
`Manmade` is the dataset class, which includes more than buildings.
A nearest-centre image-cell z-buffer limits overdraw and provides approximate
occlusion filtering. This is a calibration sanity check, not exact mesh rendering:
voxel resolution, label imperfections, object motion during camera acquisition,
and centre-based visibility filtering can leave residual offsets or edge bleed.

GT overlays use labelled keyframes only. They never reuse a previous/next voxel
label for a sweep. Missing labels are reported and skipped during GT playback;
selecting a missing label in sample mode fails with its expected file path.

```python
from data import load_nuscenes, load_sample, project_sample_voxels
from load_nuscenes import show_camera_frames

nusc = load_nuscenes()
item = load_sample(nusc, occupancy_root="data/occupancy/gts")
item["voxel_projections"] = project_sample_voxels(
    nusc, item["sample"]["token"], item["occupancy"]
)
show_camera_frames(item)
```

Run validation tests with `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`.

## Recreate the environment

Python 3.12 is used for this project.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## PyTorch training batches

Install CUDA-enabled PyTorch for the RTX 4060 before installing the remaining
requirements (CPU-only environments can use the standard requirements command):

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_dataset.py --workers 2
```

The check saves outputs/training_projection.png and a JSON validation report.
It verifies official disjoint mini scene splits, a real batch, resized camera
projection consistency, and CUDA transfer when available.

```python
from data import NuScenesOccupancyDataset, make_dataloader

def main():
    train = NuScenesOccupancyDataset(split="train", image_size=(320, 576))
    loader = make_dataloader(train, batch_size=2, num_workers=2)
    batch = next(iter(loader))
    targets = batch["semantics"].clone()
    targets[~batch["valid_mask"]] = 255
    # For model logits [B,18,200,200,16], use cross entropy with
    # ignore_index=255. Skip batches with no valid targets.

if __name__ == "__main__":
    main()
```

Each item supplies normalized float32 RGB images [6,3,H,W], resized intrinsics
[6,3,3], sample-ego-to-camera transforms [6,4,4], int64 semantics [200,200,16],
boolean mask_camera and valid_mask grids, plus sample_token and scene_name.
The batch dimension is prepended by the loader. Images use ImageNet mean
(0.485,0.456,0.406) and standard deviation (0.229,0.224,0.225).
Camera order is FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_LEFT, BACK_RIGHT.
All camera names have the CAM_ prefix.

The official mini_train and mini_val scene splits are selected by train/val.
Missing images or labels fail explicitly. Occupancy remains in [X,Y,Z] order,
at 0.4 m resolution over [-40,-40,-1] to [40,40,5.4] metres.
Class IDs 0 through 17 remain intact: class 0 is valid and class 17 is free.
Only camera-visible, non-255 voxels enter valid_mask. Preview dot thinning and
surface filtering never affect training labels. No raw LiDAR or LiDAR masks
are read. Calibration accounts for each camera's acquisition pose.

Use a __main__ guard for Windows worker processes; num_workers=0 is also
supported. Training shuffles reproducibly by default; validation does not.
The data pipeline, occupancy model, and training entry point are implemented.

## Image encoder

src/models/image_encoder.py uses a shared ImageNet-pretrained ResNet-18 through
layer3 (stride 16), with a trainable 256-to-32-channel projection. The backbone
is frozen, including BatchNorm statistics. At 320x576 image resolution, features
are [B,6,32,20,36]. ViewTransformer projects voxel centres into the cameras,
samples those features, and averages valid views; OccupancyDecoder predicts
18 class logits per voxel. OccupancyModel connects all three components.

## Training

Start with the one-frame overfit check (batch size 1, zero workers, 100 epochs):

    .\.venv\Scripts\python.exe train.py --output-dir outputs/overfit

The first run downloads official ImageNet ResNet-18 weights if not cached.
CUDA is selected automatically when available, with mixed precision enabled.
Use --device cpu or --no-amp to override this. Only trainable parameters enter
AdamW. Visible class 0 and class-17 free space remain supervised; unobserved
and ignore-255 voxels are excluded. All-ignore batches are skipped.

For the full mini split (323 training and 81 held-out validation frames):

    .\.venv\Scripts\python.exe train.py --overfit-samples 0 --epochs 10 --output-dir outputs/full_mini

Validation defaults off for overfit runs and on for full-split runs. Override
with --validate or --no-validate. Validation uses eval mode and no gradients.
It reports masked cross-entropy and voxel accuracy, including free space;
this accuracy is not class-balanced IoU.

The output directory contains:

- config.json: training/model settings, image normalization, grid and camera metadata.
- metrics.jsonl: one JSON record per epoch with training/validation statistics.
- latest.pt: model, optimizer and gradient-scaler state from the latest epoch.
- best.pt: lowest validation-loss checkpoint, or lowest training-loss checkpoint
  when validation is disabled.

Use a separate output directory for each experiment; rerunning in the same
directory replaces its logs and checkpoints. Unweighted epoch loss is averaged
over supervised voxels. Class-weighted training uses the sum of target weights
as its averaging denominator. A whole epoch without supervision raises an error.
The current script starts new runs; a command-line resume option is not implemented.

Inspect options with:

    .\.venv\Scripts\python.exe train.py --help

Run regression checks with:

    .\.venv\Scripts\python.exe -m unittest discover -s tests

## Inspect a trained checkpoint

Evaluate the first training frame recorded in your checkpoint:

    .\.venv\Scripts\python.exe evaluate.py --checkpoint outputs/overfit/best.pt

Results are saved under the checkpoint folder's evaluation directory:
metrics.json, per_class.csv, predictions.npz, bev_comparison.png,
per_class_iou.png, and camera_comparison.png. Use --output-dir to choose
another location, or --split val --sample-index 0 to inspect a held-out frame.

All metrics use camera-visible, non-ignore voxels. Occupied-class mIoU averages
IDs 0..16 with nonzero union; class 17 (free space) is reported separately.
Classes absent from both target and prediction have undefined IoU and are
excluded from macro averages. Overall accuracy is compared against predicting
free space everywhere. A training-frame diagnostic does not measure held-out
generalization.

Camera overlays show occupied surface centres only, with the same small-dot
settings and supervision mask for ground truth and prediction. Free space
is omitted from camera dots. Top-down images select the highest supervised
occupied voxel in each XY column; metrics still use all supervised 3D voxels.

## Class weighting and XYZ experiment

The original baseline remains available. Enable inverse-square-root class
weighting with --class-balance sqrt-inverse. Weights are estimated only from
camera-visible, non-ignore labels in the selected training samples, never
from validation. Observed class weights are normalized to mean one; absent
classes keep neutral weight one. The weights and counts are saved in config.json.
Validation and evaluate.py still report unweighted loss for comparison.

Enable positional information with --xyz. OccupancyModel normalizes voxel-centre
XYZ coordinates using the grid bounds and embeds them into the existing 32 feature
channels with a learned 1x1x1 convolution. This embedding is added before the
decoder. The decoder source, channel counts and convolutional stages are unchanged.
The XYZ projection starts at zero, preserving the baseline's initial output.
The model's use_xyz setting is saved in checkpoints; older checkpoints still load.

Example new run, in a separate output directory:

    .\.venv\Scripts\python.exe train.py --class-balance sqrt-inverse --xyz --output-dir outputs/new_xyz_run

The first controlled checks used the same frame, seed 0, learning rate 1e-4,
batch size 1, frozen pretrained backbone, and 100 updates per run. The saved
comparison is outputs/overfit_comparison/results.csv. Occupied-class mIoU
was 0% originally, 0.09% with class weighting, and 0.45% with weighting plus
XYZ. Those are one-frame fit diagnostics, not validation results; the model
still predicts free space for most supervised voxels.
