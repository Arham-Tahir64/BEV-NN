# Training on more driving scenes

The installed mini split has 8 training scenes (323 frames) and 2 validation
scenes (81 frames). The downloaded official occupancy challenge trainval
package contains 700 training scenes (28,130 frames) and 150 validation scenes
(6,019 frames). Its Occ3D format matches this model: 200x200x16 voxels at 0.4 m,
ego coordinates, IDs 0..16 occupied, ID 17 free, and visibility/observation masks.
SurroundOcc's own 0.5 m/LiDAR-coordinate labels are a different format.

Downloaded files and integrity hashes are recorded in
`data/occ3d_trainval/downloads/download_status.json`. Extracted data lives in
`data/occ3d_trainval/imgs` and `data/occ3d_trainval/gts`. The supplied
`downloads/annotations.json` provides calibration and frame/camera ego poses.
No raw LiDAR, sweeps, detection annotations, or separate nuScenes metadata
download is needed for this loader path. The existing mini path still works.

## Start with 100 training scenes and 20 validation scenes

The prepared subset keeps mini scenes whose train/val roles agree with the
full official split and adds randomly selected complete scenes with seed 0.
That retains six original training scenes and both original validation scenes.
Mini training scenes `scene-0553` and `scene-0796` belong to the full validation
split and are excluded from full training. It includes whole
sequences, checks that every camera image and label exists, and never combines
training scenes with validation scenes. The inventory checks file presence;
image/label contents are validated when loaded. Existing files are not silently
overwritten by the subset tool.

```powershell
.\.venv\Scripts\python.exe prepare_scene_subset.py --dataset-version v1.0-trainval --dataroot data/occ3d_trainval --annotations-json data/occ3d_trainval/downloads/annotations.json --occupancy-root data/occ3d_trainval/gts --train-count 100 --val-count 20 --include-mini-scenes --seed 0 --output-dir outputs/splits/train100_val20
```

After preparation, train the baseline architecture with the additional data:

```powershell
.\.venv\Scripts\python.exe train.py --dataset-version v1.0-trainval --dataroot data/occ3d_trainval --annotations-json data/occ3d_trainval/downloads/annotations.json --occupancy-root data/occ3d_trainval/gts --train-scenes outputs/splits/train100_val20/train_scenes.json --val-scenes outputs/splits/train100_val20/val_scenes.json --epochs 5 --overfit-samples 0 --class-balance sqrt-inverse --xyz --depth --depth-fusion residual --geometry-loss-weight 0 --output-dir outputs/train_xyz_depth_train100
```

This uses a fresh run with the selected baseline architecture and objective,
with the additional geometry loss disabled so data expansion is the main
change. It does not resume or overwrite the preserved epoch-20 mini model.
Class weights are recomputed from the selected training annotations only.
No validation labels enter class-weight estimation or training.
Do not initialize the larger official-split run from our mini baseline weights:
that checkpoint already saw the two scenes which now belong to full validation.

The command above reproduces the earlier residual-lifting baseline. New depth
runs now default to corrected probability lifting. For the fresh 100-scene
comparison, use the separate command and output directory in
[LIFTING_FIX.md](LIFTING_FIX.md). Resuming an old checkpoint retains its saved
lifting mode; it does not apply the correction to that run.

The first pilot uses five epochs. After inspecting validation and the checkpoint,
extend that existing run to 20 total epochs with:

```powershell
.\.venv\Scripts\python.exe resume_train.py --checkpoint outputs/train_xyz_depth_train100/latest.pt --epochs 20
```

Run resume only after the existing process has stopped. Never run two trainers
against the same output directory. Training and errors are logged in
`outputs/train_xyz_depth_train100/train.stdout.log` and `train.stderr.log`.

Training 100 scenes is roughly 12 times the original frame count. At the
previous measured throughput, allow roughly 15–20 minutes per epoch; this is
an estimate, and file loading, depth-target generation and GPU activity affect
it. Batch size and image/grid shapes are unchanged, so more scenes mainly add
disk use and training time rather than increasing per-batch GPU memory.

## Full training split

Omit `--train-scenes` / `--val-scenes` to use the complete official 700/150
splits, or prepare separate manifests with `--train-count 0 --val-count 0`.
Use a new output directory. At mini throughput, a full-split epoch could take
around two hours; measure before committing to a large epoch budget.

Resume, evaluation and the scene viewer read dataset version, image root,
annotation path and selected scene names from checkpoints. Old checkpoints
without the new fields default to mini. Keep the epoch-20 checkpoint as the
reference and evaluate common validation scenes for direct comparisons;
the full 20-scene validation aggregate is a different evaluation set.

## Sources

- Official package and exact frame counts: https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction#download
- FlashOcc's use of the compatible gts: https://github.com/Yzichen/FlashOCC/blob/master/doc/install.md
- SurroundOcc format differences: https://github.com/weiyithu/SurroundOcc/blob/main/docs/data.md

## BEVDepth camera-only architecture

For the new architecture, its camera-only contract, measured GPU cost and a separate fresh-run command, see [BEVDEPTH_STRUCTURE.md](BEVDEPTH_STRUCTURE.md). Its recommended training command uses existing annotation-derived depth and requires no raw LiDAR.

