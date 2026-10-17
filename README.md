# nuScenes mini loader

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
