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

By default, the script streams all six camera views through every scene in the
mini dataset. It includes intermediate camera frames from `sweeps/`, aligns views
to the nearest front-camera timestamp, and paces playback using recorded timestamps.
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
`--scene` to select a scene, and starts at its first recorded camera frame.

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

**The standard nuScenes mini download does not include dense voxel occupancy
labels. No real occupancy annotations are currently installed.** Place compatible
labels under `data/occupancy/gts`, or pass the location of another `gts` directory.
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

After installing real occupancy labels, inspect one sample with:

```powershell
.\.venv\Scripts\python.exe load_nuscenes.py --no-show --occupancy-root data/occupancy/gts
```

## Recreate the environment

Python 3.12 is used for this project.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
