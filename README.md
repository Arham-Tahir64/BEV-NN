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

The script loads all metadata and reads the first sample's front camera image,
LiDAR point cloud, and annotations. It prints their shapes and file paths.

```powershell
.\.venv\Scripts\python.exe load_nuscenes.py --sample-index 10 --camera CAM_BACK
.\.venv\Scripts\python.exe load_nuscenes.py --dataroot D:\datasets\nuscenes --version v1.0-trainval
```

## Use in Python

```python
from load_nuscenes import load_nuscenes, load_sample

nusc = load_nuscenes()
item = load_sample(nusc, sample_index=0)
rgb = item["image"]             # H x W x 3 uint8 RGB array
points = item["lidar_points"]   # N x 4 float32: x, y, z, intensity
annotations = item["annotations"]
```

LiDAR points are in the LiDAR sensor frame. Annotation positions are in the
nuScenes global frame. Apply devkit calibration and ego poses before combining
them or projecting onto camera images.

## Recreate the environment

Python 3.12 is used for this project.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
