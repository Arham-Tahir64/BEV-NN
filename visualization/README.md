The scene player defaults to the Dice run’s best validation-mIoU checkpoint:
`outputs/semantic_bev_dice_seed0/best_semantic.pt`. It runs inference through a complete scene.
It shows the six camera images, ground truth and predicted BEV, and rotatable 3D
occupancy. It does not train or change the model or decoder.

From `C:\Users\Arham\Desktop\BEV-NN`, run:

```powershell
.\.venv\Scripts\python.exe visualization\scene_player.py
```

Open **http://127.0.0.1:8767/** and click **Play**. Keep the process running while
using the viewer. Stop it with Ctrl+C. The server binds only to this computer's
loopback interface. Change `--scene` to `scene-0916` for the other validation
scene or `scene-0061` for a training scene; restart the server after changing it.
Use `--benchmark` to time every keyframe before starting the viewer.

Controls:

- **Live inference:** recompute predictions from the current six-camera input
  for each requested frame. Turning this off reuses cached results where
  available; an uncached frame is still inferred once.
- **3D geometry:** voxel centres are lighter to transfer and draw; cube mode
  renders the exact exposed voxel faces. Neither mode samples away occupied
  voxels. Both retain all 16 height layers.
- **Sync 3D views:** links rotation, pan and zoom after the gesture settles.
  Camera positions persist across frames.
- **Height:** display exaggeration only, with metric Z labels. Choose ×1 for
  physically accurate proportions; ×3 and ×6 make vertical structure clearer.
- Play/pause, previous/next, slider, loop, and 0.25×–2× playback speed. Space and
  arrow keys also control playback.

This version follows the exact labeled nuScenes **keyframes at roughly 2 Hz**.
Scene-0103 contains 40 frames spanning 19.40 seconds. It is a synchronized
keyframe sequence, rather than full-rate camera-sweep video. Camera acquisitions
retain the dataset's small inter-camera timestamp differences; the existing
pose-aware geometry handles calibration. The model has no temporal fusion: each
prediction uses that frame's six images independently, in the current ego frame.

Camera-visible, non-ignore labels define the shared comparison region. Ground
truth is used for display only and is never supplied to the model. No raw LiDAR
or radar files are loaded. Free and unknown voxels are omitted in 3D. BEV selects
the highest occupied voxel in each XY column; the 3D view retains height.

The status bar separates model time, full backend time and browser fetch/render
time. Those times are not interchangeable. A one-frame lookahead overlaps the
next inference with rendering, while keeping every scene frame in order. The
player slows down when the requested speed cannot be maintained, rather than
dropping frames. Model FPS alone does not establish end-to-end real-time speed.

The full-scene cube-mode benchmark is saved under
`visualization/outputs/scene_player/scene-0103/benchmark.json`. With the RTX 4060, the measured
steady-state averages were 43.3 ms for model forward and 259.2 ms for backend
preparation; JSON transfer and browser rendering were excluded. Performance
varies with geometry, browser load and hardware.

Geometry transport tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s visualization/tests -p test_scene_player.py
```

All visualization code and results live together:

```text
visualization/
  README.md
  scene_player.py             # Live six-camera + BEV + 3D player
  scene_player.html           # Browser UI
  evaluate_run.py             # Full-split checkpoint comparisons
  visualize_voxels_3d.py      # Standalone BEV + 3D HTML/PNG exports
  tests/
  outputs/                   # Existing evaluations, viewers and previews
  _deps/                     # Local Plotly runtime
```

The `outputs/` and `_deps/` directories are ignored by Git. Source files remain
visible as new files in Source Control. They have not been staged or committed.
Paths are relative to these scripts, so they work independently of the terminal's
current folder. Model weights and raw datasets remain in the project's existing
`outputs/` and `data/` folders.

Generate the full-split evaluation or refresh the interactive exports:

```powershell
.\.venv\Scripts\python.exe visualization\evaluate_run.py
.\.venv\Scripts\python.exe visualization\visualize_voxels_3d.py --html-only
```

For scene-0916 use `--scene scene-0916`; for another checkpoint use
`--checkpoint PATH`. The live player uses the Dice checkpoint above. The standalone
export tools still use saved results from `train_xyz_full`; they do not automatically
regenerate predictions when the live checkpoint changes. `evaluate_run.py` remains
the legacy best/latest comparison tool.

Single-frame Dice evaluation (held-out validation frame):

```powershell
.\.venv\Scripts\python.exe evaluate.py --split val --sample-index 0
```

Restart an already-running scene player to load the new default checkpoint.
Explicit `--checkpoint PATH` overrides remain supported.

If the ignored Plotly runtime needs reinstalling:

```powershell
.\.venv\Scripts\python.exe -m pip install --target visualization/_deps plotly
```
