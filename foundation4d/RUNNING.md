# Running the implemented Foundation4D-Lite pipeline

Performance update: cameras are now sampled as a batch. The launcher uses a 4096-
query execution chunk; resume can override it with `--query-chunk 4096` without
changing parameters or the objective. See SPEEDUP.md for measured results and the
paused epoch-1 run's exact resume command. The earlier profiles below used chunk 1024.

This is our compact camera-calibrated occupancy model, inspired by the paper's
BEV-to-height representation. It does not load NVIDIA Foundation4D weights.
It predicts a full 200x200x16 3D grid with 17 occupied classes plus free space.

## First full run

From the Desktop project root:

```powershell
cd C:\Users\Arham\Desktop\BEV-NN
.\foundation4d\run_train.ps1 -ImageHeight 320 -Epochs 8
```

This starts fresh from ImageNet initialization, using the existing train100/val20
split, batch 1, fixed seed 0, AdamW 1e-4 and mixed precision. It uses fixed four-point
sampling per pillar height, two projector/refiner stages, a 100x100 hidden BEV and
64 channels. The original 0.4 m target grid, calibration and known-voxel mask are
preserved. Default losses are binary BCE and occupied-only weighted semantic CE.
Depth auxiliary training and learned offsets are off so this is a clean baseline.
The launcher refuses to overwrite a nonempty run through the Python runtime.

After establishing that baseline, compare one change at a time:

```powershell
.\foundation4d\run_train.ps1 -ImageHeight 480 -Epochs 8
.\foundation4d\run_train.ps1 -ImageHeight 320 -Epochs 8 -LearnedOffsets
.\foundation4d\run_train.ps1 -ImageHeight 320 -Epochs 8 -DepthWeight 1
```

Depth targets are the existing approximate projected targets. They supervise only
the auxiliary image head, with shared encoder gradients, and are not inference
inputs or an unconditional depth lifting path. The first depth ablation weight 1
is an initial choice, not tuned. Keep epoch budgets/seeds fixed. For the old 480x864
model comparison, use the new 480x864 run rather than attributing resolution savings
to architecture alone. No full 8-epoch run was launched during implementation.

Outputs default to `foundation4d/outputs/lite_320_depth0_offset0_seed0` (or the
matching image/depth/offset/seed name). Stop with Ctrl+C; completed epochs remain.
An unfinished epoch is replayed on resume. Use a fresh directory for a new objective
or architecture; resume strictly restores the saved configuration and optimizer.

```powershell
.\.venv\Scripts\python.exe -m foundation4d.train --resume foundation4d/outputs/lite_320_depth0_offset0_seed0/latest.pt --epochs 8 --workers 2
```

`--epochs 8` means reach epoch 8, not add eight epochs. Every epoch is retained in
`epoch_001.pt`, `epoch_002.pt`, etc. Separate best.pt (validation objective),
best_iou.pt, best_semantic.pt and best_boundary.pt prevent loss of useful checkpoints.
Checkpoints contain optimizer, scaler and random-number states. Epoch-level loader
seeds reproduce sample ordering; CUDA operations can still introduce small numerical
differences. A checkpoint/log mismatch is rejected, not silently erased.

## Data and smoke tests

The default data source is `outputs/train_bevdepth_boundary_hires_train100/config.json`.
Use `--data-config PATH` and `--project-root PATH` to select another saved source.
Explicit disjoint scene lists are required. Raw images/labels are reused by path.
Class weights are estimated only from the selected training labels, never validation.
Targets/masks enter the loss; model inputs are RGB, intrinsics and camera transforms.

```powershell
.\.venv\Scripts\python.exe -m foundation4d.train --epochs 1 --overfit-samples 2 --val-samples 2 --max-train-batches 2 --workers 0 --output-dir foundation4d/outputs/my_smoke
```

This is labeled `bounded_debug` in config/logs. It cannot support claims about
held-out full-split performance. The verified smoke checkpoint supplied under
`foundation4d/outputs/smoke_verified` has only six total training updates across
three debug epochs. It is for pipeline/viewer verification, not quality comparison.

## Evaluation and the existing 3D viewer

```powershell
.\.venv\Scripts\python.exe -m foundation4d.evaluate --checkpoint foundation4d/outputs/lite_320_depth0_offset0_seed0/best_iou.pt
.\.venv\Scripts\python.exe -m foundation4d.evaluate --checkpoint foundation4d/outputs/lite_320_depth0_offset0_seed0/best_iou.pt --sample-index 0 --output-dir foundation4d/outputs/frame_preview
.\.venv\Scripts\python.exe visualization/scene_player.py --checkpoint foundation4d/outputs/lite_320_depth0_offset0_seed0/best_iou.pt --scene scene-0103 --port 8770
```

Full evaluation defaults to the entire checkpoint validation scene list. Single-frame
evaluation exports prediction/label/mask NPZ plus metrics. Output directories refuse
overwrite; choose a new destination for a different checkpoint or threshold.
The viewer supports both new and old checkpoints through an explicit architecture
factory. Its defaults still select the previous model; pass the new checkpoint.

Use the new evaluation command for this model: its objective is separate binary
BCE plus occupied-only CE. Default model forward returns 18 **decision scores** for
the legacy argmax viewer, not normalized joint semantic probability logits; do not
feed those scores into the old evaluator's CE loss. `forward_heads` returns the
actual training logits. Default binary threshold is 0.5. `--threshold` on evaluation
is explicit; tune on a separate calibration subset and freeze for reported results.

Validation logs include global occupied-class mIoU, binary IoU, precision/recall,
exact binary boundary F1, per-class metrics and near/middle/far/front/rear metrics.
Binary boundary F1 does not measure instance separation between touching objects.
New validation loss is not numerically comparable to the old single-CE loss.

## Tests and GPU resource profiles

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s foundation4d/tests -v
.\.venv\Scripts\python.exe -m foundation4d.tools.profile_train --image-size 320 576 --output-dir foundation4d/outputs/my_profile_320
.\.venv\Scripts\python.exe -m foundation4d.tools.profile_train --image-size 480 864 --output-dir foundation4d/outputs/my_profile_480
```

The profiler uses one real training frame, four warmups and at least 20 measured
optimizer updates, checks finite nonzero gradients in all primary modules and loss
decrease, and includes Adam state initialization before resetting memory peaks.
Reports contain synchronized forward/backward/optimizer stage times and separately
timed first-frame loading. Data loading, gradient inspection and logging are not
included in those stage timings; do not report their sum as end-to-end throughput.
Profile outputs include a single-frame NPZ. Default model configs target <=6 GiB
allocated / <=7 GiB reserved. Runtime also checks budgets at completed epochs;
they are checks, not a guaranteed prevention of an earlier CUDA OOM. Close competing
GPU viewers/jobs when profiling if free VRAM is inadequate; this code does not kill them.

## Current implementation boundaries

Implemented: single-frame compact model, fixed/bounded-learned sampling, optional
depth auxiliary task, masked objectives, full training/resume/evaluation, all epoch
and best checkpoint retention, existing viewer compatibility, synthetic geometry
and runtime tests, and real-data CUDA profiles.

The planned streaming temporal cache (M4) is deferred as the plan's optional stage.
The model currently uses no past/future frames, no detection head, Gaussian rendering
or learned ego motion. Geometry sampling still has depth/occlusion ambiguity; the
architecture does not guarantee elimination of streaks. Full-split accuracy remains
unknown until training and full validation finish.
