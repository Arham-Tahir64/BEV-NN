# Verified implementation status

Device: NVIDIA GeForce RTX 4060, 8 GB. PyTorch 2.11.0, torchvision 0.26.0,
Python 3.12. The default model has **11,548,578 parameters** (11.55 million).
The single-frame network preserves the 200x200x16 XYZ output and all original labels.

## Real-data GPU resource checks

The two comparison viewers were temporarily stopped for these profiles and restored
after installation. Each run used one real training frame, ImageNet initialization,
FP16 AMP at initial scale 1024, batch 1, four warmup updates and 20 measured updates.
Adam states were initialized before resetting the peak-memory counters.

| Six-camera resolution | Peak allocated | Peak reserved | Mean forward | Mean backward | Mean optimizer |
|---|---:|---:|---:|---:|---:|
| 320x576 | 1.292 GiB | 1.387 GiB | 108.7 ms | 84.6 ms | 5.2 ms |
| 480x864 | 1.497 GiB | 1.750 GiB | 112.5 ms | 96.4 ms | 4.9 ms |

All 24 updates completed at each resolution. All primary modules had finite nonzero
gradients. Training loss fell from 3.69 to about 2.29 in both short profiles. Timings
are synchronized stage means and omit input loading, gradient inspection, logging,
validation and checkpoint I/O. They are not end-to-end throughput or an inference
FPS benchmark. The warm input is reused from device memory during this microbenchmark.
These figures apply to the tested default architecture only, not every configurable
width/query/sampling/auxiliary combination.

A separate 128-update single-training-frame check reduced the unweighted objective
from 3.693 to 0.450. On that same training frame, binary IoU reached 82.45%, precision
94.98%, recall 86.20% and fixed-17-class semantic mIoU 31.26%. This is a learnability
check, NOT held-out performance. Its timings were not treated as the isolated profile.

## Functional checks

- Geometry, sampling centre alignment, camera permutation, masked/all-invalid queries,
  chunking equivalence, learned offset gradients, XYZ/class-height layout, inference
  decision-score equivalence, observed/free/unknown loss gradients and boundaries.
- Strict weights-only checkpoint serialization, optimizer restoration and RNG state
  round trip. Old viewer/model dispatch remains explicit.
- Complete CLI: two-frame train plus two-frame validation, checkpoint files, then
  resume latest.pt for a second debug epoch. Both epochs produced optimizer updates.
- Installed CLI: resumed again to epoch 3 and verified six total AdamW updates,
  then loaded that installed checkpoint through the updated 3D viewer.
- Evaluation CLI loaded the resumed checkpoint, ran a validation frame and exported
  predictions/labels/mask NPZ plus objective, occupancy, per-class, boundary and range metrics.
- A separate real-data two-update run enabled both learned offsets and depth auxiliary
  BCE, then completed validation with per-range depth diagnostics.
- Viewer SceneEngine loaded the new checkpoint strictly and produced six camera
  images, BEV output and a nonempty 3D mesh. The six-update debug checkpoint is
  explicitly not a trained quality model.

After installation, all 13 focused unit/runtime tests, 136 existing project tests
and five viewer tests passed (154 total).
No eight-epoch full training or full-validation model comparison was launched.

## Files containing evidence

- outputs/verification/profile_320.json and profile_480.json: isolated profiles.
- outputs/verification/overfit_320.json: single-training-frame learnability check.
- outputs/verification/viewer_smoke.json: viewer integration check.
- outputs/smoke_verified/latest.pt, config.json and metrics.jsonl: bounded debug run.

The implementation is a new model family. Loss magnitude is not comparable to the
previous single 18-class CE objective, and no inference quality gain is claimed.
Temporal history remains optional planned work; the completed pipeline is single-frame.
