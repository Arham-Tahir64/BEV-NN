# Geometry-aware occupancy variant

Launch from the Desktop project: `./foundation4d/run_geometry.ps1`.
Writes a separate run to `foundation4d/outputs/lite_480_geometry_seed0`.
Fresh training is required: the projector parameter shapes differ from the older
four-height model. Resume only this variant's own latest.pt with the usual train
CLI; architecture, loss coefficients and normalization weights are restored.

## Implemented changes

- A predicted 64-bin depth distribution is computed before camera fusion. Its
  local three-bin probability mass supplies soft candidate-surface support.
- Camera-space depth uses the same calibrated transform as image projection.
  Depth is sampled with pixel and bin-centre conventions matching align_corners=False.
- A unit null mass preserves weak support after normalization, including the
  single-camera case. Appearance alone cannot cancel depth attenuation.
- Fusion exposes support, feature variance, camera coverage, depth entropy,
  cumulative depth probability and in-range support to the decoder. Cumulative
  probability is an occlusion cue, not an instruction to mark voxels empty.
- Stride-8 features are sampled directly and projected into the fusion channels.
- Eight height references and depthwise vertical mixing precede BEV compression.
- Boundary BCE focuses on both endpoints of known occupied/free transitions.
- Ray-focused BCE groups up to 8,192 deterministic known voxel samples by projected
  stride-16 cell, then emphasizes nearest sampled annotated surfaces and labelled
  free samples before them. Unknown and behind-surface samples receive no ray
  gradient. Grouping is approximate and is NOT physical first-hit rendering.
- Existing capped height weighting remains enabled at 1.5.
- Depth uses linearly interpolated bin CE and capped range balancing to reduce
  hard-bin and near-range bias. Targets remain approximate projected annotations.
- Validation now reports below-1m, above-1m and above-2m metrics; the above-1m
  summary is printed each epoch. Masks continue to exclude unknown space.

Default new-run coefficients: binary 1, semantic 1, depth 1, boundary .1, ray .05.
These are conservative starting settings, not validated optimal hyperparameters.
Query chunk is 1024 to limit activation memory. Legacy architecture remains the
default unless --geometry-aware is specified. Legacy state dicts remain loadable.

## Interpretation and limits

This implements the five prioritized structural/supervision fixes from the audit.
It does not establish that sparse geometry or ray smearing is solved. Training
must establish that. Per-height recall must improve together with IoU, not simply
with more predicted occupancy.

The model still has a compact BEV hidden representation and no temporal history.
Temporal fusion requires aligned historical frames and is not supplied by this
single-frame dataset interface. Unknown is masked supervision, not a third
predicted occupancy class. No unsupported unknown labels are invented.

The existing calibration chain and camera normalization were retained because
the audit identified no defect there. Threshold stays .5; no uncalibrated filling
or threshold reduction was added. Data augmentation and LR scheduling remain
unchanged; their absence was a generalization hypothesis, not an identified bug.

Depth auxiliary loss is included in validation loss, so that scalar is not
comparable with the previous depth-disabled run. Compare IoU/mIoU/boundary and
height metrics. Training logs include boundary_loss, ray_loss and
training_objective separately; the legacy loss field omits boundary/ray terms.

## Verification

20 unit/runtime tests passed, covering legacy behavior, gradient flow from
occupancy into depth/fine features, camera permutation, all-invalid views,
non-cancelling support, and exclusion of unknown/behind-hit ray gradients.
One-frame train plus held-out validation smoke at 480x864 completed with AMP:
11,589,218 parameters, 3.515 GiB peak allocated GPU memory on RTX 4060.
This smoke result is not a reconstruction-quality evaluation.
