# Range-balanced depth and supported frustum refinement

These are independently switchable experiments motivated by the trained model's
ray-smearing audit. Improvements in accuracy or visual quality remain hypotheses.

## Training

```powershell
cd C:\Users\Arham\Desktop\BEV-NN
.\train_depth_support.ps1 -Variant range -Epochs 8
.\train_depth_support.ps1 -Variant support -Epochs 8
.\train_depth_support.ps1 -Variant both -Epochs 8
```

Use `-Variant baseline` for a matched control. Default is `both`. Each variant
starts a new model from ImageNet initialization, uses its own output directory and
refuses to overwrite runs. The train100/val20 split, six 480x864 images, 64 depth
bins, batch size 1, seed 0, weighted CE, depth weight 3, geometry weight .2 and
boundary weight .1 match the boundary high-resolution setup. Dice defaults to zero
because that run was the strongest overall reference; `-DiceWeight 0.2` instead
enables Dice in every variant you compare. Keep Dice, seeds and epoch budgets
identical across a controlled comparison. The auxiliary BEV head is disabled.

Output example: `outputs/depth_support_both_dice0_seed0`. Resume normally:

```powershell
.\.venv\Scripts\python.exe resume_train.py --checkpoint outputs/depth_support_both_dice0_seed0/latest.pt --epochs 12 --workers 2
```

The epoch argument is the total target epoch. Resume restores both new settings
from the checkpoint. Existing runs restore with their original legacy refinement
and unbalanced depth defaults. The supported architecture should be trained as a
fresh run: its zero-gain initial output is the raw context*p volume, rather than
the legacy refined volume. An old optimizer/checkpoint is not a safe warm start.
Inference tools reconstruct the saved architecture through model config.

## 1. Range-balanced depth loss

CLI: `--depth-range-balance --depth-range-cap 2`.

Use observed finite target depths in three camera-Z ranges: below 10 m, 10–20 m,
and 20 m or farther. In each batch find inverse-frequency weights jointly under
two constraints: the sample mean weight is one, and every sample weight is at
most the cap. This gives equal range contributions when the cap permits; very
rare ranges cannot dominate. Empty ranges are omitted. A single present range
gets weight one. A cap of one reproduces unweighted loss. No unknown targets are
invented and target generation/cache keys are unchanged. The same reducer supports
BCE, hard CE and linear-bin CE. Validation depth loss stays unweighted.

This increases far-target influence, not target reliability. Projected targets are
still approximate; audit independent measured depth before substantially raising
the cap. Weighting is training-only and adds no inference work. It adapts to each
batch, not the global dataset distribution.

## 2. Supported refinement

CLI: `--frustum-refinement supported` (requires BEVDepth).

Let V=context*p and R(V) be the existing frustum refinement network. Compute a
nonnegative depth gate by summing p in a +/-1-bin window and normalizing its sum
over depth. The new result is:

`V + tanh(gain) * gate * R(V)`

The trainable scalar gain starts at zero. The initial output is exactly V; the
gain learns on the first update and enables gradients to residual weights on later
updates. The original p-weighted path remains. Unlike the legacy refinement,
the residual cannot populate bins outside locally expanded support when p there
is exactly zero. Dense softmax tails are attenuated rather than hard-truncated.
Broad or misplaced depth distributions can still produce streaks; the gate cannot
correct them by itself. The gain may be negative: this allows a supported feature
correction, not an occupancy-probability interpretation.

The original refinement convolutions remain, with their existing row checkpointing;
this version does not claim a convolutional speedup. Added inference work is a small
depth-window pool, normalization and elementwise residual gate. It adds one scalar
parameter. Export/deployment performance on Jetson still needs measurement.

Defaults remain `--no-depth-range-balance` and `--frustum-refinement legacy`, so
existing model state dictionaries still load strictly. Supported checkpoints have
an additional `view_transformer.refinement.gain` key and must be constructed using
their saved model config. Do not use blanket `strict=False` to hide mismatches.

## Evaluation and verification

Training logs add `depth_range_counts_0_10_20_plus` and `depth_range_cap` when enabled.
The logged training depth loss includes range weighting; it is not directly
comparable to the unweighted training depth loss from a different run.

Validation keeps occupancy and boundary metrics unchanged, preserves argmax depth
MAE in `depth_range_mae_m`, and adds `depth_range_expected_mae_m` and
`depth_range_mass_within2m`. All three arrays use the 0–10, 10–20, 20+ m ordering.
These are against projected pseudo-targets, not independent measured depth.
Compare far-depth localization, far occupancy on matched frames, aggregate IoU,
semantic mIoU, precision/recall and boundary F1. A sharper depth distribution by
itself is not evidence of improvement.

All 136 regression tests passed, including the new validation diagnostics. Tests cover capped/mean-one weights, range edges,
all loss modes, ignored-target gradients, zero-gain identity, supported residual
gradients, support boundaries, CLI validation and strict checkpoint loading.
A two-frame real-data CUDA smoke test at full resolution completed two AdamW
updates with both options, finite nonzero gradients and a learned nonzero gain.
Peak allocated memory was 2,495.7 MiB (2.44 GiB), reserved 2,822 MiB. This used random
initialization and AMP scale 1024; it verifies execution, not accuracy or throughput.
No full training run was launched during implementation.
