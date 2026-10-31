# Experiment 1: semantic Dice and auxiliary BEV supervision

Extends the latest BEVDepth + XYZ + geometry/boundary loss experiment at 480 x 864.
This is a lightweight adaptation of semantic supervision ideas, not a reproduction
of Foundation4D. Accuracy improvements are hypotheses until held-out evaluation.

## Run controlled ablations

From the project root, run each variant separately:

```powershell
.\legacy\train_semantic_bev.ps1 -Variant baseline -Epochs 5
.\legacy\train_semantic_bev.ps1 -Variant dice -Epochs 5
.\legacy\train_semantic_bev.ps1 -Variant bev -Epochs 5
.\legacy\train_semantic_bev.ps1 -Variant both -Epochs 5
```

Default variant is `dice`, the lowest-cost first experiment. The launcher preserves
the train100/val20 split, image size, batch size, seed, class weighting and existing
loss weights. Each variant starts fresh from ImageNet initialization, uses its own
output directory, and refuses to overwrite an existing run. Compare equal epoch
budgets and several matching seeds before choosing a winner. The previously
continued boundary run has a different training budget and is not a fair control.

Training objective:

`weighted_CE + 3*depth_BCE + .2*geometry_BCE + .1*boundary_BCE + dice_weight*Dice + bev_weight*BEV_CE`

New CLI options `--semantic-dice-weight` and `--bev-aux-weight` default to zero.
The launcher uses .2 and .1 respectively when enabled; these are starting values,
not tuned optima. BEV supervision requires the `bevdepth` architecture.

Dice is computed in FP32, averages only occupied classes present in the batch,
and includes observed free voxels in probability denominators to penalize false
occupancy. Class 0 is occupied, class 17 is free, and 255/unobserved are ignored.
Cross-entropy continues to supervise absent classes. All-free batches contribute
zero Dice. A dense one-hot target tensor is not allocated.

The optional head averages final 3D decoder features over height, then applies a
32-to-32 3x3 convolution, ReLU and 32-to-18 1x1 convolution. Its target is the highest
observed occupied label per XY column. Unknown voxels above that surface make the
column ambiguous and excluded. Entirely known empty columns supervise free space;
unknown voxels below a known top surface do not invalidate its label. Height must
increase along the final tensor axis, as in this dataset. Unweighted BEV CE is
separate from Dice so the ablations isolate their contributions.

The head has 9,842 parameters (about 38.4 KiB in FP32) and executes only in training
with `return_aux=True`. It is skipped even by validation's auxiliary depth forward.
Disabled losses/head preserve legacy behavior and checkpoint keys. Enabled head
weights remain in checkpoints but incur no evaluation computation. Training
memory/time increase due to Dice probabilities and the auxiliary feature gradients;
no claim of zero training overhead is made.

Training JSON logs retain `loss` as the existing semantic CE for comparability.
New logs are `semantic_dice_loss`, `semantic_dice_classes` (sum of per-batch class
counts), `bev_aux_loss`, `bev_aux_cells` and `bev_aux_coverage` (trusted columns / all
columns in processed batches). Validation and checkpoint selection stay unchanged.
An epoch with BEV enabled but no trustworthy columns raises an error. Coverage can
vary with visibility; inspect it before interpreting an auxiliary-head ablation.

Compare validation occupied-class mIoU, per-class IoU, binary IoU, precision/recall
and boundary F1, alongside peak training memory and seconds per epoch. Auxiliary
loss values are not substitutes for held-out occupancy metrics.

## Resume and deploy

```powershell
.\.venv\Scripts\python.exe resume_train.py --checkpoint outputs/semantic_bev_both_seed0/latest.pt --epochs 10
```

Here `--epochs 10` means train through epoch 10, not ten additional epochs.
Resume restores the saved head configuration and loss weights; old checkpoints
default to both additions disabled. Resume extends that run rather than changing
the experiment definition. Do not load an old optimizer state into a newly added
head. For deployment call `model.eval()` and the ordinary logits-only forward.
To physically remove auxiliary parameters, construct the same model with
`use_bev_aux=False`, remove only `bev_aux_head.*` state keys, then load strictly.
Export the eval logits-only path. No ground-truth labels are model inputs.

## Verification

```powershell
python -m unittest discover -s tests -p test_semantic_supervision.py -v
python -m unittest discover -s tests -v
```

Tests cover masking, class averaging, free-space false positives, conservative
BEV targets, gradient flow, independent configuration, seeded core compatibility,
strict state loading, training integration and zero auxiliary calls in evaluation.

Verified locally on RTX 4060: all 129 tests passed. A two-frame, batch-1 smoke
test at six 480x864 images and 200x200x16 voxels, with both additions enabled,
completed two actual AdamW updates with finite, nonzero gradients through the
encoder, depth network, decoder and BEV head. Peak allocated CUDA memory was
2,437.8 MiB (2.38 GiB), reserved 2,730 MiB. This short run used random model
initialization, AMP with initial scale 1024, and the saved baseline class weights;
it is a correctness check, not a throughput or accuracy benchmark.

Trusted BEV coverage on those two frames was 1.67% and 3.25% (2.46% combined),
all occupied columns. This sparse, occupied-biased coverage is a real limitation
of the conservative top-surface target under camera visibility masks. Start with
Dice alone; measure coverage over the full split before relying on the BEV head.
Do not relax visibility masks merely to create more supervision. No full training
or held-out accuracy improvement has been measured for these additions yet.
