# Current geometry-aware workflow

Run all commands from the project root. Dataset locations and explicit train/val
scene lists come from the existing local baseline data configuration. Do not delete
`outputs/train_bevdepth_boundary_hires_train100/config.json`: the default loader
still reads that configuration when starting a fresh run.

## Train

```powershell
.\foundation4d\run_geometry.ps1 -Epochs 15
```

Outputs: `foundation4d/outputs/lite_480_geometry_seed0`. A nonempty output folder is
never overwritten by a fresh run. Use the Python CLI with another `--output-dir`
for a separate experiment. See [architecture details](GEOMETRY.md) for settings.

## Resume

```powershell
.\.venv\Scripts\python.exe -u -m foundation4d.train `
  --resume foundation4d/outputs/lite_480_geometry_seed0/latest.pt `
  --epochs 15 --workers 2
```

The target is total epochs. Resume restores optimizer, AMP and model/loss settings.
Checkpoints save at epoch boundaries; interrupting mid-epoch repeats that epoch.

## View a saved model

```powershell
.\.venv\Scripts\python.exe -B -u visualization/scene_player.py `
  --checkpoint foundation4d/outputs/lite_480_geometry_seed0/epoch_009.pt `
  --scene scene-0103 --port 8770
```

Open http://127.0.0.1:8770/ . Choose a scene from the checkpoint's train/validation
scene lists. Keep checkpoint and scene selection explicit for comparisons.

## Checkpoint selection

`best_iou.pt`, `best_semantic.pt` and `best_boundary.pt` track their respective
validation metrics. `best.pt` tracks total validation loss, including depth loss;
these criteria can select different epochs. Numbered epoch files preserve choices.

[Earlier baseline commands](../legacy/foundation4d/RUNNING.md) are historical.
