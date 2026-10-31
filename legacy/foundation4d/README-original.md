# Foundation4D: RTX 4060 feasibility and lightweight implementation plan

This is an implemented, locally trainable lightweight occupancy model. It contains
calibrated BEV sampling, full 3D binary/semantic outputs, training/resume/evaluation,
checkpoint retention, viewer integration, geometry tests and real-data GPU profiles.
It is our adaptation of the occupancy representation, not NVIDIA's model or weights.

**Decision:** full published Foundation4D training is impractical on the available
RTX 4060 8 GB. The compact implementation has passed full-resolution training smoke
tests on that GPU. Full-split accuracy remains unknown until a complete run.

The user's current scope is RTX 4060 training. Jetson deployment is deferred.

- [Hardware feasibility](FEASIBILITY.md)
- [Architecture and staged implementation](IMPLEMENTATION_PLAN.md)
- [Training, resume, evaluation and viewer commands](RUNNING.md)
- [Architecture configuration reference](../../foundation4d/configs/rtx4060_proposal.json)
- [Primary sources](../../foundation4d/sources.json)
- [Observed local resources](../../foundation4d/hardware_snapshot.json)
- [Measured verification results](VERIFICATION.md)

Run the standard-library-only calculator from the repository root:

```powershell
python foundation4d/tools/resource_budget.py
```

Its results are tensor/attention arithmetic, not a measured model profile.

Start a full run from the project root:

```powershell
.\legacy\foundation4d\run_train.ps1 -ImageHeight 320 -Epochs 8
```

The new model has 11.55 million parameters and a 200x200x16 voxel output. Start with
the fixed-sampling, depth-auxiliary-off baseline, then ablate the optional learned
offsets/depth task independently. The shared scene viewer now dispatches on the
checkpoint architecture; existing models/defaults remain supported. Data is reused
by path, without copying the dataset. Optional temporal history remains a later stage.
