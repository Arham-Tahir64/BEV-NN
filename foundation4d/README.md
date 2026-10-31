# Lightweight camera-only 3D occupancy

The current model is the geometry-aware variant: ResNet-18 image features,
depth-aware camera fusion, stride-8 detail, eight height references, binary and
semantic 3D heads, and masked depth/boundary/ray supervision.

- [Architecture, losses and limitations](GEOMETRY.md)
- [Current commands](RUNNING.md)
- [Previous model documentation and launchers](../legacy/README.md)

Start training from the repository root:

```powershell
.\foundation4d\run_geometry.ps1 -Epochs 15
```

This is our compact adaptation, not NVIDIA Foundation4D or its pretrained weights.
The legacy architecture remains supported to load existing checkpoints. Historical
feasibility, planning, baseline profiles and baseline launchers live under `legacy/`.
