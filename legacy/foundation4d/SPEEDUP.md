# Batched camera sampling and larger execution chunks

The projector now samples all cameras in one grid_sample batch per query chunk.
It preserves the calibration, masks, sample positions, weighted fusion, model
parameters and training objective. For 100x100 queries and two stages, the original
camera loop at chunk 1024 required 120 grid_sample calls; batching cameras and using
chunk 4096 requires six. A few floating-point reduction-order differences are possible;
this is not a promise of bit-identical training trajectories.

Measured with the user's training paused, using its saved epoch-1 weights, a real
480x864 six-camera training frame, batch 1, AMP, model/loss/backward/AdamW, two warmup
and six measured steps per case. Comparison viewers remained running. Input was
already on the GPU, so these are not end-to-end DataLoader/epoch timings.

| Execution | Median training step | Peak allocated | Peak reserved |
|---|---:|---:|---:|
| Original, chunk 1024 | 201 ms | 1.497 GiB | 1.748 GiB |
| Batched cameras, chunk 4096 | 133 ms | 1.565 GiB | 1.887 GiB |
| Original repeated control | 197 ms | 1.496 GiB | 1.750 GiB |

This is about 33% less GPU-step time / 1.5x step throughput on the measured workload.
Actual epoch improvement depends on data loading, validation and other processes.
Earlier tests while training was active showed more variable but consistent gains.
Chunk 10000 used more memory with little extra benefit over 4096, so 4096 is the
recommended setting. Reducing CPU threads alone did not improve the shared-workload
benchmark; thread defaults were left unchanged. Workers remain unchanged.

## Resume the paused run

```powershell
cd C:\Users\Arham\Desktop\BEV-NN
.\.venv\Scripts\python.exe -m foundation4d.train --resume foundation4d/outputs/lite_480_depth0_offset0_seed0/latest.pt --epochs 8 --workers 2 --query-chunk 4096
```

The saved checkpoint was verified as epoch 1 and agrees with metrics.jsonl. Resume
starts epoch 2 with saved weights, AdamW state and AMP scaler. --query-chunk is an
explicit execution-only resume override and is saved in the new checkpoint config.
Changing batch size or resolution would be a different experiment; neither is needed.
The existing full-run checkpoint was not modified by the benchmark or installation.

The PowerShell fresh-run launcher defaults to QueryChunk=4096. Direct Python fresh
runs retain the older default 1024 unless specified. To lower temporary memory, use
--query-chunk 1024; camera batching still applies. An already-running interpreter
uses its previously loaded model code until restarted.

Validation: all 13 focused geometry/loss/runtime tests pass. Additional CPU and
CUDA-AMP comparisons with original sampling checked outputs, feature gradients and
parameter gradients for fixed and learned offsets, mixed valid/invalid cameras and
batch size 2. CPU maximum output difference was below 6e-8; CUDA-AMP outputs matched
in those fixtures. Strict checkpoint loading and a copied-debug-checkpoint resume
to chunk 4096 were verified, including the expected optimizer step count. The
user's actual run was not advanced by these checks.

Low utilization alone does not establish the entire bottleneck. The demonstrated
overhead here was repeated small sampling/weighting operations and launch scheduling.
More VRAM allocation is not itself a speed improvement. The machine has 8 GB dedicated
VRAM; Windows' larger total includes shared host RAM and is not equivalent GPU memory.
