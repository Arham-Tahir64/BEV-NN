# Feasibility for this RTX 4060

## Observed machine and reference workload

The local GPU reports RTX 4060 with 8,188 MiB VRAM. Physical system memory is
31.84 GiB, with 259.14 GiB free disk at inspection. Free VRAM was only 3,832 MiB
and available system RAM 8.70 GiB; these change with other processes. Existing
viewer processes may occupy GPU memory. Do not terminate them automatically;
stop viewers before model profiling or long training if headroom is inadequate.

Earlier full-resolution training of the local occupancy model successfully used
about 2.44 GiB allocated / 2.76 GiB reserved during a two-frame AMP smoke test.
That measurement establishes a working local baseline, not the cost of a new model.
The installed training environment is Python 3.12, PyTorch 2.11.0 and torchvision
0.26.0 on Windows. No additional runtime was installed during this assessment.

## Verified publication facts

Foundation4D uses DINOv2 ViT-L/14, width 1024; 1036-square images, a 37-square
aggregation grid and 16 aggregation layers. Its occupancy branch keeps hidden
features in BEV and introduces height at the outputs. [Paper, sections 3.3/3.6](https://arxiv.org/html/2610.30340v1).

Published all-head forward measurements use one H100 80 GB, six cameras:

| Timestamps per camera | Forward seconds per window | Allocated GiB |
|---|---:|---:|
| 24 | 9.673 | 32.47 |
| 36 | 14.480 | 45.44 |
| 48 | 19.782 | 58.41 |

These exclude data preparation, Gaussian rasterization, losses, metrics and export.
Representative four/eight-timestamp training updates took 3.8/8.3 seconds on
64 H100s. Stages used approximately 170k, 45k and 10k updates. [Paper, section 4.3/Table 17](https://arxiv.org/html/2610.30340v1).

## Availability and reproduction limits

The [official NVIDIA project](https://research.nvidia.com/labs/sil/projects/foundation4d/)
links its paper and dataset; no official code or weight download was verified at
inspection. Connected GitHub repository search for Foundation4D found no NVIDIA
implementation. A third-party Rust repository is not verified official code.
The [released annotation dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-Foundation4D)
is not pretrained model weights. Consequently neither weight loading nor an
official local fine-tuning recipe can currently be promised. Recheck sources before
any future reproduction attempt. No checkpoint or dataset was downloaded here.

## Engineering conclusion and estimates

The smallest published inference case alone allocates about 4.1 times the GPU's
total memory; training requires additional gradient, optimizer and activation
storage. These measurements rule out that published workload locally, but do not
establish a universal minimum for T=1, reduced resolution, selected heads or offload.
No reliable smaller-window minimum is published in the inspected evidence.

An 80 GB accelerator is a reasonable starting class for trying the measured full
inference workloads if official assets become available, not a guaranteed minimum
or a demonstrated full-training configuration. A full published-scale training
reproduction belongs on a multi-GPU cluster. A 24 GB consumer card would not fit
the reported 32.47 GiB case; smaller/offloaded variants would require fresh tests.

As a scale illustration only, extrapolating the representative training step times
over stages 1–2 yields about 18,124 aggregate H100 GPU-hours, roughly 283 hours
on 64 GPUs. This excludes stage 3, preprocessing and overhead, and assumes constant
step cost. It is derived arithmetic, not NVIDIA's reported total training bill,
and must not be converted to a 4060 duration using advertised FLOPS ratios.

No exact total model parameter count was verified. For budgeting intuition, every
100 million trainable FP32 parameters needs about 1.49 GiB for parameters, gradients
and two FP32 Adam moment tensors, before activations/workspaces. Precision choices
change this accounting. Freezing parameters or adapters saves optimizer memory,
but does not eliminate image/token activations. Quantizing weights does not by
itself make the original training workload fit. Host offload is also constrained
by system memory and transfer cost.

## Proposed local budget (targets, not measurements)

| Resource | Initial implementation target |
|---|---|
| Batch size | 1 six-camera timestamp |
| Input | Start 320x576; test 480x864 later |
| Training precision | FP16 AMP; sensitive geometry/losses FP32 |
| Allocated memory | <=6 GiB after warmup and optimizer initialization |
| Reserved memory | <=7 GiB with viewers stopped |
| Encoder | Compact ImageNet-pretrained CNN |
| Hidden scene features | BEV 100x100x64; retain height only in output heads |
| Temporal training | Deferred; later one detached past state |
| Runtime/accuracy | Unknown until measured; no real-time or mIoU promise |

At FP16, a 200x200x16x32 hidden voxel tensor is 39.06 MiB; a 100x100x64 BEV
tensor is 1.22 MiB. That is a 32x reduction for one hidden tensor, not the complete
model. The full 18-class output still needs 21.97 MiB. Four pillar heights and four
samples across six cameras create 960,000 feature samples per layer; materializing
64 FP16 channels for all of them would need 117.19 MiB per layer. Stream camera
and query chunks rather than assuming the BEV map alone determines memory use.

The compact representation and smaller images provide a credible route to the
8 GB budget. A real measured training step is still mandatory. If it fails, lower
query count, feature width, sample count or resolution before reducing output
coverage or changing the validation mask.
