---
layout: ../../layouts/Post.astro
title: "Building an efficient camera-only 3D occupancy model"
description: "Starting with six cameras, an 8 GB GPU, and a question: how much 3D perception can I build into a model small enough for the edge?"
date: "2026-10-21"
number: "01"
status: "Architecture & learning"
tags: ["3D perception", "PyTorch", "Edge deployment"]
---

I’m building a camera-only 3D semantic occupancy network, and I’m documenting the decisions as I work through them. The goal is to take six surround-view RGB images and reconstruct a voxelized representation of the environment: where things are, and what they are.

That means predicting a 3D grid containing roads, vehicles, pedestrians, and other scene elements. It also means learning enough computer vision to understand why a particular architecture works, rather than just getting someone else’s model to run.

> **Where this project stands:** this first entry records my architecture design and research direction. The model choices and expected trade-offs below are proposals, not validated performance improvements. I don’t have IoU, mIoU, or deployment benchmarks to report yet.

## 01. Learning the fundamentals

I’m working through **Stanford CS231n: Deep Learning for Computer Vision** to build a stronger foundation for this project. Image feature extraction, spatial representations, and the jump from 2D observations to a 3D scene all become easier to reason about when the underlying operations make sense.

The project will use **PyTorch** with **nuScenes and Occ3D annotations**. I want to evaluate it along three axes: semantic prediction quality, memory use, and inference speed. A model that makes good predictions but cannot run on the target hardware would leave a large part of the project unfinished.

## 02. Letting the hardware shape the baseline

I started by exploring **SurroundOcc (ICCV 2023)** as a reference architecture for multi-camera semantic occupancy prediction. It gave me a concrete starting point for thinking about image features, camera fusion, and 3D decoding.

My hardware puts a practical constraint on how much of that architecture I can take on:

| Stage | Hardware |
| --- | --- |
| Training | NVIDIA RTX 4060 with 8 GB VRAM |
| Intended deployment | NVIDIA Jetson |

Based on my initial review, reproducing the full architecture looks too expensive for the setup I want to use. I’m starting with a lightweight, SurroundOcc-inspired baseline and treating memory efficiency and deployability as design constraints from the beginning.

The immediate goal is a complete, reproducible pipeline that I can understand and measure.

## 03. Designing a smaller first model

The proposed pipeline is:

```text
Six surround-view RGB images
          ↓
Shared ResNet-18 image encoder
          ↓
1×1 channel reduction: 256 → 32
          ↓
Geometry-based image-to-voxel sampling
          ↓
Masked mean fusion across valid cameras
          ↓
Two residual 3D convolutional blocks
          ↓
Semantic logits → spatial upsampling
```

### Image features

I’m planning to use a shared, pretrained **ResNet-18** for all six views, with inputs resized to **288 × 512**. At stride 16, the feature map has 256 channels; a 1×1 convolution reduces that to 32 before lifting features into 3D.

That reduction is deliberate: carrying a large channel dimension into a voxel grid can become expensive quickly.

### A coarse internal voxel grid

My initial design covers a **40 m × 40 m** horizontal region and uses a **50 × 50 × 8** internal voxel grid. The semantic logits would then be upsampled to **100 × 100 × 16**.

Upsampling gives a larger output grid, but it does not by itself recover detail lost in the coarse representation. Small objects and boundaries are things I’ll need to inspect when evaluating the baseline.

The dataset and model grids also need an explicit alignment step. The repository’s current data loader preserves the full Occ3D label grid; the smaller region described here is a model proposal. Bounds, axis order, class IDs, masks, and target resolution must agree before training.

### Geometry before learned fusion

For the first version, I want to replace image-to-voxel cross-attention with **geometry-based feature sampling**. A voxel is projected into each camera, image features are sampled at the projected locations, and valid camera contributions are averaged.

This gives me a relatively simple fusion operation to debug. It also avoids starting with expensive global voxel attention.

### A small 3D decoder

I’m planning two lightweight residual 3D convolutional blocks instead of a large 3D U-Net, followed by one semantic classification head and masked cross-entropy loss.

To keep that first baseline manageable, I’m deferring temporal fusion, multi-scale feature fusion, auxiliary losses, separate binary occupancy and semantic heads, custom CUDA kernels, and learned voxel-query mechanisms.

Each deferred component is something I can evaluate later against an established baseline.

## 04. Understanding the depth ambiguity

One limitation of simple geometric sampling is already clear: **projecting a voxel into an image does not establish that the voxel is visible**. Multiple points along a camera ray can land on the same image location, and some may be occluded.

That makes depth an important part of the feature-lifting problem.

I’ve started learning about **Feature Pyramid Networks (FPNs)**, which combine features at different spatial resolutions. I’m interested in whether multi-scale features could support depth estimation and improve reconstruction around boundaries and small objects.

An FPN alone does not resolve visibility. Multi-scale features and depth-aware lifting are related directions to investigate, with different jobs to do. For now, both remain potential extensions to the initial baseline.

## 05. Bringing attention back carefully

Removing attention reduces the complexity of the first implementation, but it leaves an open question about prediction quality.

My next research direction is **structurally sparse attention**, including DSA-style approaches. The idea I want to investigate is restricting attention to a smaller set of relevant features or sampling locations instead of considering every possible interaction.

This is a hypothesis to test, not an efficiency result I can claim yet. My planned comparisons are:

- Masked mean fusion versus learned attention-based fusion.
- Local or deformable attention versus DSA-style sparse strategies.
- Changes in IoU and mIoU alongside changes in latency and peak memory.
- Sparse attention versus spending the same compute budget on a larger backbone or decoder.

The useful question is whether the extra fusion capacity earns its cost on the hardware I’m targeting.

## 06. Planning for Jetson deployment

I want deployment constraints to inform the model while I’m still designing it. **TensorRT** is the inference-optimization path I plan to investigate for Jetson.

The deployment work will include exporting the PyTorch model, trying FP16 and potentially INT8 inference, and checking which projection, feature-sampling, or attention operations are supported efficiently.

I’m also interested in BEV-specific TensorRT plugins and specialized operators. Those may become relevant if standard layers leave a bottleneck in view transformation or voxel processing. They’re research directions at this stage; custom kernels are outside the first baseline.

The final measurements need to include **end-to-end latency, peak memory, and FPS on the target Jetson device**, alongside prediction quality. Those results will depend on the specific device and inference setup.

## 07. What I’ll measure next

The central research question is:

> Can a lightweight camera-only 3D occupancy network achieve useful semantic prediction quality while reducing memory requirements and inference latency enough for edge deployment?

Here’s the order I’m planning to work in:

1. Establish the ResNet-18, geometric lifting, and 3D CNN baseline.
2. Validate camera geometry, voxel projections, and the training pipeline.
3. Measure baseline IoU, mIoU, GPU memory consumption, and inference latency.
4. Introduce sparse attention and compare the accuracy-efficiency trade-off.
5. Try multi-scale features and depth-aware lifting where the baseline’s errors justify them.
6. Optimize with TensorRT and benchmark on Jetson.

The longer-term objective is to explore efficient 3D perception for resource-constrained autonomous systems. This first entry is the starting point: a set of design choices, the reasons behind them, and a plan for finding out which ones hold up.

### References & project links

- [Stanford CS231n](https://cs231n.stanford.edu/)
- [SurroundOcc — reference implementation](https://github.com/weiyithu/SurroundOcc)
- [nuScenes](https://www.nuscenes.org/)
- [Occ3D — datasets and annotations](https://github.com/Tsinghua-MARS-Lab/Occ3D)
- [BEV-NN — project source](https://github.com/Arham-Tahir64/BEV-NN)
