---
layout: ../../layouts/Post.astro
title: "Building an efficient camera-only 3D occupancy model"
description: "From geometric lifting to a fuller BEVDepth-style occupancy model: less ray smearing, more recovered structure, better-looking height, and remaining errors at distance."
date: "2026-10-21"
lastUpdated: "2026-10-26"
number: "01"
status: "BEVDepth visual results"
tags: ["3D perception", "PyTorch", "Edge deployment"]
---

I’m building a camera-only 3D semantic occupancy network, and I’m documenting the decisions as I work through them. The goal is to take six surround-view RGB images and reconstruct a voxelized representation of the environment: where things are, and what they are.

That means predicting a 3D grid containing roads, vehicles, pedestrians, and other scene elements. It also means learning enough computer vision to understand why a particular architecture works, rather than just getting someone else’s model to run.

> **Updated October 26, 2026:** the [new BEVDepth-style implementation and two result views](#13-from-lightweight-depth-to-a-fuller-bevdepth-architecture) show much less ray smearing, more recovered occupied structure, and better-looking object heights. Distant smearing still appears in playback and improves as the ego car gets closer. Earlier sections preserve the design and implementation history; the visual improvements are observations, not numerical accuracy gains inferred from screenshots.

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

## 08. What the model actually does now

**October 23, 2026 — implementation update.** The baseline is now connected end to end in `OccupancyModel`: an image encoder, a geometric view transformer, and a 3D occupancy decoder. The training entry point is implemented too. This is a concrete step beyond the design proposal, but having those pieces connected is different from learning useful scene reconstruction.

Some choices changed during implementation:

| Component | Original proposal | Current training code |
| --- | --- | --- |
| Input resolution per camera | 288 × 512 | 320 × 576 |
| Image backbone | Shared pretrained ResNet-18 | Shared pretrained ResNet-18 through layer3; backbone and BatchNorm statistics frozen |
| Image feature channels | 256 → 32 | Trainable 1×1 projection, 256 → 32 |
| Horizontal coverage | 40 m × 40 m | 80 m × 80 m: −40 to +40 m on both axes |
| Internal voxel grid | 50 × 50 × 8 | 200 × 200 × 16 at 0.4 m resolution |
| Decoder | Two residual 3D blocks | Two plain 3×3×3 convolutions with ReLU; no residual connections |
| Output | Upsample logits to 100 × 100 × 16 | Predict 18 class logits directly at 200 × 200 × 16; no spatial upsampling |

These details come from the model and dataset settings used by `train.py`, rather than the older `configs/baseline.ymal` sketch, which the training entry point does not load.

### From six images to a dense voxel prediction

Here is the implemented forward pass, with `B` representing batch size:

```text
Images: [B, 6, 3, 320, 576]
    ↓ shared frozen ResNet-18 through layer3
Image features: [B, 6, 256, 20, 36]
    ↓ trainable 1×1 convolution
Reduced features: [B, 6, 32, 20, 36]
    ↓ calibrated projection + bilinear sampling
    ↓ masked mean across valid camera views
Voxel features: [B, 32, 200, 200, 16]
    ↓ Conv3d(32 → 32, 3×3×3) + ReLU
    ↓ Conv3d(32 → 32, 3×3×3) + ReLU
    ↓ Conv3d(32 → 18, 1×1×1)
Raw class logits: [B, 18, 200, 200, 16]
    ↓ argmax over the 18 classes for prediction
Semantic labels: [B, 200, 200, 16]
```

The grid contains **640,000 voxels**, spanning −1 to 5.4 m vertically. It preserves the dataset’s `[X, Y, Z]` order, so the model and labels share the same grid without a crop or resize in this baseline.

For each voxel centre, the view transformer uses the ego-to-camera transform and resized camera intrinsics to find an image location. It samples the corresponding feature vector with bilinear `grid_sample`, then averages contributions from cameras where the voxel has depth greater than 0.1 m and projects inside the image. Voxels with no valid camera contribution receive a zero feature vector before decoding.

This is **geometric validity, not an occlusion test**. The model does not yet predict depth, learn camera-fusion weights, or determine whether a projected voxel is hidden behind another surface. Several voxels along a camera ray can sample the same image evidence.

Projection and sampling run in chunks of up to **65,536 voxels** to limit temporary tensors. This does not make the whole network sparse: the fused feature volume and the 3D decoder still operate on the full dense grid.

### What is trainable, and what gets supervised

The pretrained ResNet backbone stays frozen, including its BatchNorm statistics. The 1×1 image projection, the two 3D convolution layers, and the classifier are trainable. Gradients pass through feature sampling to the trainable image projection; the camera fusion itself has no learned weights.

The output classes are **0–17**, with **17 representing free space**. Class 0 remains a valid supervised class; 255 is an ignore label, not an extra output class. The loss is currently unweighted cross-entropy over camera-observed, non-ignore voxels. Unsupervised targets are replaced with 255 for the loss.

Those ground-truth supervision masks are used to select training and evaluation targets. They are **not inputs to the model**. At inference, its inputs are the images and camera calibration.

`train.py` defaults to a one-frame overfit check: batch size 1, 100 epochs, AdamW with a learning rate of `1e-4`, and CUDA mixed precision when available. Validation defaults off in this mode; `--overfit-samples 0` selects the full mini training split and enables held-out validation by default. These are code defaults, not a verified record of the run that produced the figure below.

## 09. First output: every supervised voxel is free

The first-iteration visualization is useful because it makes the current failure visible:

<figure>
  <a href="/BEV-NN/images/first-iteration-occupancy.png" aria-label="Open the first-iteration occupancy comparison at full resolution">
    <img src="/BEV-NN/images/first-iteration-occupancy.png" width="2560" height="960" loading="lazy" alt="Three-panel first-iteration diagnostic: ground truth contains road and scene objects, the prediction labels every supervised voxel as free, and per-column voxel accuracy varies from zero to one." />
  </a>
  <figcaption>First-iteration diagnostic supplied for this update. Left: ground-truth semantics. Middle: all supervised predictions are free. Right: fraction of supervised 3D voxels classified correctly in each XY column. <a href="/BEV-NN/images/first-iteration-occupancy.png">Open the full-resolution figure</a>.</figcaption>
</figure>

The left panel shows the highest supervised occupied voxel in each XY column. It contains a road corridor, vegetation, terrain, structures, and other semantic classes. It is a top-down summary of a 3D volume, so a pixel here is not a single complete account of everything above and below it.

The middle panel is labelled **“all supervised voxels are free.”** On the evaluated region shown, the model is selecting class 17 everywhere instead of identifying occupied surfaces and their semantic classes. The grey footprint does not establish that it learned the shape of the scene: the displayed pattern also reflects the supervision mask. Dark unsupervised areas are not evidence of correctly predicted empty space.

### Why parts of the accuracy map are still yellow

The right panel measures correctness across the supervised height entries in each XY column. A column can contain one occupied road voxel and many free voxels above it. Predicting free at every height gets the free entries right while missing the road entirely.

For illustration, if a column had 15 supervised free voxels and one supervised road voxel, an all-free prediction would score **15/16 = 93.75% accuracy** in that column while failing to recover the road. This is an explanatory example, not a measured count from this image.

When all supervised predictions are free, column accuracy is simply the fraction of supervised targets in that column that are free. Yellow means a high fraction correct; it is not an uncertainty score or evidence of successful semantic reconstruction. White areas have no reported supervised accuracy in this visualization.

This also explains the limitation of the current validation statistic in `train.py`: it counts correctly classified supervised 3D voxels, **including free space**. Overall voxel accuracy alone can hide a complete failure on occupied classes. For any occupied class present in this particular evaluated sample, an all-free prediction has no true positives. That observation is not a dataset-wide IoU benchmark.

### What I can and cannot conclude yet

The code now implements the full camera-to-voxel path, and this diagnostic shows a degenerate all-free prediction on the supplied example. It does **not** show useful occupied-scene reconstruction yet.

The result is consistent with a free-space majority shortcut under unweighted voxel loss. That is a working hypothesis, not a confirmed root cause. An early checkpoint, optimization behaviour, feature or geometry problems, and class balance still need to be checked. The image alone does not identify which of these is responsible, prove that the geometry is correct, or establish performance on other frames.

I’m keeping this figure as a qualitative development snapshot. Its training epoch, checkpoint identity, sample token, and train/validation provenance are not supplied with the image, so I’m not assigning it a numerical benchmark or calling it held-out validation.

## 10. The next milestone: recover occupied structure

Before adding attention or a larger backbone, I want to establish that the baseline can learn more than the free-space class. The immediate checks are:

1. **Record the experiment.** Pair each figure with its checkpoint, epoch, sample token, split, configuration, and loss history.
2. **Inspect class balance and predictions.** Count supervised targets and predicted labels by class, track the fraction predicted free, and inspect a confusion matrix.
3. **Complete the one-frame overfit check.** Look for occupied surfaces and semantic classes returning on that same frame as training proceeds. This tests fitting ability, not generalization.
4. **Check the lifting path.** Inspect camera projections, valid-view coverage, sampled features, and gradients through the trainable layers before assuming the architecture needs more capacity.
5. **Evaluate occupied performance explicitly.** Add occupied-vs-free IoU, occupied recall, per-class semantic IoU, and a clearly defined semantic mIoU alongside loss and voxel accuracy. These metrics are not implemented by the current `train.py` validation function.
6. **Test a targeted change.** If the measurements support class imbalance as the problem, compare a class-balanced loss against the current unweighted baseline, then evaluate on held-out scenes.

The current milestone is getting the implemented baseline to recover occupied structure reliably. Sparse attention, depth-aware lifting, and Jetson optimization remain later experiments once I have a meaningful baseline to compare against.

### Implementation references

This update describes [code revision 7f8f7d1](https://github.com/Arham-Tahir64/BEV-NN/commit/7f8f7d1), so later implementation changes do not silently redefine this snapshot:

- [Image encoder](https://github.com/Arham-Tahir64/BEV-NN/blob/7f8f7d1/src/models/image_encoder.py): frozen shared ResNet and trainable channel projection.
- [View transformer](https://github.com/Arham-Tahir64/BEV-NN/blob/7f8f7d1/src/models/view_transformer.py): voxel centres, projection, sampling, and masked mean fusion.
- [Occupancy decoder](https://github.com/Arham-Tahir64/BEV-NN/blob/7f8f7d1/src/models/occupancy_decoder.py) and [model wrapper](https://github.com/Arham-Tahir64/BEV-NN/blob/7f8f7d1/src/models/occupancy_model.py): dense 3D decoding and the connected forward pass.
- [Training entry point](https://github.com/Arham-Tahir64/BEV-NN/blob/7f8f7d1/train.py) and [dataset](https://github.com/Arham-Tahir64/BEV-NN/blob/7f8f7d1/data/nuscenes_dataset.py): active settings, supervision, optimization, and validation accuracy.

## 11. Latest visualization: where does a surface start and end?

**October 23, 2026 — depth ambiguity update.** The latest supplied comparison contains occupied semantic predictions, unlike the earlier all-free example. But their spatial extent is still wrong in important places: the prediction has long, ray-like streaks where the ground truth has more localized structures.

<figure>
  <a href="/BEV-NN/images/depth-ambiguity-occupancy.png" aria-label="Open the latest ground-truth and prediction comparison at full resolution">
    <img src="/BEV-NN/images/depth-ambiguity-occupancy.png" width="1545" height="706" loading="lazy" alt="Ground truth on the left and model prediction on the right, each with a top-down view above a 3D voxel view. Predicted semantic regions form elongated radial streaks compared with the more localized ground-truth structures." />
  </a>
  <figcaption>Latest supplied visualization: ground truth on the left, prediction on the right. Both are displayed in the same camera-visible, non-ignore comparison region. The top panels summarize occupancy in BEV; the lower panels show its 3D structure. <a href="/BEV-NN/images/depth-ambiguity-occupancy.png">Open the full-resolution image</a>.</figcaption>
</figure>

Seeing occupied labels is different from placing them correctly in 3D. In this example, semantic evidence appears spread along directions through the scene rather than consistently stopping at object boundaries. The display is filtered to a shared supervised region; it does not show complete predictions in unobserved space. The image does not supply a checkpoint, epoch, or sample identity, so I’m treating it as another qualitative observation rather than a measured improvement over the previous run.

### Many 3D voxels can share one image location

The current lifting operation asks: **“Where would this voxel appear in each camera?”** It does not explicitly answer: **“At what distance did the camera actually observe a surface?”**

Perspective projection divides camera-frame coordinates by depth. Points at different distances along the same camera ray therefore project to the same image location:

```text
Camera ── voxel A ── visible surface B ── hidden voxel C
           \________ same image location __________/

Different candidate depths → the same 2D appearance evidence
```

In the baseline, these voxels can sample the same image feature even though one is in front of the surface, one lies on it, and another is behind it. The stride-16 feature map also means nearby image projections can sample similar coarse features. Being in front of the camera and inside the image makes a sample geometrically valid; it does not establish that the voxel is occupied or visible.

The decoder must learn where that appearance belongs in depth from the available features, other views, and supervision. Without an explicit depth estimate or visibility mechanism, it can struggle to determine **where an object starts and ends along a ray**. The radial stretching in this visualization is consistent with that ambiguity. It does not by itself rule out calibration, sampling, optimization, or other representation problems.

### Occlusion makes the missing information harder to recover

For an opaque object, a camera directly observes its front-facing surface, not everything behind it. Sampling that surface’s image feature for a voxel farther down the ray does not reveal whether the voxel belongs to the same object, another object, or empty space.

Multiple views can help when they provide different views of the same region. But overlap is limited, and the current masked mean fusion does not learn which camera is occluded. This is why the geometry check and the ground-truth camera supervision mask are not substitutes for depth-aware lifting.

Even accurate visible-surface depth would not fully specify the object’s rear boundary or hidden structure. Those still require additional views, learned scene priors, or occupancy reasoning. Depth would provide a stronger constraint on feature placement, not a complete solution to 3D scene completion.

The [latest code snapshot, 69201d3](https://github.com/Arham-Tahir64/BEV-NN/commit/69201d3), also supports optional XYZ feature embeddings and class-balanced training. Coordinates can distinguish voxel positions, and class weighting can change the optimization balance, but neither directly measures surface depth. The screenshot alone does not establish which options were enabled for this run. The view transformer still uses geometric sampling and masked averaging without a depth head.

## 12. Investigating FPN and BEVDepth for depth-aware lifting

My next research direction is making feature placement more aware of distance. I’m looking into **FPN features** and **BEVDepth**, which address different parts of that problem and could be used together.

| Direction | What it contributes | What it does not provide by itself |
| --- | --- | --- |
| Feature Pyramid Network | Multi-scale image features combining finer spatial detail with stronger semantic context | A metric depth estimate or an occlusion decision |
| BEVDepth-inspired depth estimation | Explicitly learned depth information to guide image-to-3D feature placement | A ready-made 3D semantic occupancy model or complete hidden geometry |

An **FPN** builds multi-scale features with a top-down pathway and lateral connections. I’m interested in those features as inputs to a depth-prediction module, especially near boundaries and small objects. Adding an FPN alone would still leave the depth-placement question unanswered; a depth head and a learning objective would be separate choices. [Feature Pyramid Networks paper](https://arxiv.org/abs/1612.03144)

**BEVDepth** is a more direct depth-estimation reference. It uses explicit depth supervision derived from LiDAR and camera-aware depth prediction for camera-based 3D detection. LiDAR supplies training targets; inference remains camera-based. Its original task is BEV object detection, so adapting its depth component to this occupancy pipeline would require preserving the 3D voxel representation rather than simply replacing it with a detection-oriented BEV output. [BEVDepth paper](https://arxiv.org/html/2206.10092v2)

The experiment I want to investigate is predicting a distribution over candidate depths for an image location, then using that distribution to weight where its features contribute in 3D. Instead of repeatedly assigning the same appearance without an explicit depth preference, the lifting stage would have a learned estimate of where the visible surface is likely to lie. That is a proposed change, not something the current model implements.

I want to compare the existing geometric baseline, an FPN-only feature variant, and a depth-aware variant while keeping the data split and evaluation masks fixed. The checks should include occupied IoU, per-class semantic IoU, boundary quality, and the extent of ray-like errors, alongside depth quality where targets are available, memory use, and latency. That would help distinguish the benefit of richer image features from the benefit of explicit depth information.

For now, the question driving the next iteration is: **can depth-aware lifting reduce the ambiguity about where image evidence belongs in 3D, without losing the efficiency that motivated this baseline?**

## 13. From lightweight depth to a fuller BEVDepth architecture

**October 26, 2026 — implementation and visual results.** Since the last update, I implemented a lightweight BEVDepth-inspired depth path. Ray smearing was still present, so I moved on to a fuller BEVDepth-style architecture adapted to 3D semantic occupancy. The two screenshots below are results from that newer implementation.

The biggest visual change is that the predictions are less stretched along camera rays. More of the occupied scene is being recovered, and the vertical extent of objects looks much closer to the ground truth. The remaining errors are now especially noticeable at a distance rather than dominating the entire scene.

### What changed beyond the lightweight depth head

The earlier depth path added a small depth classifier to frozen image features and used depth-conditioned inverse sampling into the voxel grid. That was a useful first experiment, but adding depth probabilities alone did not make the predictions spatially reliable. A broad or misplaced depth distribution can still spread appearance over multiple distances.

The newer `bevdepth` architecture changes several parts of the pipeline together:

| Stage | Fuller implementation |
| --- | --- |
| Image features | Trainable ResNet-18 residual stages, with four image scales aligned and fused at stride 16; the stem stays frozen and ImageNet BatchNorm statistics stay fixed |
| Depth and context | Separate camera-conditioned branches using calibration descriptors and channel gates |
| Depth estimation | Residual blocks, multi-scale dilated context through ASPP, deformable convolution, and a depth-bin classifier |
| Lifting | Multiply context features by their depth probabilities before placing them in 3D |
| Frustum refinement | Refine the depth-weighted features across depth and image width before voxel pooling |
| Voxel placement | Forward-unproject frustum samples through calibration and sum them into their destination voxels while retaining height |
| Occupancy decoding | A multi-scale residual 3D decoder with two XY downsampling stages and skip fusion back to the full grid; all 16 height layers remain |

```text
Six camera images + calibration
    ↓
Multi-scale image features
    ↓
Camera-conditioned depth + context branches
    ↓
Depth probabilities × context features
    ↓
Frustum refinement → forward voxel pooling
    ↓
Multi-scale 3D decoder → 18 occupancy class logits
```

This also brings U-Net-style multi-resolution processing and skip connections into the occupancy decoder, replacing the two plain 3D convolution layers used by the original baseline in this architecture mode.

By “fuller BEVDepth” here, I mean adopting its depth/context/refinement structure rather than adding only a lightweight depth head. This is still an occupancy adaptation, not an exact reproduction of the published detector: it uses narrower ResNet-18-based features, native PyTorch pooling, and a height-preserving occupancy decoder instead of the reference BEV detection output. The implementation and these differences are recorded in [BEVDEPTH_STRUCTURE.md at revision c932fb6](https://github.com/Arham-Tahir64/BEV-NN/blob/c932fb6/BEVDEPTH_STRUCTURE.md), with the components in [bevdepth.py](https://github.com/Arham-Tahir64/BEV-NN/blob/c932fb6/src/models/bevdepth.py).

### Two result views from the newer model

<figure>
  <a href="/BEV-NN/images/bevdepth-results-view-1.png" aria-label="Open the first fuller-BEVDepth result screenshot at full resolution">
    <img src="/BEV-NN/images/bevdepth-results-view-1.png" width="1545" height="776" loading="lazy" alt="First newer-model result: ground truth on the left and prediction on the right, with BEV and 3D views showing a road corridor and more localized vertical structures than in the earlier ray-smeared result." />
  </a>
  <figcaption>Result view 1 from the newer BEVDepth-style occupancy implementation. Ground truth is on the left and prediction on the right, within the same visible, non-ignore comparison region. <a href="/BEV-NN/images/bevdepth-results-view-1.png">Open the full-resolution screenshot</a>.</figcaption>
</figure>

<figure>
  <a href="/BEV-NN/images/bevdepth-results-view-2.png" aria-label="Open the second fuller-BEVDepth result screenshot at full resolution">
    <img src="/BEV-NN/images/bevdepth-results-view-2.png" width="1588" height="769" loading="lazy" alt="Second newer-model result: side-by-side ground-truth and predicted BEV and 3D occupancy, with recovered road, cars, vegetation and upright structures, alongside remaining semantic and extent errors." />
  </a>
  <figcaption>Result view 2 from the same implementation stage. The 3D view makes object height and remaining spatial errors easier to inspect than BEV alone. These are two result views, not a controlled before-and-after pair. <a href="/BEV-NN/images/bevdepth-results-view-2.png">Open the full-resolution screenshot</a>.</figcaption>
</figure>

### What looks better now

**Ray smearing is much reduced.** Compared with the earlier visualization, occupied regions are more localized and the long radial streaks are less dominant. The reconstruction looks more like a scene with distinct structures rather than semantic colours extended along viewing directions.

**Occupied-scene coverage is higher.** In playback, I’m seeing substantially more occupied structure being recovered. This is the improvement I mean when I say the occupancy percentage looks higher. Predicting more occupied voxels is not automatically more accurate, so I still need occupied precision, recall, and IoU on matched evaluation frames to quantify how much of that additional occupancy is correct. I’m not assigning a numerical percentage from these screenshots.

**Object height looks better too.** The vertical extent of structures appears closer to the ground truth in the 3D comparison. The model retains all 16 height layers of the 0.4 m voxel grid. The viewer can exaggerate height for readability, so physical height comparisons need matching display settings and voxel-space measurements rather than screen height alone.

These are visual observations from the newer model, not evidence that any one component caused the improvement. Image features, depth estimation, lifting, refinement, and the decoder changed together.

### Distance is still the difficult case

During playback, I still see ray smearing farther from the ego vehicle. As the car approaches those regions, the predictions become cleaner and the structures more localized. That is an observation across the moving sequence; these two still images alone do not establish the near-versus-far change.

My working explanation is that distant objects occupy fewer image pixels and give the model less detail for resolving depth and boundaries. A depth distribution spread over several candidate distances can still distribute features along a ray. The current default depth grid uses 64 one-metre bins, which is another design choice to investigate alongside image resolution and depth-target quality. These are possible contributors, not a diagnosis established by the screenshots.

The architecture is still single-frame: cleaner predictions as the car gets closer come from new camera observations at each frame, not from an implemented temporal memory. Occlusion and hidden surfaces also remain ambiguous even when the visible geometry looks better.

The next useful comparison is distance-binned evaluation: near, middle, and far occupied precision/recall and IoU, together with depth errors where targets exist. I also want to inspect upper-height occupancy and follow the same scene regions as the vehicle approaches. That should show whether the cleaner playback corresponds to better geometry and help target the remaining distant smearing.


### References & project links

- [Stanford CS231n](https://cs231n.stanford.edu/)
- [SurroundOcc — reference implementation](https://github.com/weiyithu/SurroundOcc)
- [nuScenes](https://www.nuscenes.org/)
- [Occ3D — datasets and annotations](https://github.com/Tsinghua-MARS-Lab/Occ3D)
- [BEV-NN — project source](https://github.com/Arham-Tahir64/BEV-NN)
