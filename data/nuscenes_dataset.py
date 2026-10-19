"""Load nuScenes metadata, six-camera keyframes, and optional occupancy labels."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nuscenes.nuscenes import NuScenes

DEFAULT_DATAROOT = Path(__file__).resolve().parent / "nuscenes"
CAMERAS = (
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)


def load_nuscenes(
    dataroot: str | Path = DEFAULT_DATAROOT,
    version: str = "v1.0-mini",
    verbose: bool = True,
) -> NuScenes:
    """Load the official devkit database; dataroot contains samples and metadata."""
    root = Path(dataroot).expanduser().resolve()
    if not (root / version / "scene.json").is_file():
        raise FileNotFoundError(
            f"Missing nuScenes metadata: {root / version / 'scene.json'}. "
            "Extract the dataset into dataroot, so it contains the version, "
            "samples, sweeps, and maps folders."
        )
    from nuscenes.nuscenes import NuScenes

    return NuScenes(version=version, dataroot=str(root), verbose=verbose)


def load_sample(
    nusc: NuScenes,
    sample_index: int = 0,
    camera: str = "CAM_FRONT",
    occupancy_root: str | Path | None = None,
) -> dict:
    """Read one keyframe's six RGB images and 3D annotations.

    images maps each camera name to a uint8 (height, width, 3) RGB array.
    image retains the selected camera for single-camera callers.
    Pass occupancy_root (the Occ3D gts folder) to attach voxel labels and masks.
    Annotations remain in the nuScenes global frame; no coordinate
    transformation or projection is applied.
    """
    if not 0 <= sample_index < len(nusc.sample):
        raise IndexError(f"sample_index must be between 0 and {len(nusc.sample) - 1}.")
    if camera not in CAMERAS:
        raise ValueError(f"Unknown camera {camera!r}; choose one of {CAMERAS}.")

    import numpy as np
    from PIL import Image

    sample = nusc.sample[sample_index]
    image_paths = {
        channel: Path(nusc.get_sample_data_path(sample["data"][channel]))
        for channel in CAMERAS
    }
    images = {}
    for channel, path in image_paths.items():
        with Image.open(path) as image:
            images[channel] = np.array(image.convert("RGB"))
    image_path = image_paths[camera]
    rgb = images[camera]
    annotations = [nusc.get("sample_annotation", token) for token in sample["anns"]]

    item = {
        "sample": sample,
        "scene": nusc.get("scene", sample["scene_token"]),
        "image": rgb,
        "images": images,
        "image_paths": image_paths,
        "annotations": annotations,
        "image_path": image_path,
    }
    if occupancy_root is not None:
        from .occupancy_labels import load_sample_occupancy
        item["occupancy"] = load_sample_occupancy(nusc, sample["token"], occupancy_root)
    return item


# Tensor preparation is kept here so both training and the viewer share metadata.
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def resize_intrinsics(intrinsics, original_size, target_size):
    """Update K for PIL resize. Sizes are (height, width); pixel centres matter."""
    if any(d <= 0 for d in (*original_size, *target_size)):
        raise ValueError("Image dimensions must be positive.")
    sy, sx = np.asarray(target_size, dtype=float) / np.asarray(original_size, dtype=float)
    resize = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    return resize @ np.asarray(intrinsics, dtype=float)


def prepare_camera_image(path, image_size):
    """Load RGB, resize to (H,W), normalize with ImageNet statistics, return CHW."""
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
        rgb = np.asarray(image, dtype=np.float32) / 255.0
    rgb = (rgb - np.array(IMAGE_MEAN, dtype=np.float32)) / np.array(IMAGE_STD, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))


class NuScenesOccupancyDataset(Dataset):
    """Six RGB views and dense Occ3D targets, indexed by official scene splits.

    Images: float32 [6,3,H,W], normalized using IMAGE_MEAN/IMAGE_STD.
    Intrinsics: float32 [6,3,3], for the resized images.
    Ego-to-camera: float32 [6,4,4], from the sample-time occupancy ego frame.
    Semantics: int64 [200,200,16], unchanged IDs 0..17, ignore=255.
    Camera/valid masks: bool [200,200,16]; valid excludes invisible and ignore.
    Class 17 (free space) remains a supervised class wherever valid.

    No dot thinning, surface filtering, or label resampling is used for training.
    Metadata is prepared once; only small records are pickled to Windows workers.
    """
    camera_order = CAMERAS
    grid_shape = (200, 200, 16)
    num_classes = 18
    ignore_index = 255
    voxel_size = (0.4, 0.4, 0.4)
    voxel_bounds = (-40., -40., -1., 40., 40., 5.4)

    def __init__(self, dataroot=DEFAULT_DATAROOT, occupancy_root=None,
                 version="v1.0-mini", split="train", image_size=(320, 576), nusc=None):
        from nuscenes.utils.splits import create_splits_scenes
        from .occupancy_labels import DEFAULT_OCCUPANCY_ROOT
        from .voxel_projection import camera_geometry, occupancy_label_path

        if len(image_size) != 2 or any(not isinstance(d, int) or isinstance(d, bool) or d <= 0 for d in image_size):
            raise ValueError("image_size must be two positive integers (height, width).")
        if version not in ("v1.0-mini", "v1.0-trainval"):
            raise ValueError("Training labels require v1.0-mini or v1.0-trainval.")
        valid_splits = ("train", "val", "mini_train", "mini_val") if version == "v1.0-mini" else ("train", "val")
        if split not in valid_splits:
            raise ValueError(f"Choose one of {valid_splits} for {version}.")
        split_name = f"mini_{split}" if version == "v1.0-mini" and split in ("train", "val") else split
        self.split = split_name
        self.image_size = tuple(image_size)
        self.occupancy_root = Path(occupancy_root or DEFAULT_OCCUPANCY_ROOT).expanduser().resolve()
        nusc = nusc if nusc is not None else load_nuscenes(dataroot, version, verbose=False)
        if nusc.version != version:
            raise ValueError(f"Metadata version {nusc.version} does not match requested {version}.")
        wanted = set(create_splits_scenes()[split_name])
        scenes = [scene for scene in nusc.scene if scene["name"] in wanted]
        missing_scenes = wanted - {scene["name"] for scene in scenes}
        if missing_scenes:
            raise ValueError(f"Dataset is missing scenes in {split_name}: {sorted(missing_scenes)}")
        scenes.sort(key=lambda scene: nusc.get("sample", scene["first_sample_token"])["timestamp"])
        self.scene_names = tuple(scene["name"] for scene in scenes)
        self.records = []
        for scene in scenes:
            token = scene["first_sample_token"]
            while token:
                sample = nusc.get("sample", token)
                label_path = occupancy_label_path(self.occupancy_root, scene["name"], token)
                if not label_path.is_file():
                    raise FileNotFoundError(f"Missing training occupancy labels: {label_path}")
                geometry = camera_geometry(nusc, sample)
                paths, intrinsics, transforms = [], [], []
                for channel in self.camera_order:
                    path = Path(nusc.get_sample_data_path(sample["data"][channel]))
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing camera image: {path}")
                    camera = geometry[channel]
                    paths.append(path)
                    width, height = camera["image_size"]
                    intrinsics.append(resize_intrinsics(camera["intrinsics"], (height, width), self.image_size))
                    transforms.append(camera["ego_to_camera"])
                self.records.append({
                    "sample_token": token, "scene_name": scene["name"],
                    "image_paths": tuple(paths), "label_path": label_path,
                    "intrinsics": np.asarray(intrinsics, dtype=np.float32),
                    "ego_to_camera": np.asarray(transforms, dtype=np.float32),
                })
                token = sample["next"]
        if not self.records:
            raise ValueError(f"No samples in {split_name}.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from .occupancy_labels import load_occupancy_labels
        record = self.records[index]
        occupancy = load_occupancy_labels(record["label_path"])
        labels = occupancy["semantics"]
        if labels.shape != self.grid_shape:
            raise ValueError(f"Expected grid {self.grid_shape}, got {labels.shape} in {record['label_path']}")
        if not np.all(((labels >= 0) & (labels < self.num_classes)) | (labels == self.ignore_index)):
            raise ValueError(f"Unexpected semantic class IDs in {record['label_path']}")
        semantics = torch.from_numpy(labels.astype(np.int64))
        camera_mask = torch.from_numpy(occupancy["mask_camera"])
        return {
            "images": torch.stack([prepare_camera_image(path, self.image_size) for path in record["image_paths"]]),
            "intrinsics": torch.from_numpy(record["intrinsics"].copy()),
            "ego_to_camera": torch.from_numpy(record["ego_to_camera"].copy()),
            "semantics": semantics,
            "mask_camera": camera_mask,
            "valid_mask": camera_mask & (semantics != self.ignore_index),
            "sample_token": record["sample_token"], "scene_name": record["scene_name"],
        }


def _seed_worker(worker_id):
    import random
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_dataloader(dataset, batch_size=2, num_workers=0, shuffle=None, seed=0, pin_memory=None):
    """Create deterministic batches; use a __main__ guard with Windows workers."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative.")
    if shuffle is None:
        shuffle = dataset.split in ("train", "mini_train")
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory,
                      generator=torch.Generator().manual_seed(seed), worker_init_fn=_seed_worker,
                      persistent_workers=False)
