"""Explicit scene subsets constrained to the official nuScenes splits."""
import json
from pathlib import Path


def validate_scene_list(names):
    if (not isinstance(names, (list, tuple)) or not names
            or any(not isinstance(n, str) or not n.strip() for n in names)):
        raise ValueError('Scene selection must be a nonempty list of scene names.')
    if len(set(names)) != len(names):
        raise ValueError('Scene selection contains duplicate names.')
    return tuple(names)


def read_scene_list(path):
    return None if path is None else validate_scene_list(json.loads(Path(path).read_text(encoding='utf-8')))


def split_scene_names(version, split, scene_names=None):
    from nuscenes.utils.splits import create_splits_scenes
    if version not in ('v1.0-mini', 'v1.0-trainval'):
        raise ValueError('Training requires v1.0-mini or v1.0-trainval.')
    valid = ('train', 'val', 'mini_train', 'mini_val') if version=='v1.0-mini' else ('train','val')
    if split not in valid:
        raise ValueError(f'Choose one of {valid} for {version}.')
    canonical = f'mini_{split}' if version=='v1.0-mini' and split in ('train','val') else split
    official = tuple(create_splits_scenes()[canonical])
    if scene_names is None:
        return canonical, official
    requested = validate_scene_list(scene_names)
    invalid = set(requested)-set(official)
    if invalid:
        raise ValueError(f'Scenes outside official {canonical} split: {sorted(invalid)}')
    return canonical, requested
