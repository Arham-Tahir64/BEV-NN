"""Reuse the project's existing Occ3D files and calibration, without copying data."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Subset


def metadata_from_config(path, project_root, image_size):
    doc=json.loads(Path(path).read_text(encoding='utf-8'))
    metadata=dict(doc['data'])
    for key in ('dataroot','occupancy_root','annotations_json','image_root'):
        if metadata.get(key):
            p=Path(metadata[key]);metadata[key]=str(p if p.is_absolute() else Path(project_root)/p)
    metadata['image_size']=list(image_size)
    if not metadata.get('validation_scenes') or not metadata.get('train_scenes'):
        raise ValueError('Explicit train and validation scene lists are required.')
    if set(metadata['train_scenes'])&set(metadata['validation_scenes']): raise ValueError('Train/validation scene overlap.')
    return metadata


def make_dataset(metadata, split, project_root, *, scenes=None, limit=0, depth=False, depth_cache=None):
    sys.path.insert(0,str(Path(project_root).resolve()))
    from data.nuscenes_dataset import NuScenesOccupancyDataset
    names=metadata['train_scenes' if split=='train' else 'validation_scenes']
    if scenes is not None:
        if not set(scenes)<=set(names): raise ValueError('Requested scenes are outside checkpoint split.')
        names=scenes
    dataset=NuScenesOccupancyDataset(dataroot=metadata['dataroot'],occupancy_root=metadata['occupancy_root'],
        version=metadata.get('version','v1.0-trainval'),image_root=metadata.get('image_root'),
        annotations_json=metadata.get('annotations_json'),image_size=tuple(metadata['image_size']),
        split=split,scene_names=names)
    if tuple(dataset.camera_order)!=tuple(metadata['camera_order']): raise ValueError('Camera order mismatch.')
    if depth:
        from data.depth_targets import DepthTargetDataset
        dataset=DepthTargetDataset(dataset,depth_cache,target_mode='projected')
    return Subset(dataset,range(min(limit,len(dataset)))) if limit else dataset


def training_class_weights(dataset):
    """Only selected training labels; never scan validation to estimate weights."""
    source=dataset.dataset if isinstance(dataset,Subset) else dataset
    indices=dataset.indices if isinstance(dataset,Subset) else range(len(dataset))
    counts=torch.zeros(17,dtype=torch.long)
    for i in indices:
        with np.load(source.records[i]['label_path'],allow_pickle=False) as archive:
            labels=archive['semantics'];valid=archive['mask_camera'].astype(bool)&(labels<17)
            counts+=torch.from_numpy(np.bincount(labels[valid].astype(np.int64),minlength=17))
    present=counts>0;weights=torch.ones(17)
    if not present.any(): return weights,counts
    weights[present]=counts[present].float().rsqrt()
    weights[present]/=weights[present].mean()
    return weights,counts


def training_height_counts(dataset):
    source=dataset.dataset if isinstance(dataset,Subset) else dataset
    indices=dataset.indices if isinstance(dataset,Subset) else range(len(dataset))
    counts=None
    for i in indices:
        with np.load(source.records[i]['label_path'],allow_pickle=False) as archive:
            y=archive['semantics'];known=archive['mask_camera'].astype(bool)&(y!=255)
            row=np.stack([(known&(y<17)).sum((0,1)),(known&(y==17)).sum((0,1))],-1)
            counts=row if counts is None else counts+row
    if counts is None:raise ValueError('Empty training dataset.')
    return torch.from_numpy(counts)
