"""Continue an interrupted run using its saved configuration and optimizer.

--epochs is the TOTAL target epoch, not an additional count. Old checkpoints
did not save RNG state, so this restores training progress but not the exact
minibatch sequence an uninterrupted run would have used.
"""
import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import Subset

from data.nuscenes_dataset import NuScenesOccupancyDataset, make_dataloader
from data.depth_targets import DepthTargetDataset
from src.models.occupancy_model import OccupancyModel
from train import train_one_epoch, validate, save_checkpoint


def restore_training(checkpoint, model, optimizer, scaler):
    """Strict architecture restoration; optimizer restores the saved learning rate."""
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    return checkpoint['epoch'] + 1, checkpoint['config']['metrics']['best_loss']


def check_log(path, epoch):
    """Refuse ambiguous recovery rather than erase or duplicate logged epochs."""
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if [row['epoch'] for row in rows] != list(range(1, epoch+1)):
        raise ValueError('Checkpoint and metrics log disagree. Preserve the files and reconcile before resuming.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()
    if args.epochs <= 0 or args.workers < 0:
        parser.error('Epochs must be positive and workers nonnegative.')
    checkpoint_path = args.checkpoint.resolve()
    if checkpoint_path.name != 'latest.pt':
        parser.error('Use latest.pt to continue the existing run without branching its history.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    settings, metadata = config['training'], config['data']
    output = checkpoint_path.parent
    check_log(output / 'metrics.jsonl', checkpoint['epoch'])
    if args.epochs <= checkpoint['epoch']:
        print(f"Already completed epoch {checkpoint['epoch']}; no files changed.")
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable.')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    amp = settings.get('amp', True) and device.type == 'cuda'
    seed = settings.get('seed', 0) + checkpoint['epoch']
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)

    train = NuScenesOccupancyDataset(dataroot=metadata['dataroot'],
        occupancy_root=metadata['occupancy_root'], image_size=tuple(metadata['image_size']), split='train',
        version=metadata.get('version','v1.0-mini'), image_root=metadata.get('image_root'),
        scene_names=metadata['train_scenes'], annotations_json=metadata.get('annotations_json'))
    for key, actual in (('grid_shape', train.grid_shape), ('voxel_bounds', train.voxel_bounds)):
        if tuple(config['model'][key]) != tuple(actual):
            raise ValueError(f'Checkpoint {key} does not match dataset.')
    if tuple(metadata['camera_order']) != tuple(train.camera_order):
        raise ValueError('Camera order changed.')
    selected_count = settings.get('overfit_samples', 0) or len(train)
    tokens = [r['sample_token'] for r in train.records[:selected_count]]
    if tokens != metadata['train_sample_tokens']:
        raise ValueError('Training samples differ from the checkpoint.')
    if config['model'].get('use_depth', False):
        train = DepthTargetDataset(train, settings.get('depth_cache', output.parent / 'depth_cache'),
            config['model']['depth_min'], config['model']['depth_max'],
            target_mode=settings.get('depth_target_mode', 'ray'))
    selected = Subset(train, range(selected_count))
    loader = make_dataloader(selected, batch_size=settings['batch_size'], num_workers=args.workers,
                             shuffle=True, seed=seed, pin_memory=device.type == 'cuda')
    val_loader = None
    if config['selection_metric'] == 'validation_loss':
        val = NuScenesOccupancyDataset(dataroot=metadata['dataroot'],
            occupancy_root=metadata['occupancy_root'], image_size=tuple(metadata['image_size']), split='val',
            version=metadata.get('version','v1.0-mini'), image_root=metadata.get('image_root'),
            scene_names=metadata['validation_scenes'], annotations_json=metadata.get('annotations_json'))
        if tuple(val.scene_names) != tuple(metadata['validation_scenes']) or not set(train.scene_names).isdisjoint(val.scene_names):
            raise ValueError('Validation split changed or overlaps training.')
        if config['model'].get('use_depth', False):
            val = DepthTargetDataset(val, settings.get('depth_cache', output.parent / 'depth_cache'),
                config['model']['depth_min'], config['model']['depth_max'],
                target_mode=settings.get('depth_target_mode', 'ray'))
        val_loader = make_dataloader(val, batch_size=settings['batch_size'], num_workers=args.workers,
                                     shuffle=False, pin_memory=device.type == 'cuda')

    model = OccupancyModel(**{**config['model'], 'pretrained': False}).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=settings['lr'], weight_decay=settings['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    # A run originally on CPU can be resumed on CUDA without a saved scaler.
    saved_scaler = checkpoint['scaler_state_dict']
    if amp and not saved_scaler:
        checkpoint = {**checkpoint, 'scaler_state_dict': scaler.state_dict()}
    start_epoch, best_loss = restore_training(checkpoint, model, optimizer, scaler)
    best_semantic_miou = config['metrics'].get('best_semantic_miou', -1.0)
    weights = config.get('loss', {}).get('class_weights')
    weights = torch.tensor(weights, dtype=torch.float32, device=device) if weights is not None else None
    config = {**config, 'training': {**settings, 'epochs': args.epochs, 'workers': args.workers},
              'device': str(device), 'amp_enabled': amp}
    print(f'Resuming epoch {start_epoch}/{args.epochs} on {device}; best loss={best_loss:.6f}', flush=True)
    print('Legacy checkpoint: random-number state was not saved; minibatch order may differ.', flush=True)
    with (output / 'metrics.jsonl').open('a', encoding='utf-8') as log:
        for epoch in range(start_epoch, args.epochs+1):
            start = time.perf_counter()
            stats = train_one_epoch(model, loader, optimizer, scaler, device, amp, weights,
                                    depth_loss_weight=settings.get('depth_loss_weight', 1.0),
                                    depth_loss_mode=settings.get('depth_loss_mode', 'hard'),
                                    geometry_loss_weight=settings.get('geometry_loss_weight', 0.0),
                                    boundary_loss_weight=settings.get('boundary_loss_weight', 0.0))
            val_stats = validate(model, val_loader, device, amp,
                depth_loss_mode=settings.get('depth_loss_mode', 'hard')) if val_loader is not None else None
            loss = val_stats['loss'] if val_stats is not None else stats['loss']
            if not math.isfinite(loss):
                raise FloatingPointError('Non-finite epoch loss.')
            improved = loss < best_loss
            best_loss = min(best_loss, loss)
            semantic_improved = val_stats is not None and val_stats['occupied_class_miou'] > best_semantic_miou
            if semantic_improved:
                best_semantic_miou = val_stats['occupied_class_miou']
            record = dict(epoch=epoch, train=stats, validation=val_stats,
                seconds=time.perf_counter()-start, selection_metric=config['selection_metric'], best_loss=best_loss,
                best_semantic_miou=best_semantic_miou)
            state_config = {**config, 'metrics': record}
            # Match the original run's log/checkpoint order. A crash between
            # these operations is detected by check_log on the next resume.
            log.write(json.dumps(record, allow_nan=False)+'\n')
            log.flush()
            save_checkpoint(output / 'latest.pt', model, optimizer, scaler, epoch, state_config)
            if improved:
                save_checkpoint(output / 'best.pt', model, optimizer, scaler, epoch, state_config)
            if semantic_improved:
                save_checkpoint(output / 'best_semantic.pt', model, optimizer, scaler, epoch, state_config)
            print(f'Epoch {epoch}/{args.epochs}: selected loss={loss:.6f}', flush=True)
            if val_stats is not None:
                print(f"Occupied mIoU={val_stats['occupied_class_miou']:.3%}; "
                      f"manmade IoU={val_stats['manmade_iou']:.3%}; "
                      f"missing class IDs={val_stats['missing_supported_classes']}", flush=True)
                print(f"Occupied precision/recall={val_stats['occupied_precision']:.3%}/"
                      f"{val_stats['occupied_recall']:.3%}; boundary F1={val_stats.get('boundary_f1')}", flush=True)


if __name__ == '__main__':
    main()
