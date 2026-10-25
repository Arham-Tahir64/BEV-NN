"""Compare legacy and projected pseudo-depth coverage, with camera overlays."""
import argparse
import json
from pathlib import Path
import numpy as np
from data.nuscenes_dataset import NuScenesOccupancyDataset
from data.depth_targets import first_surface_depth, projected_surface_depth


def preview(record, depth, image_size, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from data.nuscenes_dataset import CAMERAS
    fig, axes = plt.subplots(2, 3, figsize=(15, 6), constrained_layout=True)
    for camera, path, values, ax in zip(CAMERAS, record['image_paths'], depth, axes.flat):
        with Image.open(path) as image:
            ax.imshow(image.convert('RGB').resize((image_size[1], image_size[0])))
        yy, xx = np.nonzero(np.isfinite(values))
        points = ax.scatter(xx*16, yy*16, c=values[yy, xx], cmap='turbo', vmin=0, vmax=64,
                            s=9, alpha=.75)
        ax.set_title(f'{camera}: {len(xx)} targets')
        ax.axis('off')
    fig.colorbar(points, ax=axes.ravel().tolist(), label='Approximate camera Z (m)', shrink=.8)
    fig.suptitle(record['scene_name']+' — nearest annotated occupied centre per feature cell')
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/depth_target_audit_v2'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for split in ('train', 'val'):
        ds = NuScenesOccupancyDataset(split=split)
        rows = []
        previews = set()
        for i, record in enumerate(ds.records):
            with np.load(record['label_path'], allow_pickle=False) as a:
                labels, observed, visible = a['semantics'], a['mask_lidar'].astype(bool), a['mask_camera'].astype(bool)
            valid = visible & (labels != 255)
            params = (record['intrinsics'], record['ego_to_camera'], ds.image_size, ds.voxel_bounds)
            old = first_surface_depth(labels, valid, *params)
            new = projected_surface_depth(labels, observed, valid, *params)
            row = {'scene':record['scene_name'], 'sample_token':record['sample_token']}
            for name, depth in (('legacy', old), ('projected', new)):
                row[name] = {'count':int(np.isfinite(depth).sum()), 'slots':int(depth.size),
                             'near10':int((depth < 10).sum()), 'far20':int((depth >= 20).sum()),
                             'camera_counts':np.isfinite(depth).sum((1, 2)).tolist()}
            rows.append(row)
            if record['scene_name'] in ('scene-0061','scene-0553','scene-0103') and record['scene_name'] not in previews:
                preview(record, new, ds.image_size, args.output_dir / f"{record['scene_name']}_targets.png")
                previews.add(record['scene_name'])
            if (i+1)%80 == 0:
                print(f'{split}: {i+1}/{len(ds)}', flush=True)
        summary = {}
        for mode in ('legacy','projected'):
            total = sum(r[mode]['count'] for r in rows)
            summary[mode] = {'frames':len(rows), 'targets':total,
                'coverage':total/sum(r[mode]['slots'] for r in rows),
                'zero_frames':sum(r[mode]['count']==0 for r in rows),
                'near10_fraction':sum(r[mode]['near10'] for r in rows)/max(total,1),
                'far20_fraction':sum(r[mode]['far20'] for r in rows)/max(total,1),
                'scene_coverage':{s:float(np.mean([r[mode]['count']/r[mode]['slots'] for r in rows if r['scene']==s])) for s in ds.scene_names}}
        report[split] = summary
        (args.output_dir / f'{split}_frames.json').write_text(json.dumps(rows, indent=2))
        print(json.dumps({split:summary}), flush=True)
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
