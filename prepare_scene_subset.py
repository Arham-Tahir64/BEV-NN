"""Inventory installed camera keyframes/Occ3D labels and select whole scenes.

No downloads or training. Only complete scenes within each official split are
eligible. Files are checked for presence, not fully decoded or validated.
"""
import argparse
import json
from pathlib import Path
import random

from data.nuscenes_dataset import CAMERAS, DEFAULT_DATAROOT, load_nuscenes
from data.occupancy_labels import DEFAULT_OCCUPANCY_ROOT
from data.scene_selection import split_scene_names


def inventory_scene(nusc, scene, occupancy_root, image_root=None):
    token=scene['first_sample_token']
    result=dict(scene=scene['name'],frames=0,missing_images=0,missing_labels=0,
                image_bytes=0,label_bytes=0,missing_examples=[])
    visited=set()
    def inspect(path, kind):
        if path.is_file():
            result[kind+'_bytes']+=path.stat().st_size
        else:
            result['missing_'+('images' if kind=='image' else 'labels')]+=1
            if len(result['missing_examples'])<3: result['missing_examples'].append(str(path))
    while token:
        if token in visited: raise ValueError(f'Cycle in sample metadata for {scene["name"]}.')
        visited.add(token)
        sample=nusc.get('sample',token)
        result['frames']+=1
        inspect(Path(occupancy_root)/scene['name']/token/'labels.npz','label')
        for channel in CAMERAS:
            path=Path(nusc.get_sample_data_path(sample['data'][channel]))
            if image_root is not None: path=Path(image_root)/channel/path.name
            inspect(path,'image')
        token=sample['next']
    result['complete']=bool(result['frames'] and not result['missing_images'] and not result['missing_labels'])
    return result


def select_complete(rows, count, seed, required=()):
    if count<0: raise ValueError('Scene count must be nonnegative; zero selects all complete scenes.')
    eligible=sorted(r['scene'] for r in rows if r['complete'])
    required=set(required)
    if required-set(eligible): raise ValueError('Required baseline scenes are incomplete locally.')
    if count>len(eligible):
        raise ValueError(f'Requested {count} scenes but only {len(eligible)} are complete locally.')
    if count and count<len(required): raise ValueError('Scene count is smaller than the required baseline scenes.')
    remaining=sorted(set(eligible)-required)
    return sorted(required|set(random.Random(seed).sample(remaining,count-len(required)))) if count else eligible


def compatible_mini_scenes(version,split):
    return sorted(set(split_scene_names('v1.0-mini',split)[1]) & set(split_scene_names(version,split)[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-version',choices=('v1.0-mini','v1.0-trainval'),default='v1.0-trainval')
    parser.add_argument('--dataroot',type=Path,default=DEFAULT_DATAROOT)
    parser.add_argument('--occupancy-root',type=Path,default=DEFAULT_OCCUPANCY_ROOT)
    parser.add_argument('--image-root',type=Path)
    parser.add_argument('--annotations-json',type=Path)
    parser.add_argument('--include-mini-scenes',action='store_true',
                        help='Retain mini scenes whose roles agree with the official selected split.')
    parser.add_argument('--train-count',type=int,default=100)
    parser.add_argument('--val-count',type=int,default=20)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--inventory-only',action='store_true')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if min(args.train_count,args.val_count,args.seed)<0: parser.error('Counts and seed must be nonnegative.')
    dest=args.output_dir.resolve()
    if any((dest/f).exists() for f in ('inventory.json','train_scenes.json','val_scenes.json')):
        parser.error('Output files already exist; use a fresh directory.')
    dest.mkdir(parents=True,exist_ok=True)
    report={'version':args.dataset_version,'dataroot':str(args.dataroot.resolve()),
        'occupancy_root':str(args.occupancy_root.resolve()),
        'image_root':str(args.image_root.resolve()) if args.image_root is not None else None,
        'annotations_json':str(args.annotations_json.resolve()) if args.annotations_json is not None else None,
        'include_mini_scenes':args.include_mini_scenes,
        'mini_scene_roles_excluded':{split:sorted(set(split_scene_names('v1.0-mini',split)[1])-
            set(compatible_mini_scenes(args.dataset_version,split))) for split in ('train','val')},
        'seed':args.seed,'check':'File presence only; labels/images are validated when loaded.'}
    def save(): (dest/'inventory.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    try: nusc=load_nuscenes(args.dataroot,args.dataset_version,verbose=False,
                          annotations_json=args.annotations_json,image_root=args.image_root)
    except FileNotFoundError as error:
        report.update(status='missing_metadata',error=str(error)); save()
        print(str(error)); return 2
    report['splits']={}
    for split in ('train','val'):
        _,official=split_scene_names(args.dataset_version,split)
        wanted=set(official)
        scenes=sorted((s for s in nusc.scene if s['name'] in wanted),key=lambda s:s['name'])
        rows=[inventory_scene(nusc,s,args.occupancy_root,args.image_root) for s in scenes]
        report['splits'][split]={'official_scenes':len(official),'metadata_scenes':len(scenes),
            'complete_scenes':sum(r['complete'] for r in rows),'scenes':rows}
        print(f'{split}: {sum(r["complete"] for r in rows)}/{len(official)} complete scenes',flush=True)
    report['status']='inventory_complete'; save()
    if args.inventory_only: return 0
    try:
        selections={split:select_complete(report['splits'][split]['scenes'],count,args.seed,
                        compatible_mini_scenes(args.dataset_version,split) if args.include_mini_scenes else ())
                    for split,count in [('train',args.train_count),('val',args.val_count)]}
        if not all(selections.values()): raise ValueError('At least one complete scene is needed in each split.')
    except ValueError as error:
        report.update(status='insufficient_complete_scenes',error=str(error)); save()
        print(str(error)); return 2
    assert set(selections['train']).isdisjoint(selections['val'])
    for split,names in selections.items():
        (dest/f'{split}_scenes.json').write_text(json.dumps(names,indent=2),encoding='utf-8')
        report['splits'][split]['selected_scenes']=names
        report['splits'][split]['selected_frames']=sum(r['frames'] for r in report['splits'][split]['scenes'] if r['scene'] in names)
    report['status']='subset_prepared'; save()
    print(f'Whole-scene manifests saved in {dest}')
    return 0


if __name__=='__main__': raise SystemExit(main())
