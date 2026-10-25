"""Regression-check direct Occ3D annotations against installed mini metadata."""
import json
from pathlib import Path
import numpy as np
import torch
from data import NuScenesOccupancyDataset
from src.models.occupancy_model import OccupancyModel


def main():
    root=Path(__file__).resolve().parent
    full=root/'data/occ3d_trainval'
    report={'splits':{}}
    for split in ('train','val'):
        old=NuScenesOccupancyDataset(split=split)
        new=NuScenesOccupancyDataset(dataroot=full,version='v1.0-mini',split=split,
            annotations_json=full/'downloads/annotations.json',
            image_root=root/'data/nuscenes/samples',occupancy_root=root/'data/occupancy/gts')
        assert [r['sample_token'] for r in old.records]==[r['sample_token'] for r in new.records]
        t=max(float(np.abs(a['ego_to_camera']-b['ego_to_camera']).max()) for a,b in zip(old.records,new.records))
        k=max(float(np.abs(a['intrinsics']-b['intrinsics']).max()) for a,b in zip(old.records,new.records))
        assert t<1e-5 and k<1e-5
        report['splits'][split]={'frames':len(old),'max_transform_difference':t,'max_intrinsics_difference':k}
        for r in old.records[::max(len(old)//4,1)]:
            target=full/'gts'/r['scene_name']/r['sample_token']/'labels.npz'
            with np.load(r['label_path']) as a, np.load(target) as b:
                for key in ('semantics','mask_camera','mask_lidar'): np.testing.assert_array_equal(a[key],b[key])
        if split=='val':
            # Same scene and image at both roots, now through the full-data path.
            scene=old.records[0]['scene_name']
            expanded=NuScenesOccupancyDataset(dataroot=full,version='v1.0-trainval',split='val',
                annotations_json=full/'downloads/annotations.json',occupancy_root=full/'gts',
                scene_names=[scene])
            a,b=old[0],expanded[0]
            for key in ('images','intrinsics','ego_to_camera','semantics','valid_mask'):
                torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
            checkpoint=torch.load(root/'outputs/baselines/depth_v2_epoch020.pt',map_location='cpu',weights_only=False)
            model=OccupancyModel(**dict(checkpoint['config']['model'],pretrained=False)).cuda().eval()
            model.load_state_dict(checkpoint['model_state_dict'],strict=True)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                x=model(*(a[key][None].cuda() for key in ('images','intrinsics','ego_to_camera')))
                y=model(*(b[key][None].cuda() for key in ('images','intrinsics','ego_to_camera')))
            torch.testing.assert_close(x,y,rtol=0,atol=0)
            report['same_frame_inference_identical']=True
    report['frames_compared']=sum(s['frames'] for s in report['splits'].values())
    (full/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
