"""Evaluate a complete held-out split or export one frame with explicit new-model losses."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .data_adapter import make_dataset
from .runtime import build_model,run_epoch,loader


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--split',choices=['train','val'],default='val')
    p.add_argument('--sample-index',type=int)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--threshold',type=float,default=.5)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    a=p.parse_args()
    if not 0<a.threshold<1 or min(a.limit,a.workers)<0:p.error('Invalid threshold/limit/workers.')
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True);config=ck['config'];epoch=ck['epoch']
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device)
    model=build_model(config['model']).to(device);model.load_state_dict(ck['model_state_dict'],strict=True);model.eval();del ck
    depth_weight=config['loss']['depth_weight']
    out=(a.output_dir or a.checkpoint.parent/'evaluation'/a.checkpoint.stem).resolve()
    # Explicit outputs refuse overwrite to preserve comparisons/calibration history.
    if out.exists() and any(out.iterdir()):raise FileExistsError('Evaluation output exists; choose a new directory.')
    out.mkdir(parents=True,exist_ok=True)
    dataset=make_dataset(config['data'],a.split,a.project_root,limit=a.limit,
                         depth=depth_weight>0,depth_cache=out/'depth_cache')
    if a.sample_index is not None:
        if not 0<=a.sample_index<len(dataset):p.error('Sample index out of range.')
        item=dataset[a.sample_index]
        with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            heads=model.forward_heads(*[item[key][None].to(device) for key in ('images','intrinsics','ego_to_camera')])
            pred=model.predict(heads,a.threshold)[0].cpu().numpy()
        np.savez_compressed(out/'predictions.npz',prediction=pred,semantics=item['semantics'].numpy(),valid_mask=item['valid_mask'].numpy())
        batches=loader([item],1,0,False,0,False)
    else:batches=loader(dataset,1,a.workers,False,0,device.type=='cuda')
    stats=run_epoch(model,batches,device,threshold=a.threshold,depth_weight=depth_weight,
                    binary_weight=config['loss']['binary_weight'],semantic_weight=config['loss']['semantic_weight'])
    result=dict(checkpoint=str(a.checkpoint.resolve()),epoch=epoch,split=a.split,frame_only=a.sample_index is not None,
                sample_index=a.sample_index,limited=bool(a.limit),metrics=stats)
    (out/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
