"""Real-frame overfit and warm training profile; no changes to prior runs/viewers."""
import argparse
import json
import time
from pathlib import Path
import torch
from foundation4d.data_adapter import metadata_from_config,make_dataset
from foundation4d.model import Foundation4DLite
from foundation4d.losses import occupancy_losses
from foundation4d.runtime import seed_all


def main():
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]
    p.add_argument('--project-root',type=Path,default=root.parent)
    p.add_argument('--data-config',type=Path)
    p.add_argument('--image-size',nargs=2,type=int,default=[320,576])
    p.add_argument('--steps',type=int,default=24)
    p.add_argument('--output-dir',type=Path,default=root/'outputs/profile_320')
    p.add_argument('--learned-offsets',action='store_true')
    a=p.parse_args()
    if a.steps<24:p.error('Use at least four warmup + twenty measured steps.')
    if not torch.cuda.is_available():raise RuntimeError('CUDA profiling requires the target GPU.')
    if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError('Profile output exists.')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);seed_all(0)
    data_path=a.data_config or a.project_root/'outputs/train_bevdepth_boundary_hires_train100/config.json'
    md=metadata_from_config(data_path,a.project_root,a.image_size)
    dataset=make_dataset(md,'train',a.project_root,scenes=md['train_scenes'][:1],limit=1)
    start=time.perf_counter();item=dataset[0];data_seconds=time.perf_counter()-start
    inputs=[item[k][None].cuda() for k in ('images','intrinsics','ego_to_camera')]
    labels=item['semantics'][None].cuda();valid=item['valid_mask'][None].cuda()
    model=Foundation4DLite(pretrained=True,learned_offsets=a.learned_offsets).cuda().train()
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    records=[];steps=0;gradients={}
    for i in range(a.steps):
        if i==4:torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.float16):heads=model.forward_heads(*inputs)
        torch.cuda.synchronize();forward=time.perf_counter()-start
        terms=occupancy_losses(heads,labels,valid);loss=terms['binary']+terms['semantic']
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite smoke loss.')
        start=time.perf_counter();scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.cuda.synchronize()
        backward=time.perf_counter()-start
        for name,module in [('encoder',model.encoder),('projector',model.projectors),('binary',model.binary_head),('semantic',model.semantic_head)]:
            grads=[v.grad for v in module.parameters() if v.grad is not None]
            if not grads or not all(torch.isfinite(v).all() for v in grads):raise FloatingPointError(f'Invalid {name} gradients.')
            gradients[name]=sum(v.abs().sum().item() for v in grads)
            if not gradients[name]>0:raise RuntimeError(f'No gradient to {name}.')
        start=time.perf_counter();old=scaler.get_scale();scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
        step=time.perf_counter()-start;steps+=int(scaler.get_scale()>=old)
        records.append(dict(loss=float(loss.detach()),binary=float(terms['binary'].detach()),semantic=float(terms['semantic'].detach()),
                            forward_seconds=forward,backward_seconds=backward,optimizer_seconds=step))
        print(f'{i+1}/{a.steps}: loss={records[-1]["loss"]:.4f}',flush=True)
    measured=records[4:]
    model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):heads=model.forward_heads(*inputs)
    prediction=model.predict(heads)[0].cpu().numpy()
    import numpy as np
    from foundation4d.metrics import summarize_confusion
    known=item['valid_mask'].numpy()&(item['semantics'].numpy()!=255)
    cm=torch.bincount(item['semantics'][torch.from_numpy(known)]*18+torch.from_numpy(prediction[known]),minlength=324).reshape(18,18)
    np.savez_compressed(a.output_dir/'predictions.npz',prediction=prediction,semantics=item['semantics'].numpy(),valid_mask=item['valid_mask'].numpy())
    metrics=dict(gpu=torch.cuda.get_device_name(),image_size=a.image_size,parameters=sum(v.numel() for v in model.parameters()),
        sample_token=item['sample_token'],warmup_steps=4,measured_steps=len(measured),optimizer_updates=steps,
        initial_loss=records[0]['loss'],final_loss=records[-1]['loss'],gradient_l1=gradients,
        first_frame_preprocessing_seconds=data_seconds,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        timing_scope='CUDA-synchronized model/loss backward/optimizer; gradient inspection and data loading excluded',
        means={key:sum(r[key] for r in measured)/len(measured) for key in ('forward_seconds','backward_seconds','optimizer_seconds')})
    metrics['single_training_frame_fit']=summarize_confusion(cm)
    if steps!=a.steps:raise RuntimeError('Some optimizer steps overflowed; inspect the run before accepting profile.')
    if metrics['final_loss']>=metrics['initial_loss']:raise RuntimeError('Single-frame loss did not decrease.')
    (a.output_dir/'profile.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))
    if metrics['peak_allocated_gib']>6 or metrics['peak_reserved_gib']>7:raise RuntimeError('4060 memory budget exceeded.')


if __name__=='__main__':main()
