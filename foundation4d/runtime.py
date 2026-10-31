"""Training/evaluation machinery with explicit masks and comparable global metrics."""
import hashlib
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .model import Foundation4DLite
from .losses import occupancy_losses,balanced_binary,boundary_binary_loss,observed_ray_loss
from .metrics import summarize_confusion, occupancy_boundary_counts, summarize_boundary_counts


def seed_all(seed):
    random.seed(seed);np.random.seed(seed % 2**32);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def loader(dataset, batch_size, workers, shuffle, seed, cuda):
    return DataLoader(dataset,batch_size=batch_size,num_workers=workers,shuffle=shuffle,
                      pin_memory=cuda,generator=torch.Generator().manual_seed(seed))


def dataset_signature(dataset):
    source=dataset.dataset if isinstance(dataset,Subset) else dataset
    ids=dataset.indices if isinstance(dataset,Subset) else range(len(dataset))
    tokens=[source.records[i]['sample_token'] for i in ids]
    return hashlib.sha256('\n'.join(tokens).encode()).hexdigest()


def build_model(config):
    settings=dict(config)
    if settings.pop('architecture',None)!='foundation4d_lite': raise ValueError('Not a Foundation4D-Lite checkpoint.')
    settings['pretrained']=False
    return Foundation4DLite(**settings)


def rng_state():
    state=np.random.get_state()
    return {'python':random.getstate(),'numpy':(state[0],state[1].tolist(),state[2],state[3],state[4]),
            'torch':torch.random.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']);s=state['numpy'];np.random.set_state((s[0],np.array(s[1],dtype=np.uint32),s[2],s[3],s[4]))
    torch.random.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])


def atomic_save(path, payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,temporary);temporary.replace(path)


def run_epoch(model, batches, device, *, optimizer=None, scaler=None, amp=True,
              class_weights=None, threshold=.5, binary_weight=1., semantic_weight=1.,
              depth_weight=0., max_batches=0, binary_height_weights=None,boundary_weight=0.,ray_weight=0.):
    training=optimizer is not None;model.train(training)
    use_amp=amp and device.type=='cuda'
    cm=torch.zeros(18,18,dtype=torch.long,device=device)
    boundaries=torch.zeros(3,dtype=torch.long,device=device)
    names=('0_10','10_20','20plus','far_front','far_rear','below1m','above1m','above2m')
    region_cm={name:torch.zeros_like(cm) for name in names}
    sums={'binary':0.,'semantic':0.,'depth':0.};counts={'binary':0,'semantic':0.,'depth':0}
    depth_count=[0,0,0];depth_abs=[0.,0.,0.];depth_mass=[0.,0.,0.]
    processed=skipped=updates=0;start=time.perf_counter()
    auxiliary={'boundary':0.,'ray':0.}
    regions=None
    with torch.set_grad_enabled(training):
        for batch_index,batch in enumerate(batches,1):
            if max_batches and batch_index>max_batches: break
            labels=batch['semantics'].to(device,non_blocking=True);valid=batch['valid_mask'].to(device,non_blocking=True)
            known=valid&(labels!=255)
            if not known.any(): skipped+=1;continue
            inputs=[batch[k].to(device,non_blocking=True) for k in ('images','intrinsics','ego_to_camera')]
            if training: optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=use_amp):
                heads=model.forward_heads(*inputs)
            terms=occupancy_losses(heads,labels,valid,class_weights)
            if training and binary_height_weights is not None:
                terms['binary']=balanced_binary(heads['binary_logits'],labels,valid,*binary_height_weights)
            objective=binary_weight*terms['binary']+semantic_weight*terms['semantic']
            if training and boundary_weight:
                bl=boundary_binary_loss(heads['binary_logits'],labels,valid)
                objective=objective+boundary_weight*bl;auxiliary['boundary']+=float(bl.detach())
            if training and ray_weight:
                rl=observed_ray_loss(heads['binary_logits'],labels,valid,inputs[1],inputs[2],model.voxel_bounds,inputs[0].shape[-2:])
                objective=objective+ray_weight*rl;auxiliary['ray']+=float(rl.detach())
            depth_loss_value=None;target=None
            if depth_weight>0:
                if heads['depth_logits'] is None or 'depth_target' not in batch: raise ValueError('Depth loss requires a depth head and targets.')
                from data.depth_targets import depth_loss
                target=batch['depth_target'].to(device)
                depth_loss_value,n=depth_loss(heads['depth_logits'],target,model.depth_edges,
                    mode='linear' if getattr(model,'geometry_aware',False) else 'bce',
                    range_balance=getattr(model,'geometry_aware',False))
                objective=objective+depth_weight*depth_loss_value
                sums['depth']+=float(depth_loss_value.detach())*n;counts['depth']+=n
            if not torch.isfinite(objective): raise FloatingPointError('Nonfinite occupancy objective.')
            if training:
                if scaler is None: raise ValueError('Training needs an AMP scaler (disabled is valid).')
                old_scale=scaler.get_scale()
                scaler.scale(objective).backward()
                scaler.step(optimizer);scaler.update()
                updates+=int(scaler.get_scale()>=old_scale)
            sums['binary']+=float(terms['binary'].detach())*terms['known'];counts['binary']+=terms['known']
            sums['semantic']+=float(terms['semantic'].detach())*terms['semantic_denominator'];counts['semantic']+=terms['semantic_denominator']
            processed+=1
            if not training:
                pred=model.predict(heads,threshold)
                cm+=torch.bincount(labels[known]*18+pred[known],minlength=324).reshape(18,18)
                boundaries+=occupancy_boundary_counts(pred,labels,valid)
                if regions is None:
                    shape=labels.shape[1:]
                    # Dataset metrics use exact output-grid ego-frame cell centres.
                    low=model.voxel_bounds[:3];high=model.voxel_bounds[3:]
                    x=(torch.arange(shape[0],device=device)+.5)*(high[0]-low[0])/shape[0]+low[0]
                    y=(torch.arange(shape[1],device=device)+.5)*(high[1]-low[1])/shape[1]+low[1]
                    z=(torch.arange(shape[2],device=device)+.5)*(high[2]-low[2])/shape[2]+low[2]
                    radius=(x[:,None]**2+y[None,:]**2).sqrt()[:,:,None]
                    regions=dict(zip(names,(radius<10,(radius>=10)&(radius<20),radius>=20,
                                  (radius>=20)&(x[:,None,None]>=0),(radius>=20)&(x[:,None,None]<0),
                                  z[None,None,:]<1,z[None,None,:]>=1,z[None,None,:]>=2)))
                for name,mask in regions.items():
                    keep=known&mask
                    region_cm[name]+=torch.bincount(labels[keep]*18+pred[keep],minlength=324).reshape(18,18)
                if target is not None:
                    p=heads['depth_logits'].float().softmax(2);centres=(model.depth_edges[:-1]+model.depth_edges[1:])/2
                    expected=(p*centres[None,None,:,None,None]).sum(2)
                    mass=(p*((centres[None,None,:,None,None]-target[:,:,None]).abs()<=2)).sum(2)
                    for i,(lo,hi) in enumerate(((0,10),(10,20),(20,float('inf')))):
                        keep=torch.isfinite(target)&(target>=lo)&(target<hi)
                        depth_count[i]+=int(keep.sum());depth_abs[i]+=float((expected-target).abs()[keep].sum());depth_mass[i]+=float(mass[keep].sum())
            if batch_index==1 or batch_index%50==0:
                print(f'{"Train" if training else "Validation"} batch {batch_index}/{len(batches)}: binary={sums["binary"]/counts["binary"]:.4f}, semantic={sums["semantic"]/max(counts["semantic"],1e-12):.4f}',flush=True)
    if not processed: raise RuntimeError('No supervised batches in epoch.')
    if training and not updates: raise RuntimeError('No optimizer updates completed; inspect AMP scale/gradients.')
    if depth_weight>0 and not counts['depth']: raise RuntimeError('No depth targets in epoch.')
    result={f'{key}_loss':sums[key]/counts[key] if counts[key] else 0. for key in sums}
    result.update({f'{k}_loss':v/processed for k,v in auxiliary.items()})
    result.update(loss=binary_weight*result['binary_loss']+semantic_weight*result['semantic_loss']+depth_weight*result['depth_loss'],
                  batches=processed,skipped_batches=skipped,optimizer_updates=updates if training else None,
                  valid_voxels=counts['binary'],occupied_weight_denominator=counts['semantic'],seconds=time.perf_counter()-start)
    result['training_objective']=result['loss']+boundary_weight*result['boundary_loss']+ray_weight*result['ray_loss']
    if not training:
        result.update(summarize_confusion(cm));result.update(summarize_boundary_counts(boundaries))
        result['accuracy']=float(cm.diag().sum()/cm.sum())
        result['range_metrics']={name:summarize_confusion(matrix) for name,matrix in region_cm.items()}
        result['threshold']=threshold
        if depth_weight>0:
            result['depth_range_expected_mae_m']=[v/n if n else None for v,n in zip(depth_abs,depth_count)]
            result['depth_range_mass_within2m']=[v/n if n else None for v,n in zip(depth_mass,depth_count)]
    return result
