"""Train or resume Foundation4D-Lite. Run: python -m foundation4d.train --help."""
import argparse
import json
import math
import shutil
from pathlib import Path
import torch
from .model import Foundation4DLite
from .data_adapter import metadata_from_config,make_dataset,training_class_weights
from .runtime import seed_all,loader,run_epoch,build_model,atomic_save,rng_state,restore_rng,dataset_signature


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]
    p.add_argument('--project-root',type=Path,default=root)
    p.add_argument('--data-config',type=Path)
    p.add_argument('--output-dir',type=Path,default=root/'foundation4d/outputs/lite_320_seed0')
    p.add_argument('--resume',type=Path)
    p.add_argument('--epochs',type=int,default=8,help='Total target epoch, including completed epochs.')
    p.add_argument('--image-size',type=int,nargs=2,default=[320,576])
    p.add_argument('--bev-shape',type=int,nargs=2,default=[100,100])
    p.add_argument('--channels',type=int,default=64)
    p.add_argument('--samples',type=int,choices=[1,4],default=4)
    p.add_argument('--layers',type=int,default=2)
    p.add_argument('--query-chunk',type=int,default=None,
                   help='Sampling execution chunk (fresh default 1024); may be overridden on resume without changing weights.')
    p.add_argument('--learned-offsets',action='store_true')
    p.add_argument('--depth-weight',type=float,default=0.)
    p.add_argument('--binary-weight',type=float,default=1.)
    p.add_argument('--semantic-weight',type=float,default=1.)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--weight-decay',type=float,default=.01)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--pretrained',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--class-balance',choices=['sqrt-inverse','none'],default='sqrt-inverse')
    p.add_argument('--overfit-samples',type=int,default=0)
    p.add_argument('--val-samples',type=int,default=0,help='0 evaluates full validation; bounded runs marked in metadata.')
    p.add_argument('--max-train-batches',type=int,default=0)
    p.add_argument('--allocated-limit-gib',type=float,default=6.)
    p.add_argument('--reserved-limit-gib',type=float,default=7.)
    a=p.parse_args()
    for key in ('epochs','channels','layers','batch_size'):
        if getattr(a,key)<1:p.error(f'{key} must be positive.')
    if a.query_chunk is not None and a.query_chunk<1:p.error('query_chunk must be positive.')
    if any(v<1 for v in (*a.image_size,*a.bev_shape)):p.error('Image/BEV dimensions must be positive.')
    for key in ('workers','seed','overfit_samples','val_samples','max_train_batches'):
        if getattr(a,key)<0:p.error(f'{key} must be nonnegative.')
    for key in ('depth_weight','binary_weight','semantic_weight','lr','weight_decay','allocated_limit_gib','reserved_limit_gib'):
        if not math.isfinite(getattr(a,key)) or getattr(a,key)<0:p.error(f'{key} must be finite nonnegative.')
    if a.lr<=0 or a.binary_weight+a.semantic_weight<=0:p.error('Require positive LR and an occupancy objective.')
    return a


def main():
    requested=parse_args();checkpoint=None
    if requested.resume:
        path=requested.resume.expanduser().resolve()
        if path.name!='latest.pt':raise ValueError('Resume latest.pt; use a new run for an experiment branch.')
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        a=argparse.Namespace(**checkpoint['run_args'])
        # Resume restores the architecture/objective/data selection. Only these runtime overrides apply.
        a.epochs=requested.epochs;a.workers=requested.workers;a.device=requested.device;a.output_dir=str(path.parent)
        config=checkpoint['config']
        if requested.query_chunk is not None:
            a.query_chunk=requested.query_chunk
            config['model']['query_chunk']=requested.query_chunk
    else:
        a=requested;config=None
        if a.query_chunk is None:a.query_chunk=1024
    root=Path(a.project_root).expanduser().resolve();out=Path(a.output_dir).expanduser().resolve()
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable.')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device)
    seed_all(a.seed)
    if checkpoint is None:
        if out.exists() and any(out.iterdir()):raise FileExistsError('Output is nonempty; choose a new directory or resume latest.pt.')
        data_path=Path(a.data_config) if a.data_config else root/'outputs/train_bevdepth_boundary_hires_train100/config.json'
        metadata=metadata_from_config(data_path,root,a.image_size)
        model_config=dict(architecture='foundation4d_lite',channels=a.channels,bev_shape=a.bev_shape,
            grid_shape=[200,200,16],voxel_bounds=[-40.,-40.,-1.,40.,40.,5.4],samples=a.samples,
            query_chunk=a.query_chunk,layers=a.layers,learned_offsets=a.learned_offsets,use_depth=a.depth_weight>0,
            threshold=.5,pretrained=a.pretrained)
        config={'model':model_config,'data':metadata,'scope':'bounded_debug' if a.overfit_samples or a.val_samples or a.max_train_batches else 'full_train100_val20',
                'loss':{'binary_weight':a.binary_weight,'semantic_weight':a.semantic_weight,'depth_weight':a.depth_weight,
                        'semantic_rule':'Known occupied voxels only; free=17, ignore=255.', 'validation':'Unweighted BCE plus occupied-only semantic CE; threshold .5.'}}
    else:
        metadata=config['data'];model_config=config['model']
        if a.epochs<=checkpoint['epoch']:print('Target epoch already completed; no changes.');return
        rows=[json.loads(s) for s in (out/'metrics.jsonl').read_text().splitlines() if s.strip()]
        if [r['epoch'] for r in rows]!=list(range(1,checkpoint['epoch']+1)):
            raise ValueError('Log/checkpoint mismatch; preserve files and reconcile before resuming.')
    if shutil.disk_usage(out if out.exists() else root).free<5*2**30:
        raise RuntimeError('Less than 5 GiB disk headroom; every epoch retains optimizer state.')
    out.mkdir(parents=True,exist_ok=True)
    train=make_dataset(metadata,'train',root,limit=a.overfit_samples,depth=a.depth_weight>0,depth_cache=out/'depth_cache')
    val=make_dataset(metadata,'val',root,limit=a.val_samples,depth=a.depth_weight>0,depth_cache=out/'depth_cache')
    signatures={key:dataset_signature(ds) for key,ds in [('train',train),('val',val)]}
    if checkpoint and signatures!=config['data_signatures']:raise ValueError('Dataset record selection changed.')
    config['data_signatures']=signatures
    config['data']['selected_train_samples']=len(train);config['data']['selected_val_samples']=len(val)
    # Depth caches and labels are supervision; never passed into model inference.
    model=build_model(model_config) if checkpoint else Foundation4DLite(**{k:v for k,v in model_config.items() if k!='architecture'})
    model.to(device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr,weight_decay=a.weight_decay)
    scaler=torch.amp.GradScaler('cuda',enabled=a.amp and device.type=='cuda',init_scale=1024)
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'],strict=True);optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        saved=checkpoint['scaler_state_dict']
        if saved and scaler.is_enabled():scaler.load_state_dict(saved)
        restore_rng(checkpoint['rng_state'])
        weights=torch.tensor(config['loss']['class_weights']) if config['loss']['class_weights'] else None
        best=checkpoint['best'];start=checkpoint['epoch']+1
    else:
        weights,counts=training_class_weights(train) if a.class_balance=='sqrt-inverse' else (None,None)
        config['loss']['class_weights']=weights.tolist() if weights is not None else None
        config['loss']['class_counts']=counts.tolist() if counts is not None else None
        best={'loss':float('inf'),'semantic':-1.,'iou':-1.,'boundary':-1.};start=1
    if weights is not None:weights=weights.to(device)
    args_dict={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    (out/'config.json').write_text(json.dumps({**config,'training':args_dict},indent=2))
    print(f'Foundation4D-Lite: {len(train)} train / {len(val)} val; {device}; {config["scope"]}; {sum(p.numel() for p in model.parameters()):,} parameters',flush=True)
    if checkpoint is not None:del checkpoint
    for epoch in range(start,a.epochs+1):
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        training=run_epoch(model,loader(train,a.batch_size,a.workers,True,a.seed+epoch,device.type=='cuda'),device,
            optimizer=optimizer,scaler=scaler,amp=a.amp,class_weights=weights,
            binary_weight=a.binary_weight,semantic_weight=a.semantic_weight,depth_weight=a.depth_weight,max_batches=a.max_train_batches)
        validation=run_epoch(model,loader(val,a.batch_size,a.workers,False,a.seed,device.type=='cuda'),device,amp=a.amp,
            binary_weight=a.binary_weight,semantic_weight=a.semantic_weight,depth_weight=a.depth_weight)
        allocated=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None
        reserved=torch.cuda.max_memory_reserved(device)/2**30 if device.type=='cuda' else None
        values={'loss':validation['loss'],'semantic':validation['occupied_class_miou'],'iou':validation['binary_occupied_iou'],
                'boundary':validation['boundary_f1'] if validation['boundary_f1'] is not None else -1.}
        improved={k:(v<best[k] if k=='loss' else v>best[k]) for k,v in values.items()}
        for key in values:
            if improved[key]:best[key]=values[key]
        config['metrics']={'best_loss':best['loss'],'best_semantic_miou':best['semantic'],'best_binary_iou':best['iou'],'best_boundary_f1':best['boundary']}
        payload=dict(epoch=epoch,config=config,run_args=args_dict,model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                     scaler_state_dict=scaler.state_dict(),rng_state=rng_state(),best=dict(best))
        atomic_save(out/f'epoch_{epoch:03d}.pt',payload);atomic_save(out/'latest.pt',payload)
        for key,filename in [('loss','best.pt'),('semantic','best_semantic.pt'),('iou','best_iou.pt'),('boundary','best_boundary.pt')]:
            if improved[key]:atomic_save(out/filename,payload)
        record=dict(epoch=epoch,train=training,validation=validation,peak_allocated_gib=allocated,peak_reserved_gib=reserved,best=dict(best),scope=config['scope'])
        with (out/'metrics.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record)+'\n')
        print(f'Epoch {epoch}: val_loss={validation["loss"]:.4f}, IoU={values["iou"]:.3%}, mIoU={values["semantic"]:.3%}, boundary_F1={values["boundary"]:.3%}, allocated_GiB={allocated}',flush=True)
        if allocated is not None and a.allocated_limit_gib and allocated>a.allocated_limit_gib:
            raise RuntimeError('Memory budget exceeded; checkpoint saved. Lower model settings in a new run.')
        if reserved is not None and getattr(a,'reserved_limit_gib',7.) and reserved>getattr(a,'reserved_limit_gib',7.):
            raise RuntimeError('Reserved-memory budget exceeded; checkpoint saved. Lower model settings in a new run.')


if __name__=='__main__':main()
