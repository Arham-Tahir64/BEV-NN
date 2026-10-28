"""Analytic tensor budget only; does not estimate full-model activation memory."""
import json
from pathlib import Path


def tensor_mib(shape, bytes_per_element=2):
    import math
    if any(int(v) != v or v <= 0 for v in shape):
        raise ValueError('Dimensions must be positive integers.')
    return math.prod(shape)*bytes_per_element/2**20


def budget(config):
    x,y,z=config['output']['grid_xyz']
    h,w=config['bev_projector']['bev_hw']; c=config['bev_projector']['channels']
    n=config['inputs']['cameras']; heights=config['bev_projector']['pillar_heights']
    samples=config['bev_projector']['samples_per_height_per_camera']
    return {
        'status':'analytic_tensor_sizes_not_measured_model_memory',
        'legacy_hidden_32channel_volume_mib':tensor_mib((x,y,z,32)),
        'proposed_bev_state_mib':tensor_mib((h,w,c)),
        'dense_18class_output_mib':tensor_mib((18,x,y,z)),
        'samples_per_projector_layer':h*w*n*heights*samples,
        'fully_materialized_sample_features_mib':tensor_mib((h,w,n,heights,samples,c)),
        'one_camera_query_chunk_features_mib':tensor_mib((config['bev_projector']['query_chunk_size'],heights,samples,c)),
        'naive_global_bev_attention_4heads_fp32_mib':tensor_mib((4,h*w,h*w),4),
        'warning':'Totals exclude saved backward activations, encoder, optimizer, convolution workspaces and allocator overhead.'
    }


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'configs/rtx4060_proposal.json').read_text())
    print(json.dumps(budget(config),indent=2))
