import contextlib
import io
import unittest
import torch
from torch import nn
from foundation4d.runtime import run_epoch
from foundation4d.model import Foundation4DLite


class Tiny(nn.Module):
    voxel_bounds=(-1.,-1.,-1.,1.,1.,1.)
    predict=staticmethod(Foundation4DLite.predict)
    def __init__(self):
        super().__init__()
        self.binary=nn.Parameter(torch.zeros(2))
        self.semantic=nn.Parameter(torch.zeros(17))

    def forward_heads(self,*args):
        return dict(binary_logits=self.binary.reshape(1,1,1,2),
                    semantic_logits=self.semantic.reshape(1,17,1,1,1).expand(1,17,1,1,2),depth_logits=None)


def sample(labels=(0,17),mask=(True,True)):
    return dict(images=torch.zeros(1,6,3,1,1),intrinsics=torch.eye(3).repeat(1,6,1,1),
                ego_to_camera=torch.eye(4).repeat(1,6,1,1),semantics=torch.tensor(labels).reshape(1,1,1,2),
                valid_mask=torch.tensor(mask).reshape(1,1,1,2))


class RuntimeTests(unittest.TestCase):
    def test_training_updates_both_heads_and_counts_updates(self):
        m=Tiny();opt=torch.optim.AdamW(m.parameters(),lr=.01)
        with contextlib.redirect_stdout(io.StringIO()):
            result=run_epoch(m,[sample()],torch.device('cpu'),optimizer=opt,
                             scaler=torch.amp.GradScaler('cuda',enabled=False),amp=False)
        self.assertEqual(result['optimizer_updates'],1)
        self.assertGreater(m.binary[0].item(),0);self.assertLess(m.binary[1].item(),0)
        self.assertGreater(m.semantic[0].item(),0)

    def test_validation_exact_global_metrics_and_no_weight_change(self):
        m=Tiny();before=m.binary.detach().clone()
        with contextlib.redirect_stdout(io.StringIO()):
            result=run_epoch(m,[sample(),sample(labels=(255,255))],torch.device('cpu'),amp=False)
        self.assertEqual(result['binary_occupied_iou'],.5)
        self.assertEqual(result['occupied_precision'],.5)
        self.assertEqual(result['occupied_recall'],1.)
        self.assertEqual(result['skipped_batches'],1)
        self.assertEqual(result['range_metrics']['0_10']['binary_occupied_iou'],.5)
        torch.testing.assert_close(m.binary,before);self.assertIsNone(m.binary.grad)

    def test_all_unknown_epoch_raises(self):
        with self.assertRaisesRegex(RuntimeError,'No supervised'):
            run_epoch(Tiny(),[sample(labels=(255,255))],torch.device('cpu'),amp=False)


if __name__=='__main__':unittest.main()
