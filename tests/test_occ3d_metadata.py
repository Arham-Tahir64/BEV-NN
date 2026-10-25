"""Check asynchronous camera poses and safe dataset installation."""
import copy
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import numpy as np
from PIL import Image
from data.occ3d_metadata import Occ3DMetadata, CAMERA_NAMES
from data.scene_selection import split_scene_names
from data.voxel_projection import camera_geometry
from extract_occ3d import checked_target
from prepare_scene_subset import select_complete, compatible_mini_scenes


class Occ3DMetadataTests(unittest.TestCase):
    def fixture(self,root):
        name=split_scene_names('v1.0-trainval','train')[1][0]
        pose=lambda x:{'translation':[x,0,0],'rotation':[1,0,0,0]}
        cameras={}
        for channel in CAMERA_NAMES:
            path=root/'imgs'/channel/'frame.jpg'; path.parent.mkdir(parents=True)
            Image.new('RGB',(8,4)).save(path)
            cameras[channel]={'intrinsics':np.eye(3).tolist(),'extrinsic':pose(.5),
                'ego_pose':{**pose(4),'timestamp':900},'img_path':channel+'/frame.jpg'}
        doc={'train_split':[name],'val_split':[], 'scene_infos':{name:{'frame':{
            'timestamp':1000,'ego_pose':pose(5),'camera_sensor':cameras,
            'prev':'EOF','next':'EOF','gt_path':'gts/'+name+'/frame/labels.npz'}}}}
        path=root/'annotations.json'; path.write_text(json.dumps(doc))
        return path,doc

    def test_uses_sample_time_pose_and_separate_camera_acquisition_pose(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); path,_=self.fixture(root)
            meta=Occ3DMetadata(path,root)
            sample=meta.get('sample','frame')
            self.assertEqual(sample['next'],'')
            geometry=camera_geometry(meta,sample)
            for channel in CAMERA_NAMES:
                expected=np.eye(4); expected[0,3]=.5
                np.testing.assert_allclose(geometry[channel]['ego_to_camera'],expected)
                self.assertEqual(geometry[channel]['image_size'],(8,4))

    def test_broken_temporal_chain_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); path,doc=self.fixture(root)
            frame=next(iter(doc['scene_infos'].values()))['frame']; frame['next']='missing'
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,'temporal chain'): Occ3DMetadata(path,root)

    def test_large_subset_can_preserve_original_baseline_scenes(self):
        rows=[{'scene':str(i),'complete':True} for i in range(10)]
        selected=select_complete(rows,5,0,required=('1','2'))
        self.assertTrue({'1','2'}<=set(selected))
        self.assertEqual(len(selected),5)
        with self.assertRaises(ValueError): select_complete(rows,1,0,required=('1','2'))

    def test_mini_roles_cannot_leak_into_official_full_split(self):
        anchors=compatible_mini_scenes('v1.0-trainval','train')
        self.assertEqual(len(anchors),6)
        self.assertNotIn('scene-0553',anchors)
        self.assertNotIn('scene-0796',anchors)
        self.assertEqual(len(compatible_mini_scenes('v1.0-trainval','val')),2)

    def test_extraction_rejects_traversal_drives_and_links(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve()
            for name in ('../outside','/outside','C:/outside','gts/../../outside','imgs\\outside'):
                with self.assertRaises(ValueError): checked_target(root,tarfile.TarInfo(name))
            member=tarfile.TarInfo('gts/link'); member.type=tarfile.SYMTYPE
            with self.assertRaises(ValueError): checked_target(root,member)
            self.assertEqual(checked_target(root,tarfile.TarInfo('./imgs/camera/file.jpg')),root/'imgs/camera/file.jpg')


if __name__=='__main__': unittest.main()
