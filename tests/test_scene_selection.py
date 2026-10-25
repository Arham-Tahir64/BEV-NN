"""Prevent scene leakage, partial-scene training, and mini/trainval confusion."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from data.nuscenes_dataset import CAMERAS, NuScenesOccupancyDataset
from data.scene_selection import read_scene_list, split_scene_names
from prepare_scene_subset import inventory_scene, select_complete


class SceneSelectionTests(unittest.TestCase):
    def test_official_split_counts_and_no_overlap(self):
        self.assertEqual(len(split_scene_names('v1.0-mini','train')[1]),8)
        train=split_scene_names('v1.0-trainval','train')[1]
        val=split_scene_names('v1.0-trainval','val')[1]
        self.assertEqual(len(train),700)
        self.assertEqual(len(val),150)
        self.assertTrue(set(train).isdisjoint(val))

    def test_rejects_validation_leakage_duplicates_empty_and_bad_version(self):
        val=split_scene_names('v1.0-trainval','val')[1][0]
        train=split_scene_names('v1.0-trainval','train')[1][0]
        for names in ([val],[train,train],[]):
            with self.assertRaises(ValueError): split_scene_names('v1.0-trainval','train',names)
        with self.assertRaises(ValueError): split_scene_names('v1.0-test','train')

    def test_json_scene_list_is_explicit_and_validated(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'scenes.json'
            path.write_text(json.dumps(['scene-0001']))
            self.assertEqual(read_scene_list(path),('scene-0001',))
            path.write_text(json.dumps({'scenes':['scene-0001']}))
            with self.assertRaises(ValueError): read_scene_list(path)
        self.assertIsNone(read_scene_list(None))

    def test_subset_randomness_is_reproducible_and_only_uses_complete_scenes(self):
        rows=[{'scene':str(i),'complete':i!=3} for i in range(20)]
        self.assertEqual(select_complete(rows,5,42),select_complete(rows,5,42))
        self.assertNotIn('3',select_complete(rows,0,0))
        with self.assertRaises(ValueError): select_complete(rows,20,0)

    def test_full_dataset_selects_explicit_scene_and_custom_keyframe_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            name=split_scene_names('v1.0-trainval','train')[1][0]
            scene={'name':name,'token':'scene-token','first_sample_token':'sample'}
            sample={'token':'sample','next':'','timestamp':0,'data':{c:c for c in CAMERAS}}
            label=root/'gts'/name/'sample'/'labels.npz'
            label.parent.mkdir(parents=True); label.touch()
            for channel in CAMERAS:
                path=root/'imgs'/channel/f'{channel}.jpg'
                path.parent.mkdir(parents=True); path.touch()
            nusc=SimpleNamespace(version='v1.0-trainval',scene=[scene],
                get=lambda table,token:sample,
                get_sample_data_path=lambda token:str(root/'missing'/(token+'.jpg')))
            geometry={c:{'image_size':(8,4),'intrinsics':np.eye(3),'ego_to_camera':np.eye(4)} for c in CAMERAS}
            with patch('data.voxel_projection.camera_geometry',return_value=geometry):
                dataset=NuScenesOccupancyDataset(dataroot=root,occupancy_root=root/'gts',
                    version='v1.0-trainval',scene_names=[name],image_root=root/'imgs',nusc=nusc)
                self.assertEqual(len(dataset),1)
                self.assertEqual(dataset.version,'v1.0-trainval')
                self.assertEqual(dataset.scene_names,(name,))
                self.assertEqual(dataset.records[0]['image_paths'][0],root/'imgs'/'CAM_FRONT'/'CAM_FRONT.jpg')
                with self.assertRaisesRegex(ValueError,'missing scenes'):
                    NuScenesOccupancyDataset(dataroot=root,version='v1.0-trainval',nusc=nusc)
            row=inventory_scene(nusc,scene,root/'gts',root/'imgs')
            self.assertTrue(row['complete'])
            (root/'imgs'/'CAM_FRONT'/'CAM_FRONT.jpg').unlink()
            row=inventory_scene(nusc,scene,root/'gts',root/'imgs')
            self.assertFalse(row['complete'])
            self.assertEqual(row['missing_images'],1)
