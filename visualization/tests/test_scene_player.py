"""Geometry transport checks for the scene inference viewer."""
import base64
import io
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from scene_player import image_uri, pack_centres, pack_mesh
from visualize_voxels_3d import top_down


class ScenePlayerTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.full((200,200,16),17,dtype=np.uint8)
        self.valid = np.zeros_like(self.labels,dtype=bool)

    def test_centres_keep_exact_height_and_ignore_invisible(self):
        self.labels[100,99,15]=4
        self.labels[101,99,2]=11
        self.valid[100,99,15]=True
        packed=pack_centres(self.labels,self.valid)
        indices=np.frombuffer(base64.b64decode(packed['indices']),dtype=np.uint8).reshape(-1,3)
        self.assertEqual(packed['count'],1)
        np.testing.assert_array_equal(indices,[[100,99,15]])
        centres=np.array([-40,-40,-1])+(indices.astype(float)+.5)*.4
        np.testing.assert_allclose(centres,[[.2,-.2,5.2]])

    def test_corner_encoding_keeps_upper_grid_boundary(self):
        self.labels[199,199,15]=4
        self.valid[199,199,15]=True
        packed=pack_mesh(self.labels,self.valid)
        corners=np.frombuffer(base64.b64decode(packed['corners']),dtype=np.uint8).reshape(-1,4,3)
        self.assertEqual(packed['faces'],6)
        np.testing.assert_array_equal(corners.max((0,1)),[200,200,16])
        np.testing.assert_array_equal(corners.min((0,1)),[199,199,15])

    def test_bev_selects_highest_visible_occupied_layer(self):
        self.labels[20,30,1]=11
        self.labels[20,30,7]=4
        self.labels[20,30,15]=16
        self.valid[20,30,:8]=True
        self.assertEqual(top_down(self.labels,self.valid)[20,30],4)
        self.assertEqual(top_down(self.labels,self.valid)[0,0],18)

    def test_empty_mesh_and_centres(self):
        self.assertEqual(pack_mesh(self.labels,self.valid)['faces'],0)
        self.assertEqual(pack_centres(self.labels,self.valid)['count'],0)

    def test_camera_image_uri_round_trip(self):
        image=Image.new('RGB',(384,216),(22,33,44))
        data=image_uri(image,'JPEG')
        self.assertTrue(data.startswith('data:image/jpeg;base64,'))
        with Image.open(io.BytesIO(base64.b64decode(data.split(',',1)[1]))) as decoded:
            self.assertEqual(decoded.size,(384,216))


if __name__=='__main__':
    unittest.main()
