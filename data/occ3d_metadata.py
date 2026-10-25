"""Camera/pose metadata adapter for the official Occ3D annotations.json.

Supports the metadata methods used by our dataset and viewer, without requiring
raw sensor files or a separate nuScenes database download. It does not provide
3D object-box annotations or the complete nuScenes devkit API.
"""
import json
from pathlib import Path, PurePosixPath
from PIL import Image
from .scene_selection import split_scene_names

CAMERA_NAMES=('CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT')


class Occ3DMetadata:
    def __init__(self, annotations_json, dataroot, version='v1.0-trainval', image_root=None):
        self.version=version
        self.dataroot=str(Path(dataroot).resolve())
        self.image_root=Path(image_root).resolve() if image_root is not None else Path(dataroot).resolve()/'imgs'
        doc=json.loads(Path(annotations_json).read_text(encoding='utf-8'))
        if not isinstance(doc,dict) or not {'train_split','val_split','scene_infos'}<=doc.keys():
            raise ValueError('Expected official Occ3D scene_infos and split annotations.')
        for split in ('train','val'):
            _,official=split_scene_names('v1.0-trainval',split)
            if set(doc[split+'_split'])-set(official):
                raise ValueError(f'Occ3D annotations contain scenes outside official {split} split.')
        if set(doc['train_split'])&set(doc['val_split']): raise ValueError('Annotation train/val overlap.')
        wanted=set(split_scene_names(version,'train')[1])|set(split_scene_names(version,'val')[1])
        self._frames={}; self._frame_scenes={}; self.scene=[]; self._scenes={}
        for name,frames in doc['scene_infos'].items():
            if name not in wanted: continue
            if not frames: raise ValueError(f'No frames in {name}.')
            ordered=sorted(frames,key=lambda token:int(frames[token]['timestamp']))
            scene={'name':name,'token':name,'first_sample_token':ordered[0]}
            self.scene.append(scene); self._scenes[name]=scene
            for i,token in enumerate(ordered):
                frame=frames[token]
                expected_next=ordered[i+1] if i+1<len(ordered) else ''
                expected_prev=ordered[i-1] if i else ''
                if self._end(frame['next'])!=expected_next or self._end(frame['prev'])!=expected_prev:
                    raise ValueError(f'Broken temporal chain in {name}/{token}.')
                if token in self._frames: raise ValueError('Duplicate sample token across scenes.')
                if set(frame['camera_sensor'])!=set(CAMERA_NAMES): raise ValueError('Expected all six cameras.')
                self._frames[token]=frame; self._frame_scenes[token]=name

    @staticmethod
    def _end(token): return '' if token in ('','EOF',None) else token

    def _sensor(self,token):
        sample,channel=token.rsplit(':',1)
        return sample,channel,self._frames[sample]

    def get_sample_data_path(self,token):
        sample,channel,frame=self._sensor(token)
        if channel not in CAMERA_NAMES: raise ValueError('Reference pose has no sensor image.')
        filename=PurePosixPath(frame['camera_sensor'][channel]['img_path']).name
        return str(self.image_root/channel/filename)

    def get(self,table,token):
        if table=='scene': return self._scenes[token]
        if table=='sample':
            frame=self._frames[token]
            # Put the exact sample-time reference first. Camera acquisitions
            # have their own ego poses and need not share this timestamp.
            data={'OCC3D_REFERENCE':token+':reference'}
            data.update({c:token+':'+c for c in CAMERA_NAMES})
            return {'token':token,'timestamp':int(frame['timestamp']),'data':data,
                'scene_token':self._frame_scenes[token], 'next':self._end(frame['next']),
                'prev':self._end(frame['prev'])}
        sample,channel,frame=self._sensor(token)
        if table=='ego_pose':
            return frame['ego_pose'] if channel=='reference' else frame['camera_sensor'][channel]['ego_pose']
        if channel=='reference' and table=='sample_data':
            return {'timestamp':int(frame['timestamp']),'ego_pose_token':token}
        camera=frame['camera_sensor'][channel]
        if table=='calibrated_sensor':
            return {**camera['extrinsic'],'camera_intrinsic':camera['intrinsics']}
        if table=='sample_data':
            with Image.open(self.get_sample_data_path(token)) as im: width,height=im.size
            return {'timestamp':int(camera['ego_pose']['timestamp']),'ego_pose_token':token,
                    'calibrated_sensor_token':token,'width':width,'height':height}
        raise KeyError(f'Occ3D adapter does not provide table {table}.')

    @property
    def sample(self): return [self.get('sample',token) for token in self._frames]
