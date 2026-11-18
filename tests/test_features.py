import ast
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from fastapi.testclient import TestClient

import pose_features as features
from server.app import app
from server.inference import Predictor
from tests.helpers import frame

ROOT = Path(__file__).resolve().parents[1]


def notebook_sources():
    nb = json.loads((ROOT/'ui_prmd_tcn_colab.ipynb').read_text(encoding='utf-8'))
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']


def trajectory():
    base = np.array([[p[a] for a in ('x','y','z')] for p in frame(0)['joints']])
    points = np.repeat(base[None], 91, axis=0)
    t = np.linspace(0,np.pi,91)
    points[:,8,2] += np.sin(t)*.25
    points[:,10,2] -= np.sin(t)*.1
    return points


class AngleFeaturesTest(unittest.TestCase):
    def test_known_geometry_and_invalid_vectors(self):
        np.testing.assert_allclose(features.angle_between([[1,0,0]]*3,
            [[1,0,0],[0,1,0],[-1,0,0]]), [0,.5,1])
        p = trajectory()[:1]
        angles = features.angle_features(p,[0,-1,0])
        np.testing.assert_allclose(angles[0,[0,1,8,9,10,15,16,17]], [1,1,0,0,0,0,0,0], atol=1e-7)
        broken = p.copy(); broken[:,8] = broken[:,6]
        with self.assertRaisesRegex(ValueError,'Degenerate'):
            features.window_features(broken,[0,-1,0])
        with self.assertRaises(ValueError): features.window_features(p,[0,0,0])

    def test_scale_rotation_reflection_and_feature_mirror(self):
        p = trajectory()
        original = features.window_features(p,[0,-1,0])
        rotation = np.array([[0,1,0],[0,0,1],[1,0,0]])
        converted = features.window_features((p @ rotation)*1000 + [100,200,300],
                                            np.array([0,-1,0]) @ rotation)
        np.testing.assert_allclose(original, converted, atol=1e-7)
        mirrored = p[:,[1,0,3,2,5,4,7,6,9,8,11,10]].copy()
        mirrored[:,:,0] *= -1
        np.testing.assert_allclose(features.window_features(mirrored,[0,-1,0]),
                                   original[:,features.MIRROR_INDICES]*features.MIRROR_SIGNS, atol=1e-7)
        self.assertGreater(original[:,36+15].max(),.05)

    def test_notebook_source_and_syntax(self):
        for source in notebook_sources():
            compile(source,'notebook','exec')
        cell = next(s for s in notebook_sources() if s.startswith('PREPROCESS_SOURCE'))
        self.assertEqual(ast.literal_eval(ast.parse(cell).body[0].value),
                         (ROOT/'pose_features.py').read_text(encoding='utf-8'))
        setup = next(s for s in notebook_sources() if s.startswith('MODE ='))
        self.assertIn("OUT = Path('/content/ui_prmd_tcn_output')",setup)
        export = next(s for s in notebook_sources() if s.startswith('model=model.cpu()'))
        self.assertIn('feature_units=FEATURE_UNITS',export)
        self.assertIn('/content/drive/MyDrive/ui_prmd_tcn_output.zip',export)

    def test_position_and_angle_blocks_and_augmentation(self):
        p = trajectory()
        full = features.window_features(p,[0,-1,0])
        self.assertEqual(full.shape,(91,54))
        np.testing.assert_array_equal(full[:,:36],features.position_features(p,[0,-1,0]))
        np.testing.assert_array_equal(full[:,36:],features.angle_features(p,[0,-1,0]))
        xyz = full[:,:36].reshape(-1,12,3)
        np.testing.assert_allclose((xyz[:,6]+xyz[:,7])/2,0,atol=1e-7)
        self.assertLess(full[:,:36].min(),0)
        # Run the notebook's actual Torch augmentation with a deterministic mask.
        code = next(s for s in notebook_sources() if s.startswith('MODEL_SOURCE'))
        fn = next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='augment')
        env = dict(torch=torch,MIRROR_INDICES=features.MIRROR_INDICES,MIRROR_SIGNS=features.MIRROR_SIGNS)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'augment','exec'),env)
        with patch('torch.rand',return_value=torch.zeros(1)):
            augmented = env['augment'](torch.from_numpy(full[None])).numpy()[0]
        mirrored = p[:,[1,0,3,2,5,4,7,6,9,8,11,10]].copy()
        mirrored[:,:,0]*=-1
        np.testing.assert_allclose(augmented,features.window_features(mirrored,[0,-1,0]),atol=1e-7)


class AngleModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        src = next(s for s in notebook_sources() if s.startswith('MODEL_SOURCE'))
        env = {}
        exec(ast.literal_eval(ast.parse(src).body[0].value), env)
        config_env = {'SEED':42}
        exec(next(s for s in notebook_sources() if s.startswith('CLASSES')), config_env)
        cls.metadata = dict(config=config_env['CFG'], classes=config_env['CLASSES'],
                            codes=config_env['CODES'], joints=features.JOINTS, mediapipe_indices=features.MP_IDS)
        cls.metadata['config']['input_features'] = 54
        cls.model = env['PoseTCN'](cls.metadata['config'], np.zeros((1,1,54)), np.ones((1,1,54)))
        # Exercise the real notebook architecture's optimizer path, not accuracy.
        x = torch.from_numpy(features.window_features(trajectory()[:90],[0,-1,0])[None])
        optimizer = torch.optim.AdamW(cls.model.parameters(), lr=.001)
        loss = torch.nn.functional.cross_entropy(cls.model(x), torch.tensor([0]))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        cls.model.eval()
        torch.jit.trace(cls.model,x).save(str(cls.directory/'pose_tcn.torchscript.pt'))
        source = (ROOT/'pose_features.py').read_text(encoding='utf-8')
        (cls.directory/'preprocess.py').write_text(source,encoding='utf-8')
        cls.metadata.update(feature_schema=features.FEATURE_SCHEMA,feature_names=features.FEATURE_NAMES,
            feature_units=features.FEATURE_UNITS, input_shape=['batch',90,54],
            preprocess_sha256=hashlib.sha256(source.encode('utf-8')).hexdigest())
        (cls.directory/'metadata.json').write_text(json.dumps(cls.metadata),encoding='utf-8')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_xyz_json_to_angle_model_matches_training(self):
        predictor = Predictor(self.directory)
        p = trajectory()[::2]  # 15 Hz client, 30 Hz model; XYZ interpolation comes first.
        times = np.arange(len(p))*1000/15
        frames = []
        for t, points in zip(times,p):
            landmarks = [dict(x=0,y=0,z=0,visibility=0) for _ in range(33)]
            for j, point in zip(features.MP_IDS, points):
                landmarks[j] = dict(zip(['x','y','z'],map(float,point)),visibility=0)
            frames.append({'timestamp_ms':float(t),'landmarks':landmarks})
        # Independent interpolation of XYZ, then same feature function used by training.
        target = times[-1] - np.arange(89,-1,-1)*1000/30
        expected_points = np.stack([np.interp(target,times,p.reshape(len(p),36)[:,k])
                                    for k in range(36)],axis=-1).reshape(90,12,3)
        expected = features.window_features(expected_points,[0,-1,0])
        np.testing.assert_allclose(predictor.prepare_features(frames),expected,atol=1e-6)
        with torch.inference_mode():
            reference = self.model(torch.from_numpy(expected[None])).softmax(-1)[0].numpy()
        np.testing.assert_allclose(predictor.predict(frames)['probabilities'],reference,atol=1e-6)

    def test_angle_websocket_and_metadata_guard(self):
        with patch.dict(os.environ, {'POSE_MODEL_DIR': str(self.directory)}), TestClient(app) as client:
            self.assertEqual(client.get('/api/config').json()['input_features'],54)
            with client.websocket_connect('/ws/pose') as ws:
                for i in range(90):
                    ws.send_json(frame(i*1000/30))
                    reply=ws.receive_json()
                self.assertEqual(reply['status'],'prediction')
                self.assertEqual(len(reply['probabilities']),len(self.metadata['classes']))
        original = (self.directory/'metadata.json').read_text()
        try:
            invalid = dict(self.metadata,preprocess_sha256='wrong')
            (self.directory/'metadata.json').write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError,'preprocessing mismatch'):
                Predictor(self.directory)
            invalid = dict(self.metadata,input_shape=['batch',90,36])
            (self.directory/'metadata.json').write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError,'input contract'):
                Predictor(self.directory)
        finally:
            (self.directory/'metadata.json').write_text(original)


if __name__ == '__main__':
    unittest.main()
