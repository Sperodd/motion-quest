import unittest
import os
from unittest.mock import patch
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from server.app import app
from server.inference import PoseSession, Predictor

ROOT = Path(__file__).resolve().parents[1]


def frame(t):
    # Synthetic standing skeleton in MediaPipe world coordinates (not an accuracy test).
    points = [[-.2,-.6,0],[.2,-.6,0],[-.3,-.3,0],[.3,-.3,0],
              [-.3,0,0],[.3,0,0],[-.15,0,0],[.15,0,0],
              [-.15,.5,0],[.15,.5,0],[-.15,1,0],[.15,1,0]]
    return {'version':1,'type':'frame','coordinate_system':'mediapipe_world','timestamp_ms':t,
            'joints':[dict(zip(['x','y','z'],p),visibility=1.) for p in points]}


class StreamingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = ROOT/'ui_prmd_tcn_output'
        if not (cls.directory/'metadata.json').exists():
            cls.directory = ROOT/'ui_prmd_tcn_angles_output'
        if not (cls.directory/'metadata.json').exists():
            raise unittest.SkipTest('No trained artifact available for regression test')
        cls.predictor = Predictor(cls.directory)

    def test_stream_parity_and_gap(self):
        session = PoseSession(self.predictor)
        for i in range(91):
            result = session.push(frame(i*1000/30))
            if result['status'] == 'prediction':
                reference = self.predictor.predict(list(session.frames))
                np.testing.assert_allclose(result['probabilities'],reference['probabilities'])
        self.assertEqual(len(session.frames),90)
        self.assertEqual(session.push(frame(4000))['status'],'prediction')
        self.assertGreater(len(session.frames),1)

    def test_quality_order_and_independent_sessions(self):
        a,b = PoseSession(self.predictor),PoseSession(self.predictor)
        a.push(frame(0))
        with self.assertRaises(ValueError): a.push(frame(0))
        bad = frame(33); bad['joints'][0]['visibility']=.1
        self.assertEqual(a.push(bad)['status'],'warming_up')
        self.assertEqual(len(a.frames),2)
        self.assertFalse(b.frames)
        bad = frame(40); bad['joints'][0]['x']=float('nan')
        with self.assertRaises(ValueError): a.push(bad)

    def test_http_and_websocket_real_model(self):
        with patch.dict(os.environ, {'POSE_MODEL_DIR':str(self.directory)}), TestClient(app) as client:
            self.assertEqual(client.get('/').status_code,200)
            self.assertEqual(len(client.get('/api/config').json()['joints']),12)
            with client.websocket_connect('/ws/pose') as ws:
                for i in range(90):
                    message = frame(i*1000/30)
                    for point in message['joints']: point['visibility'] = 0.
                    ws.send_json(message)
                    result = ws.receive_json()
                self.assertEqual(result['status'],'prediction')
                self.assertAlmostEqual(sum(result['probabilities']),1,places=5)
                ws.send_json({'version':1,'type':'reset'})
                self.assertEqual(ws.receive_json()['status'],'tracking_lost')
                ws.send_text('{bad json')
                self.assertEqual(ws.receive_json()['status'],'invalid_input')
                ws.send_json(frame(0))
                self.assertEqual(ws.receive_json()['status'],'warming_up')


if __name__ == '__main__':
    unittest.main()
