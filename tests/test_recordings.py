import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

import pose_features as pf
from recording_data import load_webcam_windows
from test_stream import frame

ROOT=Path(__file__).resolve().parents[1]


def bundle():
    records=[]
    for i,split in enumerate(('train','val','test')):
        frames=[]
        for j in range(151):
            item=frame(j*1000/30)
            for k,p in enumerate(item['joints']):
                p['z'] += .001*np.sin(j*.13+k+i)
            frames.append({k:item[k] for k in ('timestamp_ms','joints')})
        records.append(dict(subject_id='person01',session_id=f'session{i}',split=split,label='idle',
                            completion='completed',frames=frames))
    return dict(format='pose_tcn_recordings_v1',coordinate_system='mediapipe_world',
                joints=pf.JOINTS,mediapipe_indices=pf.MP_IDS,recordings=records)


class RecordingTests(unittest.TestCase):
    def load(self,data):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'captures.json'
            path.write_text(json.dumps(data),encoding='utf-8')
            return load_webcam_windows(path,pf.JOINTS,pf.MP_IDS,pf.window_features,pf.resample_window)

    def test_mapping_server_parity_and_splits(self):
        data=bundle(); windows,audit=self.load(data)
        self.assertEqual(len(audit),3)
        for split,rows in windows.items():
            self.assertGreater(len(rows),0)
            self.assertEqual({r['label'] for r in rows},{4})
            self.assertTrue(all(r['features'].shape==(90,54) for r in rows))
        captured=data['recordings'][0]['frames'][:90]
        server_frames=[]
        for f in captured:
            lm=[dict(x=0,y=0,z=0,visibility=1) for _ in range(33)]
            for idx,j in zip(pf.MP_IDS,f['joints']): lm[idx]=j
            server_frames.append(dict(timestamp_ms=f['timestamp_ms'],landmarks=lm))
        expected=pf.mediapipe_json_window(dict(coordinate_system='mediapipe_world',frames=server_frames))
        np.testing.assert_allclose(windows['train'][0]['features'],expected,atol=1e-6)

    def test_reject_duplicate_wrong_order_missing_split_and_interruption(self):
        data=bundle(); data['recordings'][1]['session_id']='session0'
        with self.assertRaisesRegex(ValueError,'Duplicate'): self.load(data)
        data=bundle(); data['recordings'][1]['frames']=copy.deepcopy(data['recordings'][0]['frames'])
        with self.assertRaisesRegex(ValueError,'duplicate recording contents'): self.load(data)
        data=bundle(); data['recordings'][0]['label']='m07'
        with self.assertRaisesRegex(ValueError,'invalid subject/split/label'): self.load(data)
        data=bundle(); data['joints']=data['joints'][::-1]
        with self.assertRaisesRegex(ValueError,'joint order'): self.load(data)
        data=bundle(); data['recordings'].pop()
        with self.assertRaisesRegex(ValueError,'test:'): self.load(data)
        data=bundle(); data['recordings'][0]['completion']='interrupted'
        with self.assertRaisesRegex(ValueError,'interrupted'): self.load(data)

    def test_gap_and_visibility_not_synthesized_as_idle(self):
        data=bundle()
        for f in data['recordings'][0]['frames']: f['joints'][0]['visibility']=.1
        with self.assertRaisesRegex(ValueError,'no usable'): self.load(data)
        data=bundle(); data['recordings'][0]['frames']=data['recordings'][0]['frames'][::15]
        with self.assertRaisesRegex(ValueError,'no usable'): self.load(data)

    def test_notebook_loader_sync(self):
        nb=json.loads((ROOT/'ui_prmd_tcn_colab.ipynb').read_text(encoding='utf-8'))
        sources=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        source=next(s for s in sources if s.startswith('RECORDING_SOURCE'))
        self.assertEqual(ast.literal_eval(ast.parse(source).body[0].value),
                         (ROOT/'recording_data.py').read_text(encoding='utf-8'))
        env={'SEED':42}
        exec(next(s for s in sources if s.startswith('CLASSES')),env)
        self.assertEqual(env['NUM_CLASSES'],5)
        self.assertEqual(env['CODES'],['m01','m03','m04','m06','idle'])


if __name__=='__main__': unittest.main()
