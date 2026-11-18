"""Load and validate model artifacts, preprocess windows, and run inference."""
import importlib.util
import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch

import pose_features


class Predictor:
    def __init__(self, directory: Path):
        self.metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
        self.window = self.metadata['config']['window']
        self.fps = self.metadata['config']['target_fps']
        self.ids = self.metadata['mediapipe_indices']
        self.feature_schema = self.metadata.get('feature_schema', 'positions_v1')
        dimensions = {pose_features.FEATURE_SCHEMA: 54, 'joint_angles_v1': 18, 'positions_v1': 36}
        if self.feature_schema not in dimensions:
            raise ValueError('Unknown feature schema; use matching model and preprocessing')
        self.input_features = dimensions[self.feature_schema]
        if len(self.ids) != 12 or self.metadata['input_shape'][1:] != [self.window, self.input_features]:
            raise ValueError('Unsupported model input contract')
        if self.feature_schema != 'positions_v1':
            canonical = Path(pose_features.__file__).read_text(encoding='utf-8')
            exported = (directory / 'preprocess.py').read_text(encoding='utf-8')
            digest = hashlib.sha256(exported.encode('utf-8')).hexdigest()
            if (digest != self.metadata.get('preprocess_sha256')
                    or (self.feature_schema == pose_features.FEATURE_SCHEMA and exported != canonical)):
                raise ValueError('Feature preprocessing mismatch: sync notebook and use its matching export')
            expected_names = pose_features.FEATURE_NAMES if self.input_features == 54 else pose_features.ANGLE_NAMES
            expected_units = pose_features.FEATURE_UNITS if self.input_features == 54 else 'radians/pi'
            if (self.metadata.get('feature_names') != expected_names
                    or self.metadata.get('feature_units') != expected_units
                    or self.ids != pose_features.MP_IDS
                    or self.metadata['config'].get('input_features') != self.input_features):
                raise ValueError('Feature contract mismatch')
        spec = importlib.util.spec_from_file_location('training_preprocess', directory / 'preprocess.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if list(module.MP_IDS) != self.ids:
            raise ValueError('Metadata/preprocessor joint order mismatch')
        self.preprocess = module.mediapipe_json_window if self.feature_schema != 'positions_v1' else None
        self.normalize_window = getattr(module, 'normalize_window', None)
        self.model = torch.jit.load(str(directory / 'pose_tcn.torchscript.pt'), map_location='cpu').eval()
        self.lock = threading.Lock()
        with torch.inference_mode():
            result = self.model(torch.zeros(1, self.window, self.input_features))
        if list(result.shape) != [1, len(self.metadata['classes'])]:
            raise ValueError('Model output classes mismatch')

    def prepare_features(self, frames):
        if self.preprocess is not None:
            return self.preprocess({'coordinate_system': 'mediapipe_world', 'frames': frames},
                                   window=self.window, fps=self.fps)
        # Legacy XYZ model remains usable until the new angle model is trained.
        # Experimental permissive input: interpolate through tracking gaps and
        # ignore visibility, but retain the exact exported body normalization.
        times = np.asarray([f['timestamp_ms'] for f in frames], dtype=float)/1000
        points = np.asarray([[[f['landmarks'][j][axis] for axis in ('x','y','z')]
                              for j in self.ids] for f in frames], dtype=float)
        target = times[-1] - np.arange(self.window-1,-1,-1)/self.fps
        flat = points.reshape(len(points),36)
        sampled = np.stack([np.interp(target,times,flat[:,k]) for k in range(36)],axis=-1)
        return self.normalize_window(sampled.reshape(self.window,12,3), [0,-1,0])

    def predict(self, frames):
        start = time.perf_counter()
        features = self.prepare_features(frames)
        if features.shape != (self.window, self.input_features) or not np.isfinite(features).all():
            raise ValueError('Invalid preprocessed model input')
        with self.lock, torch.inference_mode():
            probabilities = self.model(torch.from_numpy(features[None])).softmax(-1)[0].numpy()
        if not np.isfinite(probabilities).all():
            raise ValueError('Nonfinite model output')
        index = int(probabilities.argmax())
        return {'status': 'prediction', 'class_id': self.metadata['codes'][index],
                'label': self.metadata['classes'][index], 'score': float(probabilities[index]),
                'probabilities': probabilities.tolist(),
                'inference_ms': round((time.perf_counter()-start)*1000, 2)}
