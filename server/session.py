"""Per-connection frame validation, buffering and inference scheduling."""
import math
from collections import deque

from .settings import INFERENCE_INTERVAL_MS


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class PoseSession:
    def __init__(self, predictor):
        self.predictor = predictor
        self.frames = deque(maxlen=600)
        self.last_timestamp = None
        self.last_inference = -math.inf
        self.duration_ms = (predictor.window-1)*1000/predictor.fps

    def reset(self):
        self.frames.clear()
        self.last_timestamp = None
        self.last_inference = -math.inf

    def push(self, message):
        if not isinstance(message, dict) or message.get('version') != 1:
            raise ValueError('Expected protocol version 1')
        if message.get('type') == 'reset':
            self.reset()
            return {'status': 'tracking_lost', 'progress': 0}
        if message.get('type') != 'frame' or message.get('coordinate_system') != 'mediapipe_world':
            raise ValueError('Expected a mediapipe_world frame')
        timestamp = message.get('timestamp_ms')
        points = message.get('joints')
        if not finite_number(timestamp) or timestamp < 0:
            raise ValueError('Invalid timestamp_ms')
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError('Timestamps must increase')
        if not isinstance(points, list) or len(points) != 12:
            raise ValueError('Expected 12 joints in /api/config order')
        for point in points:
            if not isinstance(point, dict) or not all(finite_number(point.get(k)) for k in ('x','y','z','visibility')):
                raise ValueError('Invalid joint coordinates or visibility')
            if not 0 <= point['visibility'] <= 1:
                raise ValueError('Visibility must be in [0,1]')
        self.last_timestamp = timestamp
        # Keep the landmark mapping used by the training export.
        landmarks = [dict(x=0., y=0., z=0., visibility=1.) for _ in range(33)]
        for index, point in zip(self.predictor.ids, points):
            landmarks[index] = point
        self.frames.append({'timestamp_ms': timestamp, 'landmarks': landmarks})
        cutoff = timestamp-self.duration_ms
        while len(self.frames) > 2 and self.frames[1]['timestamp_ms'] <= cutoff:
            self.frames.popleft()
        elapsed = timestamp-self.frames[0]['timestamp_ms']
        if elapsed < self.duration_ms-1e-6:
            return {'status': 'warming_up', 'progress': min(elapsed/self.duration_ms, 1)}
        if timestamp-self.last_inference < INFERENCE_INTERVAL_MS:
            return {'status': 'collecting', 'progress': 1}
        result = self.predictor.predict(list(self.frames))
        self.last_inference = timestamp
        return {**result, 'timestamp_ms': timestamp, 'progress': 1}
