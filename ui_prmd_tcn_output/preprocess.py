import numpy as np

JOINTS = ['left_shoulder','right_shoulder','left_elbow','right_elbow',
          'left_wrist','right_wrist','left_hip','right_hip',
          'left_knee','right_knee','left_ankle','right_ankle']
MP_IDS = [11,12,13,14,15,16,23,24,25,26,27,28]

def normalize_window(points, up):
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (12,3) or not np.isfinite(p).all():
        raise ValueError('Expected finite [time,12,3] points')
    center = (p[:,6] + p[:,7]) / 2
    shoulder = (p[:,0] + p[:,1]) / 2
    scale = np.median(np.linalg.norm(shoulder-center, axis=-1))
    if scale < 1e-6: raise ValueError('Degenerate torso')
    vertical = np.asarray(up, dtype=np.float64)
    vertical /= np.linalg.norm(vertical)
    lateral = np.median(p[:,7]-p[:,6], axis=0)
    lateral -= np.dot(lateral, vertical)*vertical
    if np.linalg.norm(lateral) < 1e-6: raise ValueError('Degenerate hip axis')
    lateral /= np.linalg.norm(lateral)
    depth = np.cross(lateral, vertical)
    basis = np.stack([lateral,vertical,depth], axis=1)
    return (((p-center[:,None,:]) @ basis) / scale).reshape(len(p),36).astype(np.float32)

def mediapipe_json_window(payload, window=90, fps=30, visibility_threshold=0.5):
    # JSON: {coordinate_system: 'mediapipe_world', frames:
    # [{timestamp_ms: 0, landmarks: [{x,y,z,visibility}, ... 33 items]}, ...]}
    if payload.get('coordinate_system') != 'mediapipe_world':
        raise ValueError('Only MediaPipe world landmarks are supported')
    frames = payload['frames']
    times = np.asarray([f['timestamp_ms'] for f in frames], dtype=float)/1000
    if len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Timestamps must be finite and strictly increasing')
    if np.max(np.diff(times)) > 0.2: raise ValueError('Tracking gap > 0.2 seconds')
    if any(len(f['landmarks']) != 33 for f in frames): raise ValueError('Need 33 landmarks')
    p = np.asarray([[[f['landmarks'][j][a] for a in ('x','y','z')]
                     for j in MP_IDS] for f in frames], dtype=float)
    quality = np.asarray([[f['landmarks'][j].get('visibility',0) for j in MP_IDS]
                          for f in frames], dtype=float)
    if not np.isfinite(p).all() or not np.isfinite(quality).all() or np.any(quality<visibility_threshold):
        raise ValueError('Low visibility or invalid coordinates: collect a new window')
    target = times[-1] - np.arange(window-1,-1,-1)/fps
    if target[0] < times[0]-1e-6: raise ValueError('Need at least 3 seconds of history')
    flat = p.reshape(len(p),-1)
    sampled = np.stack([np.interp(target,times,flat[:,k]) for k in range(36)],axis=-1)
    return normalize_window(sampled.reshape(window,12,3), [0,-1,0])
