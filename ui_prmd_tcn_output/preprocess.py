"""Shared Colab/server contract: body-normalized XYZ (36) + angles (18).

Only the selected 12 XYZ joints are used. No UI-PRMD Angles files are read.
Edit this file, then run scripts/sync_notebook_features.py before training.
"""
import numpy as np

FEATURE_SCHEMA = 'positions_angles_v1'
JOINTS = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
          'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
          'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
MP_IDS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
ANGLE_NAMES = [
    'left_knee', 'right_knee', 'left_hip', 'right_hip',
    'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
    'thigh_spread', 'shin_spread', 'trunk_to_up',
    'left_thigh_to_down', 'right_thigh_to_down',
    'left_shin_to_down', 'right_shin_to_down',
    'knee_asymmetry', 'hip_asymmetry', 'thigh_tilt_asymmetry',
]
ANGLE_MIRROR_INDICES = [1, 0, 3, 2, 5, 4, 7, 6, 8, 9, 10, 12, 11, 14, 13, 15, 16, 17]
POSITION_NAMES = [f'{joint}_{axis}' for joint in JOINTS for axis in ('lateral','up','depth')]
FEATURE_NAMES = POSITION_NAMES + ANGLE_NAMES
FEATURE_UNITS = ['torso_length'] * 36 + ['radians/pi'] * 18
MIRROR_INDICES = [joint * 3 + axis for joint in (1,0,3,2,5,4,7,6,9,8,11,10)
                  for axis in range(3)] + [36 + i for i in ANGLE_MIRROR_INDICES]
MIRROR_SIGNS = [-1 if i % 3 == 0 else 1 for i in range(36)] + [1] * 18


def angle_between(u, v):
    """arccos(clip(dot(unit(u), unit(v)), -1, 1)) / pi; output in [0,1]."""
    u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    nu, nv = np.linalg.norm(u, axis=-1, keepdims=True), np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(nu < 1e-8) or np.any(nv < 1e-8):
        raise ValueError('Degenerate joint vector; cannot compute angle')
    cosine = np.sum((u / nu) * (v / nv), axis=-1)
    return np.arccos(np.clip(cosine, -1., 1.)) / np.pi


def angle_features(points, up):
    """[T,12,3] resampled XYZ -> [T,18] float32, before train standardization.

    Vicon: up=[0,0,1]. MediaPipe world: up=[0,-1,0].
    Triple (A,B,C) measures the interior angle at B: A-B versus C-B.
    Thus an extended knee is 180 degrees (1.0), not zero flexion.
    """
    p = np.asarray(points, dtype=np.float64)
    vertical = np.asarray(up, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (12, 3) or len(p) == 0 or not np.isfinite(p).all():
        raise ValueError('Expected finite [time,12,3] points')
    if vertical.shape != (3,) or not np.isfinite(vertical).all() or np.linalg.norm(vertical) < 1e-8:
        raise ValueError('Invalid up axis')
    vertical = vertical / np.linalg.norm(vertical)
    center = (p[:, 6] + p[:, 7]) / 2
    torso = (p[:, 0] + p[:, 1]) / 2 - center
    scale = np.linalg.norm(torso, axis=-1)
    if np.any(scale < 1e-8):
        raise ValueError('Degenerate torso')
    # Translation/scale do not change angles. Normalize for a unit-independent
    # degeneracy tolerance across Vicon millimeters and MediaPipe meters.
    p = (p - center[:, None, :]) / scale[:, None, None]

    def joint(a, b, c):
        return angle_between(p[:, a]-p[:, b], p[:, c]-p[:, b])

    values = [joint(6,8,10), joint(7,9,11), joint(0,6,8), joint(1,7,9),
              joint(0,2,4), joint(1,3,5), joint(2,0,6), joint(3,1,7)]
    lt, rt = p[:,8]-p[:,6], p[:,9]-p[:,7]
    ls, rs = p[:,10]-p[:,8], p[:,11]-p[:,9]
    trunk = (p[:,0]+p[:,1])/2 - (p[:,6]+p[:,7])/2
    values += [angle_between(lt,rt), angle_between(ls,rs), angle_between(trunk,vertical),
               angle_between(lt,-vertical), angle_between(rt,-vertical),
               angle_between(ls,-vertical), angle_between(rs,-vertical)]
    values += [np.abs(values[0]-values[1]), np.abs(values[2]-values[3]),
               np.abs(values[11]-values[12])]
    return np.stack(values, axis=-1).astype(np.float32)


def position_features(points, up):
    """Hip-centered XYZ in a window-fixed body basis, scaled by median torso length.

    Lateral = horizontal left-to-right hip direction; depth = lateral cross up.
    Keep gravity-based vertical instead of aligning with the tilted torso.
    """
    p = np.asarray(points, dtype=np.float64)
    vertical = np.asarray(up, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (12,3) or not len(p) or not np.isfinite(p).all():
        raise ValueError('Expected finite [time,12,3] points')
    if vertical.shape != (3,) or not np.isfinite(vertical).all() or np.linalg.norm(vertical) < 1e-8:
        raise ValueError('Invalid up axis')
    vertical = vertical / np.linalg.norm(vertical)
    center = (p[:,6] + p[:,7]) / 2
    scale = np.median(np.linalg.norm((p[:,0]+p[:,1])/2-center, axis=-1))
    if scale < 1e-8:
        raise ValueError('Degenerate torso')
    centered = (p-center[:,None,:])/scale
    # Mean, rather than component-wise median, preserves rotation equivariance.
    lateral = np.mean(centered[:,7]-centered[:,6], axis=0)
    lateral -= np.dot(lateral,vertical)*vertical
    if np.linalg.norm(lateral) < 1e-8:
        raise ValueError('Degenerate horizontal hip direction')
    lateral /= np.linalg.norm(lateral)
    basis = np.stack([lateral,vertical,np.cross(lateral,vertical)], axis=1)
    return (centered @ basis).reshape(len(p),36).astype(np.float32)


def window_features(points, up):
    """[T,12,3] -> [T,54]: 36 positions followed by 18 angles.

    Both feature groups use the same resampled XYZ. Train standardization is
    per feature channel and happens inside the exported model, not here.
    """
    return np.concatenate([position_features(points,up), angle_features(points,up)],axis=-1)


def resample_window(points, timestamps_ms, window=90, fps=30):
    """Resample XYZ first, then compute angles; never interpolate angle values.

    Matches the current permissive streaming policy: long gaps are interpolated.
    Visibility is not used as an angle feature or a filter.
    """
    p = np.asarray(points, dtype=np.float64)
    times = np.asarray(timestamps_ms, dtype=np.float64) / 1000
    if p.shape != (len(times),12,3) or not np.isfinite(p).all():
        raise ValueError('Expected finite timestamped [time,12,3] points')
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Timestamps must be finite and strictly increasing')
    if window < 2 or fps <= 0:
        raise ValueError('Invalid window/fps')
    target = times[-1] - np.arange(window-1,-1,-1)/fps
    if target[0] < times[0]-1e-6:
        raise ValueError('Insufficient window history')
    flat = p.reshape(len(p),36)
    return np.stack([np.interp(target,times,flat[:,k]) for k in range(36)],axis=-1).reshape(window,12,3)


def mediapipe_json_window(payload, window=90, fps=30):
    """Exported notebook/server adapter for a buffered list of 33-landmark frames."""
    if payload.get('coordinate_system') != 'mediapipe_world':
        raise ValueError('Only MediaPipe world landmarks are supported')
    frames = payload['frames']
    if any(len(frame['landmarks']) != 33 for frame in frames):
        raise ValueError('Need 33 landmarks per frame')
    points = [[[frame['landmarks'][j][axis] for axis in ('x','y','z')]
               for j in MP_IDS] for frame in frames]
    sampled = resample_window(points, [frame['timestamp_ms'] for frame in frames], window, fps)
    return window_features(sampled, [0,-1,0])
