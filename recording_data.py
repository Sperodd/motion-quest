"""Read browser JSON and build leak-audited 3-second windows for Colab/local use.

Feature calculation is injected from the shared training/server preprocessing.
No model prediction is used as a training label.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

RECORDING_CODES = ['m01','m03','m04','m06','idle']


def load_webcam_windows(path, joint_names, mp_ids, feature_fn, resample_fn,
                        window=90, fps=30, stride=30):
    path=Path(path)
    paths=sorted(path.rglob('*.json')) if path.is_dir() else [path]
    if not paths or not all(p.is_file() for p in paths):
        raise ValueError(f'No webcam recording JSON found: {path}')
    result={s:[] for s in ('train','val','test')}
    audit=[]; sessions=set(); hashes=set()
    duration=(window-1)*1000/fps
    for file in paths:
        data=json.loads(file.read_text(encoding='utf-8'))
        if (data.get('format')!='pose_tcn_recordings_v1'
            or data.get('coordinate_system')!='mediapipe_world'
            or data.get('joints')!=list(joint_names) or data.get('mediapipe_indices')!=list(mp_ids)):
            raise ValueError(f'{file.name}: recording format/joint order mismatch')
        for r in data['recordings']:
            sid=r['session_id']; subject=r['subject_id']; split=r['split']; label=r['label']
            if not isinstance(sid,str) or not sid or sid in sessions:
                raise ValueError('Duplicate or invalid recording session ID; do not copy sessions across splits')
            if not isinstance(subject,str) or not subject or split not in result or label not in RECORDING_CODES:
                raise ValueError(f'{sid}: invalid subject/split/label')
            sessions.add(sid)
            if r.get('completion')!='completed':
                raise ValueError(f'{sid}: interrupted recording. Remove it in the browser and record again.')
            times=np.asarray([f['timestamp_ms'] for f in r['frames']],dtype=np.float64)
            if len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
                raise ValueError(f'{sid}: insufficient or non-monotonic timestamps')
            if any(len(f['joints'])!=12 for f in r['frames']):
                raise ValueError(f'{sid}: expected 12 joints')
            points=np.asarray([[[p[k] for k in ('x','y','z')] for p in f['joints']]
                               for f in r['frames']],dtype=np.float64)
            visibility=np.asarray([[p['visibility'] for p in f['joints']] for f in r['frames']],dtype=np.float64)
            if (not np.isfinite(points).all() or not np.isfinite(visibility).all()
                or np.any(visibility<0) or np.any(visibility>1)):
                raise ValueError(f'{sid}: invalid coordinates/visibility')
            digest=hashlib.sha256(points.tobytes()+(times-times[0]).tobytes()).hexdigest()
            if digest in hashes:
                raise ValueError(f'{sid}: duplicate recording contents, even with a different ID')
            hashes.add(digest)
            accepted=0; rejected=0
            # Choose observed endpoints at ~stride/fps intervals. Use the same
            # ending-at-latest-frame interpolation as the server.
            endpoints=np.arange(times[0]+duration,times[-1]+1e-6,stride*1000/fps)
            for endpoint in endpoints:
                end=int(np.searchsorted(times,endpoint))
                if end>=len(times): continue
                begin=int(np.searchsorted(times,times[end]-duration,side='right'))-1
                if begin<0: continue
                ts=times[begin:end+1]; xyz=points[begin:end+1]
                if np.max(np.diff(ts))>200 or np.min(visibility[begin:end+1])<.5:
                    rejected+=1; continue
                try:
                    sampled=resample_fn(xyz,ts,window,fps)
                    features=feature_fn(sampled,[0,-1,0])
                except ValueError:
                    rejected+=1; continue
                result[split].append(dict(features=features,label=RECORDING_CODES.index(label),
                    file=f'webcam_{sid}',subject=f'mp_{subject}',session_id=sid,
                    start_frame=-1,start_ms=float(ts[-1]-duration),source='mediapipe'))
                accepted+=1
            audit.append(dict(file=file.name,session_id=sid,subject=f'mp_{subject}',split=split,
                label=label,frames=len(times),seconds=float((times[-1]-times[0])/1000),
                accepted_windows=accepted,rejected_windows=rejected,sha256=digest))
            if not accepted:
                raise ValueError(f'{sid}: no usable 3-second window (visibility, gaps or pose geometry). Record again.')
    for split, rows in result.items():
        if not any(r['label']==RECORDING_CODES.index('idle') for r in rows):
            raise ValueError(f'{split}: at least one independent idle recording with valid windows is required')
    return result,audit
