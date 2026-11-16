"""Check browser captures before uploading to Drive; does not modify recordings."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pose_features import JOINTS, MP_IDS, window_features, resample_window
from recording_data import load_webcam_windows

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('path',help='webcam_recordings.json or directory of recording bundles')
    args=parser.parse_args()
    windows,audit=load_webcam_windows(args.path,JOINTS,MP_IDS,window_features,resample_window)
    for row in audit:
        print(f"{row['split']:5} {row['label']:4} {row['subject']} {row['seconds']:.1f}s: "
              f"{row['accepted_windows']} usable / {row['rejected_windows']} rejected windows")
    for split in windows:
        print(split,len(windows[split]),'windows')
    print('Recording validation passed. This checks format/quality, not label correctness.')
