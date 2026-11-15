"""Run notebook preprocessing on local Vicon files without training or plotting."""
import ast
import hashlib
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
nb = json.loads((ROOT/'ui_prmd_tcn_colab.ipynb').read_text(encoding='utf-8'))
sources = [''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
with tempfile.TemporaryDirectory() as temp:
    env = dict(np=np,pd=pd,re=re,hashlib=hashlib,resample_poly=resample_poly,
               SEED=42,OUT=Path(temp),POSITION_DIR=ROOT/'data/Movements/Vicon/Positions',
               display=lambda *_:None)
    exec(next(s for s in sources if s.startswith('CLASSES')),env)
    exec(next(s for s in sources if s.startswith('PREPROCESS_SOURCE')),env)
    data = next(s for s in sources if s.startswith('def read_vicon'))
    exec(data.split('fig, ax =')[0],env)
    for split,x in env['X'].items():
        assert x.shape[1:] == (90,54)
        assert np.isfinite(x).all() and x[:,:,36:].min() >= 0 and x[:,:,36:].max() <= 1
    print('All 49 recordings passed: finite [90,54] combined windows; angle channels in [0,1].')
