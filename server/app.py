import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .inference import PoseSession, Predictor

ROOT = Path(__file__).resolve().parents[1]


@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(2)
    directory = Path(os.environ.get('POSE_MODEL_DIR', ROOT / 'ui_prmd_tcn_output'))
    if not directory.is_dir():
        raise FileNotFoundError(f'{directory}: retrain the combined notebook and extract ui_prmd_tcn_output.zip here. '
                                'For the previous angle model, set POSE_MODEL_DIR=ui_prmd_tcn_angles_output.')
    app.state.predictor = Predictor(directory)
    if app.state.predictor.feature_schema != 'positions_angles_v1':
        logging.warning('Previous model active (%s). Retrain the combined notebook for 54 input features.',
                        app.state.predictor.feature_schema)
    yield


app = FastAPI(title='Pose TCN local test', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=ROOT / 'web'), name='static')


@app.get('/')
def index():
    return FileResponse(ROOT / 'web/index.html')


@app.get('/api/config')
def config():
    p = app.state.predictor
    return {'version': 1, 'joints': p.metadata['joints'], 'mediapipe_indices': p.ids,
            'classes': p.metadata['classes'], 'window': p.window, 'fps': p.fps,
            'inference_interval_ms': 250, 'visibility_threshold': 0.0,
            'feature_schema': p.feature_schema, 'input_features': p.input_features}


@app.websocket('/ws/pose')
async def stream(ws: WebSocket):
    # Same-origin browser access; native clients (future Godot) may omit Origin.
    origin = ws.headers.get('origin')
    if origin and urlsplit(origin).netloc != ws.headers.get('host'):
        await ws.close(code=1008)
        return
    await ws.accept()
    session = PoseSession(ws.app.state.predictor)
    try:
        while True:
            raw = await ws.receive_text()
            if len(raw) > 16000:
                await ws.close(code=1009)
                return
            try:
                message = json.loads(raw)
                result = await asyncio.to_thread(session.push, message)
            except (ValueError, KeyError, TypeError) as error:
                result = {'status': 'invalid_input', 'message': str(error), 'progress': 0}
            except Exception:
                logging.exception('Inference failed')
                await ws.close(code=1011, reason='Inference failed')
                return
            await ws.send_json(result)
    except WebSocketDisconnect:
        pass
