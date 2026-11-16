"""Embed the canonical feature source in the standalone Colab notebook."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sync():
    path = ROOT / 'ui_prmd_tcn_colab.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    source = (ROOT / 'pose_features.py').read_text(encoding='utf-8')
    cell = next(c for c in notebook['cells'] if ''.join(c['source']).startswith('PREPROCESS_SOURCE ='))
    code = ('PREPROCESS_SOURCE = ' + repr(source) + '\n'
            'exec(PREPROCESS_SOURCE)\n'
            "CFG['input_features'] = len(FEATURE_NAMES)\n"
            "(OUT/'preprocess.py').write_text(PREPROCESS_SOURCE, encoding='utf-8')\n")
    cell['source'] = code.splitlines(keepends=True)
    cell['execution_count'], cell['outputs'] = None, []
    recording_source = (ROOT / 'recording_data.py').read_text(encoding='utf-8')
    recording_cell = next((c for c in notebook['cells'] if ''.join(c['source']).startswith('RECORDING_SOURCE =')), None)
    if recording_cell is not None:
        recording_cell['source'] = ('RECORDING_SOURCE = '+repr(recording_source)+'\nexec(RECORDING_SOURCE)\n'+"assert CODES == RECORDING_CODES, 'Notebook and webcam class orders must match'\n").splitlines(keepends=True)
        recording_cell['execution_count'], recording_cell['outputs'] = None, []
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    sync()
