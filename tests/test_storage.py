import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_storage(tmp_path, monkeypatch):
    workspace_dir = tmp_path / 'workspace'
    db_file = tmp_path / 'db' / 'ws.sqlite'
    monkeypatch.setenv('WORKSPACE_BASE', str(workspace_dir))
    monkeypatch.setenv('WORKSPACE_INDEX_DB', str(db_file))
    import api.storage as storage

    return importlib.reload(storage)


def test_index_and_list_workspaces(tmp_path, monkeypatch):
    storage = load_storage(tmp_path, monkeypatch)

    ws_dir = storage.WORKSPACE_BASE / 'WS123'
    (ws_dir / '00_raw').mkdir(parents=True)
    (ws_dir / '01_detector').mkdir()
    (ws_dir / '02_ocr').mkdir()
    (ws_dir / '03_choice').mkdir()
    (ws_dir / '04_render').mkdir()

    meta = {'id': 'WS123', 'created_at': '2025-10-09T00:00:00Z', 'source': 'test'}
    storage.write_meta('WS123', meta)

    detector_status = {'status': 'success', 'ts': '2025-10-09T00:01:00Z', 'error': None, 'retries': 0}
    (ws_dir / '01_detector.status').write_text(json.dumps(detector_status), encoding='utf-8')

    storage.index_workspace(ws_dir)

    rows = storage.list_workspaces()
    assert len(rows) == 1
    row = rows[0]
    assert row['id'] == 'WS123'
    assert row['meta']['source'] == 'test'
    assert row['status']['01_detector']['status'] == 'success'

    ws = storage.get_workspace('WS123')
    assert ws is not None
    assert ws['id'] == 'WS123'


def test_list_workspace_files(tmp_path, monkeypatch):
    storage = load_storage(tmp_path, monkeypatch)
    ws_dir = storage.WORKSPACE_BASE / 'WS999'
    (ws_dir / '00_raw').mkdir(parents=True)
    (ws_dir / '00_raw/raw_source.mp4').write_bytes(b'video')

    files = storage.list_workspace_files(ws_dir)
    assert files['raw']['exists'] is True
    assert isinstance(files['raw']['path'], Path)
    assert files['final']['exists'] is False
