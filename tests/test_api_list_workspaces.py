import json

from tests.utils import prepare_app, get_test_client


def test_list_and_get_workspace(tmp_path, monkeypatch):
    app, storage = prepare_app(tmp_path, monkeypatch)
    ws_dir = storage.resolve_workspace('WS123')
    (ws_dir / '00_raw').mkdir(parents=True)
    (ws_dir / '00_raw/raw_thumb.jpg').write_bytes(b'data')
    (ws_dir / '04_render').mkdir(parents=True)
    (ws_dir / '04_render/final_1080x1920.mp4').write_bytes(b'video-data')

    meta = {'id': 'WS123', 'created_at': '2025-10-09T00:00:00Z'}
    storage.write_meta('WS123', meta)

    status_payload = {'status': 'success', 'ts': '2025-10-09T00:01:00Z', 'error': None, 'retries': 0}
    (ws_dir / '01_detector.status').write_text(json.dumps(status_payload), encoding='utf-8')

    storage.index_workspace(ws_dir)

    with get_test_client(app) as client:
        resp = client.get('/api/v1/workspaces')
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]['id'] == 'WS123'
        assert data[0]['status']['01_detector'] == 'success'
        assert data[0]['files']['thumb']['exists'] is True

        detail = client.get('/api/v1/workspaces/WS123')
        assert detail.status_code == 200
        detail_json = detail.json()
        assert detail_json['files']['final']['exists'] is True
        assert detail_json['status']['01_detector'] == 'success'
