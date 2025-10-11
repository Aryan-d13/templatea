from tests.utils import prepare_app, get_test_client


def test_range_file_response(tmp_path, monkeypatch):
    app, storage = prepare_app(tmp_path, monkeypatch)
    ws_dir = storage.resolve_workspace('WSRANGE')
    (ws_dir / '04_render').mkdir(parents=True)
    final_path = ws_dir / '04_render/final_1080x1920.mp4'
    final_path.write_bytes(b'0123456789')

    storage.index_workspace(ws_dir)

    with get_test_client(app) as client:
        resp = client.get('/api/v1/workspaces/WSRANGE/files/final', headers={'Range': 'bytes=0-3'})
        assert resp.status_code == 206
        assert resp.content == b'0123'
        assert resp.headers['Content-Range'].startswith('bytes 0-3/')


def test_logs_listing(tmp_path, monkeypatch):
    app, storage = prepare_app(tmp_path, monkeypatch)
    ws_dir = storage.resolve_workspace('WSLOG')
    (ws_dir / 'logs').mkdir(parents=True)
    (ws_dir / 'logs/ocr.log').write_text('hello', encoding='utf-8')

    storage.index_workspace(ws_dir)

    with get_test_client(app) as client:
        resp = client.get('/api/v1/workspaces/WSLOG/files/logs')
        assert resp.status_code == 200
        payload = resp.json()
        assert payload[0]['name'] == 'ocr.log'
