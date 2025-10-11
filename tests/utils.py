import importlib
from pathlib import Path

import pytest

pytest.importorskip('fastapi')
from fastapi.testclient import TestClient


def prepare_app(tmp_path, monkeypatch):
    """Configure environment and return (APP, storage_module)."""
    workspace_dir = tmp_path / 'workspace'
    workspace_dir.mkdir()
    db_file = tmp_path / 'db.sqlite'
    monkeypatch.setenv('WORKSPACE_BASE', str(workspace_dir))
    monkeypatch.setenv('WORKSPACE_INDEX_DB', str(db_file))
    templates_dir = Path(__file__).resolve().parents[1] / 'templates'
    monkeypatch.setenv('TEMPLATE_DIR', str(templates_dir))
    monkeypatch.delenv('API_KEY', raising=False)

    import api.storage as storage
    import api.template_registry as template_registry

    storage = importlib.reload(storage)
    template_registry = importlib.reload(template_registry)

    import api.app as app_module

    app_module.storage = storage
    app_module.template_registry = template_registry
    app_module = importlib.reload(app_module)

    return app_module.APP, storage


def get_test_client(app):
    return TestClient(app)
