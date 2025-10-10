import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_registry(tmp_path, monkeypatch):
    module_dir = tmp_path / 'modules'
    module_dir.mkdir()
    (module_dir / 'dummy_template.py').write_text(
        'def render(input_video, output_video, text, options):\n'
        '    return True\n'
    , encoding='utf-8')
    monkeypatch.syspath_prepend(str(module_dir))

    templates_dir = tmp_path / 'templates'
    templates_dir.mkdir()
    manifest = {
        'id': 'dummy',
        'name': 'Dummy Template',
        'module': 'dummy_template',
        'entrypoint': 'render',
        'description': 'Test manifest',
        'preview': None,
        'schema': {'defaults': {'foo': 'bar'}},
    }
    (templates_dir / 'dummy.json').write_text(json.dumps(manifest), encoding='utf-8')

    monkeypatch.setenv('TEMPLATE_DIR', str(templates_dir))
    import api.template_registry as template_registry

    return importlib.reload(template_registry), manifest


def test_list_templates(tmp_path, monkeypatch):
    registry, manifest = load_registry(tmp_path, monkeypatch)
    templates = registry.list_templates()
    assert len(templates) == 1
    assert templates[0]['id'] == manifest['id']


def test_get_renderer_func(tmp_path, monkeypatch):
    registry, manifest = load_registry(tmp_path, monkeypatch)
    renderer = registry.get_renderer_func(manifest['id'])
    assert renderer('in.mp4', 'out.mp4', 'hello', {'foo': 'override'}) is True
