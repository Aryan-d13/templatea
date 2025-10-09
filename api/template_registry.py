"""
Template registry that loads JSON manifests and exposes renderer helpers.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

# from pathlib import Path
# find repo root (two levels up from this file: api/template_registry.py -> repo root)
TEMPLATES_DIR = (Path(__file__).resolve().parent.parent / "templates").resolve()

def _load_manifest(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


class TemplateRegistryError(RuntimeError):
    pass


class TemplateNotFound(TemplateRegistryError):
    pass


BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = Path(os.getenv("TEMPLATE_DIR", BASE_DIR / "templates"))


def _load_manifest(path: Path) -> Dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if "id" not in data:
        raise TemplateRegistryError(f"Template manifest {path.name} missing 'id'")
    if "module" not in data:
        raise TemplateRegistryError(f"Template manifest {path.name} missing 'module'")
    module_name = data["module"]
    if importlib.util.find_spec(module_name) is None:
        raise TemplateRegistryError(f"Template module '{module_name}' not found for {data['id']}")
    data["_path"] = str(path)
    return data


def _iter_manifests() -> List[Dict]:
    if not TEMPLATES_DIR.exists():
        return []
    manifests: List[Dict] = []
    for file in sorted(TEMPLATES_DIR.glob("*.json")):
        manifests.append(_load_manifest(file))
    return manifests


def list_templates() -> List[Dict]:
    return _iter_manifests()


def get_template(template_id: str) -> Dict:
    for manifest in _iter_manifests():
        if manifest["id"] == template_id:
            return manifest
    raise TemplateNotFound(f"Template '{template_id}' not found")


def _resolve_renderer_callable(module, manifest: Dict) -> Callable:
    preferred = manifest.get("entrypoint")
    fallback_names = ["render", "process", "process_marketingspots_template", "run"]
    candidates = [preferred] if preferred else []
    candidates.extend(fallback_names)
    for name in candidates:
        if not name:
            continue
        func = getattr(module, name, None)
        if callable(func):
            return func
    raise TemplateRegistryError(f"No callable renderer found in module '{module.__name__}'")


def get_renderer_func(template_id: str) -> Callable[[str, str, str, Optional[Dict]], bool]:
    manifest = get_template(template_id)
    module = importlib.import_module(manifest["module"])
    renderer_callable = _resolve_renderer_callable(module, manifest)
    defaults = dict(manifest.get("schema", {}).get("defaults", {}))
    base_config = None
    if hasattr(module, "TEMPLATE_CONFIG"):
        try:
            base_config = copy.deepcopy(getattr(module, "TEMPLATE_CONFIG"))
        except Exception:
            base_config = getattr(module, "TEMPLATE_CONFIG")

    renderer_signature = inspect.signature(renderer_callable)
    renderer_params = renderer_signature.parameters

    def _apply_override(target: Dict, path: str, value) -> None:
        parts = path.split(".")
        node = target
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value

    def _runner(input_video: str, output_video: str, text: str, options: Optional[Dict] = None) -> bool:
        merged_options = dict(defaults)
        if options:
            merged_options.update(options)

        config_payload = None
        extra_kwargs: Dict[str, object] = {}

        config_overrides = {k: v for k, v in merged_options.items() if "." in k}
        for key in config_overrides:
            merged_options.pop(key, None)

        if base_config is not None:
            try:
                config_payload = copy.deepcopy(base_config)
            except Exception:
                config_payload = base_config
        elif config_overrides:
            config_payload = {}

        if config_payload is not None:
            for path, value in config_overrides.items():
                _apply_override(config_payload, path, value)

        for key, value in merged_options.items():
            if key in renderer_params:
                extra_kwargs[key] = value

        call_args = [input_video, output_video, text]
        call_kwargs = dict(extra_kwargs)

        if config_payload is not None:
            if "config" in renderer_params:
                call_kwargs["config"] = config_payload
            elif "options" in renderer_params:
                call_kwargs["options"] = config_payload
            else:
                call_args.append(config_payload)

        result = renderer_callable(*call_args, **call_kwargs)
        return bool(result) if result is not None else True

    return _runner


__all__ = ["TemplateRegistryError", "TemplateNotFound", "list_templates", "get_template", "get_renderer_func"]
