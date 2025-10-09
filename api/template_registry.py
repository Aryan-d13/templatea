
"""
Template registry that loads JSON manifests for templates and exposes renderer helpers.
Supports both legacy flat manifests:
    templates/*.json
and the new canonical layout:
    templates/<page_name>/template.json (+ assets/ ... inside each folder)
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = Path(os.getenv("TEMPLATE_DIR", BASE_DIR / "templates")).resolve()


class TemplateRegistryError(RuntimeError):
    pass


class TemplateNotFound(TemplateRegistryError):
    pass


def _load_manifest(path: Path) -> Dict[str, Any]:
    """Load and validate a single JSON manifest."""
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data: Dict[str, Any] = json.load(handle)
    except Exception as e:
        raise TemplateRegistryError(f"Failed to parse manifest {path}: {e}") from e

    # Always remember where this came from
    data["_path"] = str(path)

    # Validate minimal keys
    if "id" not in data:
        # Be nice: if missing, try using folder name as id
        folder = path.parent.name
        if folder:
            data["id"] = folder
        else:
            raise TemplateRegistryError(f"Template manifest {path.name} missing 'id'")

    if "module" not in data:
        raise TemplateRegistryError(f"Template manifest {path.name} missing 'module'")

    module_name = data["module"]
    if importlib.util.find_spec(module_name) is None:
        raise TemplateRegistryError(f"Template module '{module_name}' not found for {data['id']}")

    return data


def _iter_manifests() -> List[Dict[str, Any]]:
    """Yield all manifests from both the legacy and the new folder-per-template layout."""
    if not TEMPLATES_DIR.exists():
        return []

    manifests: List[Dict[str, Any]] = []

    # 1) New layout: templates/<page_name>/template.json
    for child in sorted(TEMPLATES_DIR.iterdir()):
        if child.is_dir():
            manifest_path = child / "template.json"
            # Also allow "manifest.json" as a backup name
            if not manifest_path.exists():
                alt = child / "manifest.json"
                if alt.exists():
                    manifest_path = alt
            if manifest_path.exists():
                manifests.append(_load_manifest(manifest_path))

    # 2) Legacy fallback: templates/*.json (but skip a root-level template.json if present)
    for file in sorted(TEMPLATES_DIR.glob("*.json")):
        if file.name.lower() == "template.json":
            # If someone dropped a single template.json at root, ignore it to avoid confusion.
            continue
        manifests.append(_load_manifest(file))

    return manifests


def list_templates() -> List[Dict[str, Any]]:
    """Return all template manifests."""
    return _iter_manifests()


def get_template(template_id: str) -> Dict[str, Any]:
    for manifest in _iter_manifests():
        if manifest.get("id") == template_id:
            return manifest
    raise TemplateNotFound(f"Template '{template_id}' not found")


def _resolve_renderer_callable(module, manifest: Dict[str, Any]) -> Callable:
    """Pick a renderer entrypoint from a module."""
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


def get_renderer_func(template_id: str) -> Callable[[str, str, str, Optional[Dict[str, Any]]], bool]:
    """Return a callable that runs the template's renderer with smart defaults/overrides.

    The returned function has the signature:
        runner(input_video, output_video, text, options=None) -> bool
    where 'options' can contain kwargs for the renderer or dotted paths to override config.
    """
    manifest = get_template(template_id)
    module = importlib.import_module(manifest["module"])
    renderer_callable = _resolve_renderer_callable(module, manifest)

    # Defaults from manifest schema
    defaults: Dict[str, Any] = dict(manifest.get("schema", {}).get("defaults", {}))

    # Optional module-level base config
    base_config = None
    if hasattr(module, "TEMPLATE_CONFIG"):
        try:
            base_config = copy.deepcopy(getattr(module, "TEMPLATE_CONFIG"))
        except Exception:
            base_config = getattr(module, "TEMPLATE_CONFIG")

    renderer_signature = inspect.signature(renderer_callable)
    renderer_params = renderer_signature.parameters

    def _apply_override(target: Dict[str, Any], path: str, value: Any) -> None:
        parts = path.split(".")
        node = target
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value

    def _runner(input_video: str, output_video: str, text: str, options: Optional[Dict[str, Any]] = None) -> bool:
        # Merge defaults with user options
        merged_options: Dict[str, Any] = dict(defaults)
        if options:
            merged_options.update(options)

        config_payload = None
        extra_kwargs: Dict[str, Any] = {}

        # Pull out dotted-path overrides (e.g., "audio.volume": 0.8)
        config_overrides = {k: v for k, v in merged_options.items() if "." in k}
        for key in config_overrides:
            merged_options.pop(key, None)

        # Start from base config when available, or create a minimal one if overrides exist
        if base_config is not None:
            try:
                config_payload = copy.deepcopy(base_config)
            except Exception:
                config_payload = base_config
        elif config_overrides:
            config_payload = {}

        # Apply dotted overrides
        if config_payload is not None:
            for path, value in config_overrides.items():
                _apply_override(config_payload, path, value)

        # Pass-through renderer kwargs that match its signature
        for key, value in merged_options.items():
            if key in renderer_params:
                extra_kwargs[key] = value

        call_args = [input_video, output_video, text]
        call_kwargs = dict(extra_kwargs)

        # Attach config payload if the renderer advertises it
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
