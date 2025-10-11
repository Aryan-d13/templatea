#!/usr/bin/env python3
"""
Command-line entrypoint for the template rendering engine.

Examples:
    python cli/template_renderer.py --input input.mp4 --output out.mp4 --text "Hello world"
    python cli/template_renderer.py --input in.mp4 --output out.mp4 --choice-file workspace/03_choice/choice.txt --template-root templates/clean_ad
    python cli/template_renderer.py --input in.mp4 --output out.mp4 --text "Ad copy" --override text.color="#ff00ff"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from template_engine import TemplateEngine, TemplateRenderRequest  # noqa: E402

DEFAULT_TEMPLATE_DIR = ROOT / "templates" / "default"


def _parse_override(raw: str) -> Tuple[str, Any]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Overrides must be provided as key=value")
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("Override key cannot be empty")
    value = value.strip()
    if not value:
        return key, ""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    return key, parsed


def _collect_overrides(raw_items: Optional[Iterable[str]]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if not raw_items:
        return overrides
    for raw in raw_items:
        key, value = _parse_override(raw)
        overrides[key] = value
    return overrides


def _resolve_text(text: Optional[str], choice_file: Optional[Path]) -> str:
    if choice_file:
        try:
            return choice_file.read_text(encoding="utf-8-sig").strip()
        except FileNotFoundError:
            raise SystemExit(f"Choice file not found: {choice_file}")
    if text is not None:
        return text
    raise SystemExit("Either --text or --choice-file must be provided.")


def _format_preview(request: TemplateRenderRequest, overrides: Dict[str, Any]) -> str:
    payload = {
        "input": request.input_video_path,
        "output": request.output_video_path,
        "template_root": request.template_root,
        "text": request.text,
        "overrides": overrides,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a video using a Templatea template.")
    parser.add_argument("--input", required=True, dest="input_video", help="Source video to composite.")
    parser.add_argument("--output", required=True, dest="output_video", help="Destination MP4 path.")
    parser.add_argument("--text", help="Caption/heading text to render.")
    parser.add_argument("--choice-file", type=Path, help="Path to a text file containing caption content.")
    parser.add_argument(
        "--template-root",
        default=str(DEFAULT_TEMPLATE_DIR),
        help=f"Template folder (default: {DEFAULT_TEMPLATE_DIR}).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Shallow template override as key=value (supports dotted keys). Repeat for multiple overrides.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs and print payload without rendering.",
    )
    return parser


def run_cli(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input_video)
    output_path = Path(args.output_video)
    template_root = Path(args.template_root).resolve()
    overrides = _collect_overrides(args.override)
    text_value = _resolve_text(args.text, args.choice_file)

    if not input_path.exists() and not args.dry_run:
        parser.error(f"Input video not found: {input_path}")

    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if not template_root.exists():
        parser.error(f"Template root not found: {template_root}")

    request = TemplateRenderRequest(
        input_video_path=str(input_path),
        output_video_path=str(output_path),
        text=text_value,
        template_root=str(template_root),
        overrides=overrides or None,
    )

    if args.dry_run:
        print(_format_preview(request, overrides))
        return 0

    engine = TemplateEngine(request)
    success = engine.render()
    if not success:
        return 1
    print(f"[template_renderer] Rendered {output_path}")
    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
