#!/usr/bin/env python3
"""
Gemini OCR batch tester.
Takes a still frame from each video, sends it to Gemini, and prints the extracted ad copy.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Iterable, List

try:
    import cv2  # type: ignore
except ImportError as exc:
    raise SystemExit("OpenCV (cv2) is required to run this script") from exc

import requests

DEFAULT_PROMPT = (
    "Look at the image and extract only the main body text or commentary of the post "
    "that relates to the advertisement. Do not include the social media account name, "
    "handle, date/series information, or any header/branding text. Please provide only "
    "the extracted text as the response."
)

PERPLEXITY_SYSTEM_PROMPT = (
    "You are an advanced advertising copywriter and digital research assistant.\n"
    "You receive two caption variants for the same Instagram ad:\n"
    "1) OCR_EXTRACTED_CAPTION from a video frame\n"
    "2) DOWNLOADER_CAPTION sourced from the reel downloader's text file\n\n"
    "Review both carefully. If they differ, cross-check and rely on the more complete / accurate "
    "details while keeping all creative ideas consistent with the ad's intent. When facts are uncertain, "
    "choose language that remains truthful but still produces strong marketing hooks.\n\n"
    "If the caption references anything that benefits from live data (brands, events, trends, etc.), perform "
    "a real-time search before drafting the copies so they stay precise.\n\n"
    "Always generate exactly three alternative one-liner ad copies. Each line must be a punchy, ad-ready "
    "sentence of at most 15 words.\n\n"
    "Format your response strictly as a JSON object (no markdown, comments, or extra text) that matches:\n"
    '{\n  "one_liners": [\n    "First punchy one-liner.",\n    "Second option here.",\n'
    '    "Third option here."\n  ]\n}\n'
    "Even if the captions seem uncertain or incomplete, you must still return this JSON object with three "
    "plausible, high-quality options."
)

PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"


def load_env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def generate_ai_one_liners(
    ocr_caption: str,
    api_key: str,
    *,
    downloader_caption: str | None = None,
    timeout: int = 60,
    model: str = "sonar-pro",
) -> List[str]:
    headers = {
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    user_sections = [
        "OCR_EXTRACTED_CAPTION:",
        ocr_caption if ocr_caption else "[EMPTY]",
        "",
        "DOWNLOADER_CAPTION:",
        downloader_caption if downloader_caption else "[NOT AVAILABLE]",
    ]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": PERPLEXITY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "\n".join(user_sections).strip(),
            },
        ],
        "enable_search_classifier": True,
    }

    response = requests.post(PERPLEXITY_API_URL, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Perplexity response structure: {json.dumps(data, indent=2)}") from exc

    content = content.strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Perplexity response is not valid JSON: {content}") from exc

    one_liners = parsed.get("one_liners")
    if not isinstance(one_liners, list):
        raise RuntimeError(f"Perplexity response missing 'one_liners': {parsed}")

    cleaned = [str(line).strip() for line in one_liners if isinstance(line, str) and line.strip()]
    if len(cleaned) != 3:
        raise RuntimeError(f"Expected exactly 3 one-liners, received: {cleaned}")

    return cleaned


def gather_video_files(paths: Iterable[str], extensions: Iterable[str]) -> List[Path]:
    exts = {ext.lower().lstrip(".") for ext in extensions}
    files: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            if path.suffix.lower().lstrip(".") in exts:
                files.append(path)
        elif path.is_dir():
            for ext in exts:
                files.extend(path.rglob(f"*.{ext}"))
    return sorted({f.resolve() for f in files})


def extract_frame(video_path: Path, fraction: float) -> bytes:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has no frames: {video_path}")

    fraction = min(max(fraction, 0.0), 1.0)
    target_idx = int(total_frames * fraction)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)

    success, frame = cap.read()
    cap.release()
    if not success or frame is None:
        raise RuntimeError(f"Failed to capture frame at index {target_idx} in {video_path}")

    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError(f"Could not encode frame from {video_path} to JPEG")

    return buffer.tobytes()


def call_gemini(
    api_key: str,
    image_bytes: bytes,
    prompt: str,
    model: str = "gemini-2.5-flash",
    timeout: int = 90,
) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": encoded_image,
                        }
                    },
                ]
            }
        ]
    }

    response = requests.post(url, params={"key": api_key}, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    try:
        candidates = data["candidates"]
        first = candidates[0]
        parts = first["content"]["parts"]
        text = parts[0]["text"]
        return text.strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Gemini response: {json.dumps(data, indent=2)}") from exc


def process_video(path: Path, api_key: str, frame_fraction: float, prompt: str, model: str) -> dict:
    frame_bytes = extract_frame(path, frame_fraction)
    text = call_gemini(api_key, frame_bytes, prompt=prompt, model=model)
    return {
        "video": str(path),
        "text": text,
        "status": "success",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch test Gemini OCR on video stills.")
    parser.add_argument("inputs", nargs="+", help="Video files or directories to process")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=["mp4", "mov", "mkv", "avi"],
        help="File extensions to include when scanning directories",
    )
    parser.add_argument(
        "--frame-fraction",
        type=float,
        default=0.5,
        help="Fraction through the video to sample (0=start, 1=end)",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model to use",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Override prompt text",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        help="Optional path to save a JSON summary",
    )
    parser.add_argument(
        "--api-key",
        help="Override GEMINI_API_KEY environment variable",
    )
    parser.add_argument(
        "--perplexity-key",
        help="Override PERPLEXITY_API_KEY environment variable",
    )
    parser.add_argument(
        "--perplexity-model",
        default="sonar-pro",
        help="Perplexity model to use for AI one-liner generation",
    )
    parser.add_argument(
        "--perplexity-timeout",
        type=int,
        default=60,
        help="Timeout (seconds) for Perplexity requests",
    )
    parser.add_argument(
        "--disable-ai-copy",
        action="store_true",
        help="Skip generating AI recommended one-liner copies",
    )
    parser.add_argument(
        "--caption-dir",
        type=Path,
        help="Optional directory containing manual caption text files (one per video)",
    )
    parser.add_argument(
        "--caption-extension",
        default=".txt",
        help="Extension to look for when loading manual caption files (default: .txt)",
    )

    args = parser.parse_args()
    load_env_file()

    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY (environment or .env) or use --api-key to provide a Gemini API key")

    perplexity_api_key = (args.perplexity_key or os.getenv("PERPLEXITY_API_KEY") or "").strip()
    ai_copy_enabled = not args.disable_ai_copy
    if ai_copy_enabled and not perplexity_api_key:
        print("[WARN] PERPLEXITY_API_KEY not provided. Skipping AI recommended copies.")
        ai_copy_enabled = False
    elif ai_copy_enabled and args.perplexity_timeout <= 0:
        print("[WARN] Invalid Perplexity timeout. Skipping AI recommended copies.")
        ai_copy_enabled = False

    caption_dir = args.caption_dir.resolve() if args.caption_dir else None
    caption_extension = args.caption_extension.strip()
    if caption_extension and not caption_extension.startswith("."):
        caption_extension = f".{caption_extension}"
    elif not caption_extension:
        caption_extension = ".txt"

    videos = gather_video_files(args.inputs, args.extensions)
    if not videos:
        raise SystemExit("No video files found to process")

    results = []
    for video in videos:
        print(f"[INFO] Processing {video}")
        try:
            result = process_video(video, api_key, args.frame_fraction, args.prompt, args.model)
            print(f"[OK] Extracted text:\n{result['text']}\n")

            ocr_caption = result["text"].strip()
            caption_text = ocr_caption
            caption_source = "ocr"
            manual_caption_file: Path | None = None
            manual_caption_text = ""
            manual_candidates = []

            # Future downloader pipeline: these sidecar caption files will be emitted by the reel downloader.
            # For now we rely on the manually supplied test files to simulate that behavior.
            if caption_dir:
                manual_candidates.append(caption_dir / f"{video.stem}{caption_extension}")
            manual_candidates.append(video.with_suffix(caption_extension))

            for candidate in manual_candidates:
                try:
                    if candidate.exists() and candidate.is_file():
                        manual_caption_file = candidate
                        manual_caption_text = candidate.read_text(encoding="utf-8").strip()
                        break
                except OSError as os_err:
                    print(f"[WARN] Unable to read manual caption file {candidate}: {os_err}")

            if manual_caption_file:
                if manual_caption_text:
                    caption_text = manual_caption_text
                    caption_source = "manual_file"
                    print(f"[INFO] Using manual caption from {manual_caption_file}")
                else:
                    print(f"[WARN] Manual caption file {manual_caption_file} is empty. Falling back to OCR text.")

            result["caption_source"] = caption_source
            result["manual_caption_file"] = str(manual_caption_file) if manual_caption_file else None
            result["manual_caption"] = manual_caption_text if manual_caption_text else None
            # Preserve OCR text even when a downloader caption overrides it for Perplexity.
            result["ocr_caption"] = ocr_caption if ocr_caption else None
            result["effective_caption"] = caption_text

            if ai_copy_enabled:
                if not (ocr_caption or manual_caption_text):
                    result["ai_copy_status"] = "skipped"
                    result.setdefault("ai_recommended_copies", [])
                    print("[WARN] No caption text available (OCR and downloader empty). Skipping AI recommended copies.\n")
                else:
                    try:
                        ai_one_liners = generate_ai_one_liners(
                            ocr_caption,
                            perplexity_api_key,
                            downloader_caption=manual_caption_text if manual_caption_text else None,
                            timeout=args.perplexity_timeout,
                            model=args.perplexity_model,
                        )
                        result["ai_copy_status"] = "success"
                        result["ai_recommended_copies"] = ai_one_liners
                        print("[OK] AI recommended copies:")
                        for idx, line in enumerate(ai_one_liners, start=1):
                            print(f"  {idx}. {line}")
                        print()
                    except Exception as ai_exc:
                        result["ai_copy_status"] = "error"
                        result["ai_copy_error"] = str(ai_exc)
                        result.setdefault("ai_recommended_copies", [])
                        print(f"[ERROR] Failed to generate AI recommended copies: {ai_exc}\n")
            else:
                result.setdefault("ai_copy_status", "skipped")
                result.setdefault("ai_recommended_copies", [])

        except Exception as exc:  # pylint: disable=broad-except
            result = {
                "video": str(video),
                "status": "error",
                "error": str(exc),
            }
            print(f"[ERROR] {video}: {exc}\n")
        results.append(result)

    if args.json_report:args.json_report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_report, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved report to {args.json_report}")


if __name__ == "__main__":
    main()
