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

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or use --api-key to provide a Gemini API key")

    videos = gather_video_files(args.inputs, args.extensions)
    if not videos:
        raise SystemExit("No video files found to process")

    results = []
    for video in videos:
        print(f"[INFO] Processing {video}")
        try:
            result = process_video(video, api_key, args.frame_fraction, args.prompt, args.model)
            print(f"[OK] Extracted text:\n{result['text']}\n")
        except Exception as exc:  # pylint: disable=broad-except
            result = {
                "video": str(video),
                "status": "error",
                "error": str(exc),
            }
            print(f"[ERROR] {video}: {exc}\n")
        results.append(result)

    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[INFO] Saved report to {args.json_report}")


if __name__ == "__main__":
    main()
