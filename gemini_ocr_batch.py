#!/usr/bin/env python3
"""
Gemini OCR batch tester with Groq validation layer - FIXED VERSION
Key fixes:
1. Proper HTTP header casing (Content-Type, Authorization)
2. Better error handling for Groq validator error objects
3. Enhanced debug logging
"""

from __future__ import annotations
import traceback
import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List

from builtins import print as _builtin_print


def _safe_print(*args, **kwargs):
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe_args = []
    for arg in args:
        if isinstance(arg, str):
            safe_args.append(arg.encode(encoding, errors="backslashreplace").decode(encoding, errors="ignore"))
        else:
            safe_args.append(arg)
    _builtin_print(*safe_args, **kwargs)


print = _safe_print  # type: ignore

try:
    import cv2  # type: ignore
except ImportError as exc:
    raise SystemExit("OpenCV (cv2) is required to run this script") from exc

import requests


def _normalize_dashes(text: str) -> str:
    """Replace em/en dashes with a comma-space combo for downstream compatibility."""
    if not text:
        return text
    return text.replace("\u2014", ", ").replace("\u2013", ", ")

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
    "You MUST respond with ONLY a valid JSON object. No markdown code fences, no ```json blocks, "
    "no commentary, no extra text before or after. Just the raw JSON object.\n\n"
    "The JSON structure must be:\n"
    '{\n  "one_liners": [\n    "First punchy one-liner.",\n    "Second option here.",\n'
    '    "Third option here."\n  ]\n}\n\n'
    "Even if the captions seem uncertain or incomplete, you must still return this JSON object with three "
    "plausible, high-quality options. Output ONLY the JSON object, nothing else."
)

GROQ_JSON_EXTRACTOR_PROMPT = '''You are a strict **JSON extractor & repairer**.
Input: a single arbitrary string (may contain valid JSON, broken JSON, or text around JSON).
Output: **exactly one** JSON value (object or array) – nothing else. No markdown, no code fences, no explanation, no extra characters before or after the JSON (no leading/trailing bytes, not even a newline). The JSON must be 100% RFC-validated (double quotes for strings, no comments, no trailing commas).

Rules – follow them exactly:

1. If the input contains one valid JSON value, output it (canonicalize: double quotes, valid booleans/null).
2. If the input contains multiple JSON values/objects, output a JSON array of those values in the order found.
3. If the input is messy, attempt to **repair** only obvious, local issues:
   * Convert single-quoted strings to double-quoted strings.
   * Quote unquoted object keys.
   * Remove JavaScript-style comments (`//` and `/* */`).
   * Remove trailing commas.
   * Fix obvious missing commas between members/array items when unambiguous.
   * Fix common escape issues inside strings.
   * Convert `True`/`False`/`None` (Python-style) to `true`/`false`/`null`.
   * Preserve numeric formats (integers/floats), but do not invent numbers.
   * Preserve the original text of string values except for necessary quote/escape fixes.
   * Do not attempt speculative structural changes (no adding nested objects/keys that are not implied). Be conservative.
4. If you can confidently produce repaired JSON, output it.
5. If you cannot extract or repair any JSON, output exactly this JSON object (replace the value of `original` with the full original input string, escaped as a JSON string):
   {"error":"no json could be extracted","original": "<full original input here>"}
6. If you must report partial success (some pieces parseable, others not), return an object:
   {"extracted": <value-or-array>, "unparsed":"<remaining unparseable text if any>"}
   (Only use this when necessary; prefer returning the parsed value or the `error` object above.)
7. Never output any extra keys except `error`, `original`, `extracted`, or `unparsed` when following rule 5–6.
8. ALWAYS ensure the output is valid JSON (no trailing commas, proper quoting, proper true/false/null).
9. Output must be the **raw JSON text only** – nothing else, not even a single extra space or newline outside the JSON.
10.Under no circumstances should you use em dashes (—) or en dashes (–) in your output.
11.Review and revise your response to ensure em dashes never appear in the final output for any reason.

Now parse and respond with the single, valid JSON value for the following input string (do not repeat the input, do not add commentary):

{input_text}'''

PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


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


def extract_json_from_string(text: str) -> dict | list | None:
    """
    Sophisticated JSON extractor that handles markdown code blocks and malformed JSON.
    Returns parsed JSON object/array or None if extraction fails.
    """
    text = text.strip()
    
    # Remove markdown code fences
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'`(.*?)`',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            text = match.group(1).strip()
            break
    
    # Try direct parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object or array boundaries
    json_patterns = [
        (r'\{.*\}', re.DOTALL),  # Find outermost object
        (r'\[.*\]', re.DOTALL),  # Find outermost array
    ]
    
    for pattern, flags in json_patterns:
        match = re.search(pattern, text, flags)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
    
    return None


def call_groq_validator(
    api_key: str,
    input_text: str,
    model: str = "OpenAI/gpt-oss-120b",
    timeout: int = 60,
) -> dict:
    """
    Calls Groq API to validate and extract/repair JSON from input text.
    Returns a dict with 'status', 'result', and optional 'error' keys.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    prompt = GROQ_JSON_EXTRACTOR_PROMPT.format(input_text=input_text)
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        
        content = data["choices"][0]["message"]["content"].strip()
        
        # Try to parse the Groq response
        parsed = extract_json_from_string(content)
        if parsed is None:
            # Fallback: try direct parse
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                return {
                    "status": "error",
                    "error": f"Groq returned non-JSON content: {content[:200]}",
                }

        # If Groq returned a primitive (string/number/bool/null) treat it as an error
        if not isinstance(parsed, (dict, list)):
            return {
                "status": "error",
                "error": "Groq returned a non-object JSON value",
                "raw": parsed,
            }

        return {
            "status": "success",
            "result": parsed,
        }

        
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
        }


def call_groq_direct_generation(
    ocr_caption: str,
    api_key: str,
    *,
    downloader_caption: str | None = None,
    timeout: int = 60,
    model: str = "OpenAI/gpt-oss-120b",
) -> List[str]:
    """
    Calls Groq directly with the same prompt as Perplexity (fallback when Perplexity fails).
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
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
            {"role": "user", "content": "\n".join(user_sections).strip()},
        ],
        "temperature": 0.7,
    }
    
    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    
    content = data["choices"][0]["message"]["content"].strip()
    
    # Extract JSON
    parsed = extract_json_from_string(content)
    if parsed is None:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Groq response is not valid JSON: {content!r}") from exc

    # If parsed is a primitive (e.g. "error"), give explicit, helpful error
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Groq returned non-object JSON: {repr(parsed)}")

    one_liners = parsed.get("one_liners")
    if not isinstance(one_liners, list):
        raise RuntimeError(f"Groq response missing or bad 'one_liners': {repr(parsed)}")

    
    cleaned = [str(line).strip() for line in one_liners if isinstance(line, str) and line.strip()]
    if len(cleaned) != 3:
        raise RuntimeError(f"Expected exactly 3 one-liners, received: {cleaned}")
    
    return cleaned


def generate_ai_one_liners(
    ocr_caption: str,
    perplexity_api_key: str,    # kept in signature for compatibility; ignored now
    groq_api_key: str,
    *,
    downloader_caption: str | None = None,
    timeout: int = 60,
    perplexity_model: str = "sonar-pro",  # unused, kept for compatibility
    groq_model: str = "OpenAI/gpt-oss-120b",
) -> dict:
    """
    Groq-only AI one-liners generation (Perplexity commented out).
    Returns dict with 'one_liners', 'source', and 'validation_notes'.
    """
    result = {
        "one_liners": [],
        "source": "unknown",
        "validation_notes": [],
    }

    # Ensure we have Groq API key
    if not groq_api_key:
        result["validation_notes"].append("GROQ_API_KEY missing; AI generation disabled.")
        raise RuntimeError("GROQ_API_KEY required for AI generation")

    # Prepare the user-friendly prompt for Groq:
    # It receives both OCR-extracted caption and the downloader caption (if any).
    groq_system = (
        "You are an expert advertising copywriter and JSON-output validator. "
        "You will receive two inputs: OCR_EXTRACTED_CAPTION (what was read from the video frame) "
        "and DOWNLOADER_CAPTION (the caption text provided by the downloader, if any). "
        "Compare them and produce **three** short, punchy one-line ad copy suggestions that are "
        "marketing-ready. Be creative but do not hallucinate facts. If something is unclear, "
        "prefer neutral phrasing. Output EXACTLY one valid JSON object (no extra text) with shape:\n"
        '{"one_liners": ["...", "...", "..."]}\n'
        "Use only double-quoted JSON strings and return a valid RFC-compliant JSON object."
    )

    user_payload_parts = []
    user_payload_parts.append("OCR_EXTRACTED_CAPTION:")
    user_payload_parts.append(ocr_caption if ocr_caption else "")
    user_payload_parts.append("")
    user_payload_parts.append("DOWNLOADER_CAPTION:")
    user_payload_parts.append(downloader_caption if downloader_caption else "")

    try:
        # Call Groq direct generation (this is the single allowed pathway now)
        one_liners = call_groq_direct_generation(
            ocr_caption,
            groq_api_key,
            downloader_caption=downloader_caption,
            timeout=timeout,
            model=groq_model,
        )
        result["one_liners"] = one_liners
        result["source"] = "groq_direct"
        result["validation_notes"].append("Groq direct generation successful")
        return result

    except Exception as groq_exc:
        result["validation_notes"].append(f"Groq direct generation failed: {str(groq_exc)}")
        # final fallback: return small safe defaults (so pipeline doesn't break)
        if ocr_caption:
            snippet = ocr_caption.strip().splitlines()[0][:120]
            fallback = [
                snippet[:60],
                snippet[:40] + ("…" if len(snippet) > 40 else ""),
                "Watch this ad now!"
            ]
        else:
            fallback = ["Watch this now!", "Don’t miss this!", "Tap to see why!"]

        result["one_liners"] = fallback
        result["source"] = "local_fallback"
        result["validation_notes"].append("Returning local fallback one-liners")
        return result



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
    parser = argparse.ArgumentParser(description="Batch test Gemini OCR on video stills with Groq validation.")
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
        "--groq-key",
        help="Override GROQ_API_KEY environment variable",
    )
    parser.add_argument(
        "--groq-model",
        default="OpenAI/gpt-oss-120b",
        help="Groq model to use for validation and fallback generation",
    )
    parser.add_argument(
        "--perplexity-timeout",
        type=int,
        default=60,
        help="Timeout (seconds) for Perplexity requests",
    )
    parser.add_argument(
        "--groq-timeout",
        type=int,
        default=60,
        help="Timeout (seconds) for Groq requests",
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
    groq_api_key = os.getenv("GROQ_API_KEY") or args.groq_key or ""
    
    ai_copy_enabled = not args.disable_ai_copy
    if ai_copy_enabled and not perplexity_api_key:
        print("[WARN] PERPLEXITY_API_KEY not provided. Will use Groq direct generation only.")
    if ai_copy_enabled and not groq_api_key:
        print("[ERROR] GROQ_API_KEY is required for AI copy generation.")
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

            ocr_caption = _normalize_dashes(result["text"].strip())
            caption_text = ocr_caption
            caption_source = "ocr"
            manual_caption_file: Path | None = None
            manual_caption_text = ""
            manual_candidates = []

            if caption_dir:
                manual_candidates.append(caption_dir / f"{video.stem}{caption_extension}")
            manual_candidates.append(video.with_suffix(caption_extension))

            for candidate in manual_candidates:
                try:
                    if candidate.exists() and candidate.is_file():
                        manual_caption_file = candidate
                        manual_caption_text = _normalize_dashes(candidate.read_text(encoding="utf-8").strip())
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

            caption_text = _normalize_dashes(caption_text)
            result["caption_source"] = caption_source
            result["manual_caption_file"] = str(manual_caption_file) if manual_caption_file else None
            result["manual_caption"] = manual_caption_text if manual_caption_text else None
            result["ocr_caption"] = ocr_caption if ocr_caption else None
            result["effective_caption"] = caption_text

            if ai_copy_enabled:
                if not (ocr_caption or manual_caption_text):
                    result["ai_copy_status"] = "skipped"
                    result.setdefault("ai_recommended_copies", [])
                    result.setdefault("ai_copy_source", None)
                    result.setdefault("ai_validation_notes", [])
                    print("[WARN] No caption text available (OCR and downloader empty). Skipping AI recommended copies.\n")
                else:
                    try:
                        ai_result = generate_ai_one_liners(
                            ocr_caption,
                            perplexity_api_key,
                            groq_api_key,
                            downloader_caption=manual_caption_text if manual_caption_text else None,
                            timeout=args.perplexity_timeout,
                            perplexity_model=args.perplexity_model,
                            groq_model=args.groq_model,
                        )
                        sanitized_copies = [
                            _normalize_dashes(line) for line in ai_result["one_liners"]
                        ]
                        result["ai_copy_status"] = "success"
                        result["ai_recommended_copies"] = sanitized_copies
                        result["ai_copy_source"] = ai_result["source"]
                        result["ai_validation_notes"] = ai_result["validation_notes"]
                        
                        print(f"[OK] AI recommended copies (source: {ai_result['source']}):")
                        for idx, line in enumerate(sanitized_copies, start=1):
                            print(f"  {idx}. {line}")
                        if ai_result["validation_notes"]:
                            print(f"[INFO] Validation notes:")
                            for note in ai_result["validation_notes"]:
                                print(f"  - {note}")
                        print()
                    
                    except Exception as ai_exc:
                        result["ai_copy_status"] = "error"
                        result["ai_copy_error"] = str(ai_exc)
                        result.setdefault("ai_recommended_copies", [])
                        result.setdefault("ai_copy_source", None)
                        result.setdefault("ai_validation_notes", [])
                        print(f"[ERROR] Failed to generate AI recommended copies: {ai_exc}")
                        print("[ERROR] Traceback:")
                        traceback.print_exc()
                        print()


            else:
                result.setdefault("ai_copy_status", "skipped")
                result.setdefault("ai_recommended_copies", [])
                result.setdefault("ai_copy_source", None)
                result.setdefault("ai_validation_notes", [])

        except Exception as exc:
            result = {
                "video": str(video),
                "status": "error",
                "error": str(exc),
            }
            print(f"[ERROR] {video}: {exc}\n")
        results.append(result)

    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_report, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved report to {args.json_report}")


if __name__ == "__main__":
    main()
