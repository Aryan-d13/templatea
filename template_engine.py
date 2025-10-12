"""
template_engine.py
A modular template renderer that uses a template.json and assets folder.
Works with PIL + ffmpeg subprocess. Designed to be a drop-in helper
for process_marketingspots_template in marketingspots_template.py.

Enhanced with:
- Dual text system (top_text, bottom_text)
- Video-relative positioning
- Header image support
- Color word highlighting
- Advanced logo positioning with rotation
"""

from dataclasses import dataclass
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json
import string
import tempfile
import requests
import re
import subprocess
import shlex
import os
import math
import logging
from typing import Tuple, List, Set, Optional, Dict
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GROQ_API_KEY", "").strip()

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
HIGHLIGHT_SYSTEM_PROMPT = (
    "You are a text analyzer. Given a hook copy and a number n, identify the n consecutive words "
    "that would be most impactful when highlighted. Follow the user's rules exactly."
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[template_engine] %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger.propagate = False

# Small util to call ffprobe
def probe_video_size(path: str) -> Tuple[int,int]:
    try:
        cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 {shlex.quote(path)}"
        out = subprocess.check_output(shlex.split(cmd), stderr=subprocess.DEVNULL).decode().strip()
        w, h = out.split(",")
        return int(w), int(h)
    except Exception:
        # fallback default
        return 1080, 1920

def load_template_cfg(template_root: Path):
    cfg_path = template_root / "template.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing template.json at {cfg_path}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))

def get_line_height(font: ImageFont.FreeTypeFont) -> int:
    """
    Return a font-specific line height that stays consistent regardless of the glyphs in the actual text.
    """
    try:
        ascent, descent = font.getmetrics()
        line_height = ascent + descent
        if line_height > 0:
            return line_height
    except Exception:
        pass

    try:
        bbox = font.getbbox("Ag")
        if bbox:
            return max(1, bbox[3] - bbox[1])
    except Exception:
        pass

    return max(1, getattr(font, "size", 12))


def measure_wrapped_lines(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    draw: ImageDraw.Draw,
    line_spacing_factor: float = 1.2,
    line_height: Optional[int] = None,
):
    """Wrap text to fit within max_width. Returns (lines, total_height)."""
    words = text.split()
    lines: List[str] = []
    cur = ""

    for w in words:
        test = (cur + " " + w).strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w

    if cur:
        lines.append(cur)

    if not lines:
        return [], 0

    line_height = line_height or get_line_height(font)
    line_step = max(1, int(line_height * line_spacing_factor))
    total_height = line_height + (len(lines) - 1) * line_step

    return lines, total_height


def choose_font(draw, font_path: Path, requested_size: int, min_size: int, max_lines: int, max_width: int, text: str):
    """Binary search for largest font that fits text within max_lines"""
    lo = min_size
    hi = requested_size
    best_size = lo
    
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            font = ImageFont.truetype(str(font_path), mid)
        except Exception:
            font = ImageFont.load_default()
        
        lines, _ = measure_wrapped_lines(text, font, max_width, draw)
        
        if len(lines) <= max_lines:
            best_size = mid
            lo = mid + 1
        else:
            hi = mid - 1
    
    try:
        return ImageFont.truetype(str(font_path), best_size)
    except Exception:
        return ImageFont.load_default()

def apply_text_casing(text: str, mode: Optional[str]) -> str:
    """Normalize text casing according to the requested mode."""
    if not text:
        return text

    normalized = (mode or "original").lower()

    # NEW: Handle "none" to preserve original text
    if normalized == "none":
        return text

    if normalized in {"upper", "uppercase", "all_caps", "caps"}:
        return text.upper()
    if normalized in {"lower", "lowercase", "all_lower"}:
        return text.lower()
    if normalized in {"title", "titlecase", "title_case"}:
        return string.capwords(text)
    if normalized in {"sentence", "sentencecase", "sentence_case", "capitalize"}:
        idx = None
        for i, ch in enumerate(text):
            if ch.isalpha():
                idx = i
                break
        if idx is None:
            return text
        prefix = text[:idx]
        first = text[idx].upper()
        rest = text[idx + 1:].lower()
        return f"{prefix}{first}{rest}"

    return text

def draw_highlight_rect(draw: ImageDraw.Draw, rect: Tuple[int,int,int,int], radius: int, fill):
    """Safe rounded rectangle drawing"""
    try:
        draw.rounded_rectangle(rect, radius=radius, fill=fill)
    except Exception:
        draw.rectangle(rect, fill=fill)

def parse_highlight_words(manual_words: List) -> List[List[str]]:
    """
    Parse manual_words which can be:
    - Single words: ["Samsung", "Ballie"]
    - Phrases: ["AI-powered home helper"]
    Returns list of word lists (each inner list is a phrase to highlight together)
    """
    result = []
    for item in manual_words:
        if isinstance(item, str):
            # Check if it's a phrase (multiple words)
            words = item.strip().split()
            result.append(words)
        else:
            result.append([str(item)])
    return result

def find_phrase_positions(lines: List[str], phrase: List[str]) -> List[Tuple[int, int, int]]:
    """
    Find all occurrences of a phrase in the lines.
    Returns list of (line_idx, word_start_idx, word_end_idx) tuples.
    Handles phrases that span across line breaks.
    """
    positions = []
    
    # Flatten all words with their line and position info
    all_words = []
    for line_idx, line in enumerate(lines):
        words = line.split()
        for word_idx, word in enumerate(words):
            cleaned = word.strip(".,!?:;\"'()[]{}").lower()
            all_words.append((line_idx, word_idx, word, cleaned))
    
    # Search for phrase
    phrase_clean = [w.strip(".,!?:;\"'()[]{}").lower() for w in phrase]
    phrase_len = len(phrase_clean)
    
    i = 0
    while i <= len(all_words) - phrase_len:
        # Check if phrase matches starting at position i
        match = True
        for j in range(phrase_len):
            if all_words[i + j][3] != phrase_clean[j]:
                match = False
                break
        
        if match:
            # Found a match
            start_line = all_words[i][0]
            end_line = all_words[i + phrase_len - 1][0]
            
            if start_line == end_line:
                # Phrase is on same line
                positions.append((start_line, all_words[i][1], all_words[i + phrase_len - 1][1]))
            else:
                # Phrase spans multiple lines - highlight each line segment separately
                current_line = start_line
                for k in range(phrase_len):
                    word_line = all_words[i + k][0]
                    word_pos = all_words[i + k][1]
                    
                    if word_line != current_line:
                        current_line = word_line
                    
                    # Add individual word position
                    positions.append((word_line, word_pos, word_pos))
            
            i += phrase_len
        else:
            i += 1
    
    return positions

def draw_highlights(
    draw: ImageDraw.Draw,
    lines: List[str],
    font: ImageFont.FreeTypeFont,
    x_origin: int,
    y_origin: int,
    highlight_phrases: List[List[str]],
    highlight_color: str,
    line_origins: Optional[List[int]] = None,
    pad_px: int = 7,
    line_positions: Optional[List[int]] = None,
    line_height: Optional[int] = None,
    line_step: Optional[int] = None,
):
    """Draw highlight rectangles behind chosen words/phrases."""
    if not highlight_phrases:
        return

    radius = max(4, int(font.size * 0.2))
    line_height = line_height or get_line_height(font)
    line_step = line_step or max(1, line_height)

    if line_positions is None:
        line_positions = []
        current_y = y_origin
        for idx in range(len(lines)):
            line_positions.append(current_y)
            if idx < len(lines) - 1:
                current_y += line_step

    highlight_positions = set()
    for phrase in highlight_phrases:
        positions = find_phrase_positions(lines, phrase)
        for line_idx, start_word, end_word in positions:
            for word_idx in range(start_word, end_word + 1):
                highlight_positions.add((line_idx, word_idx))

    for line_idx, line in enumerate(lines):
        line_top = line_positions[line_idx] if line_idx < len(line_positions) else y_origin
        words = line.split()
        cursor_x = (
            line_origins[line_idx]
            if line_origins and line_idx < len(line_origins)
            else x_origin
        )

        for word_idx, word in enumerate(words):
            bbox = draw.textbbox((cursor_x, line_top), word, font=font)
            highlight_rect = (
                bbox[0] - pad_px,
                bbox[1] - pad_px,
                bbox[2] + pad_px,
                bbox[3] + pad_px,
            )
            if (line_idx, word_idx) in highlight_positions:
                try:
                    draw.rounded_rectangle(highlight_rect, radius=radius, fill=highlight_color)
                except Exception:
                    draw.rectangle(highlight_rect, fill=highlight_color)

            word_width = font.getlength(word)
            space_width = font.getlength(' ') if word_idx < len(words) - 1 else 0
            cursor_x += word_width + space_width


def _fallback_highlight_phrases(text: str, top_k: int) -> List[List[str]]:
    cleaned = [w.strip(".,!?:;\"'()[]{}") for w in text.split()]
    cleaned = [w for w in cleaned if w]
    if not cleaned:
        logger.debug("highlight fallback: no tokens available")
    cleaned.sort(key=lambda w: -len(w))
    limit = max(1, top_k)
    phrases = [[w] for w in cleaned[:limit]]
    logger.debug("highlight fallback selected tokens=%s", phrases)
    return phrases


def _call_groq_highlight_phrase(text: str, n: int, api_key: str, model: str = "llama-3.3-70b-versatile", timeout: int = 10) -> Optional[List[str]]:
    if not api_key or not text or n <= 0:
        return None
    
    n=1
    logger.debug("highlight ai preparing Groq call key_present=%s key_length=%d", bool(api_key), len(api_key))

    logger.debug("highlight ai request: text_len=%d top_k=%d model=%s", len(text), n, model)

    user_prompt = (
        "You are a text analyzer. Given a hook copy, identify word "
        "that would be most impactful when highlighted.\n\n"
        f"INPUT:\nHook Copy: {text}\n"
        "RULES:\n"
        "1. Select exactly 1 consecutive words from the hook copy\n"
        "2. Choose word that is most attention-grabbing or emotionally impactful\n"
        "4. Return ONLY the selected word as it appears in the text\n"
        "5. No explanations, no formatting, no quotation marks, no additional text\n\n"
        "OUTPUT:\n"
        "[return only and only the selected word]"
    )

    logger.debug("highlight ai prompt preview=%r", user_prompt[:200])

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": HIGHLIGHT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }

    try:
        resp = requests.post(GROQ_CHAT_COMPLETIONS_URL, headers=headers, json=payload, timeout=timeout)
        logger.debug("highlight ai response http_status=%s", resp.status_code)
        resp.raise_for_status()
        logger.debug("highlight ai response status=%s body=%r", resp.status_code, resp.text[:200])
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 'unknown'
        body = exc.response.text[:200] if exc.response is not None and getattr(exc.response, 'text', None) else ''
        logger.warning("highlight ai request failed http status=%s body=%r", status, body)
        return None
    except Exception as exc:
        logger.warning("highlight ai request failed: %s", exc)
        return None

    candidate = re.sub(r"\s+", " ", content.strip(" \"'"))
    logger.debug("highlight ai raw candidate=%r", candidate)
    words = candidate.split()
    if len(words) != n:
        logger.info("highlight ai rejected: expected %d words, got %d", n, len(words))
        return None

    pattern = r"\b" + r"\s+".join(re.escape(w) for w in words) + r"\b"

    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        logger.info("highlight ai rejected: candidate not found in original text")
        return None

    original_span = text[match.start():match.end()]
    original_words = original_span.split()
    if len(original_words) != n:
        return None

    logger.info("highlight ai accepted phrase=%s", original_words)
    return original_words


def select_highlight_words_via_ai(text: str, top_k: int = 3) -> List[List[str]]:
    fallback = _fallback_highlight_phrases(text, top_k)
    if not text:
        logger.info("highlight ai skipped: empty text")
        logger.info("highlight fallback tokens=%s", fallback)
        return fallback
    if top_k <= 0:
        logger.info("highlight ai skipped: non-positive top_k=%s", top_k)
        logger.info("highlight fallback tokens=%s", fallback)
        return fallback

    logger.debug(
        "highlight ai env check: key_present=%s key_length=%d",
        bool(api_key),
        len(api_key),
    )

    
    if not api_key:
        logger.info("highlight ai skipped: GROQ_API_KEY missing")
        logger.info("highlight fallback tokens=%s", fallback)
        return fallback

    phrase = _call_groq_highlight_phrase(text, top_k, api_key)

    if phrase:
        logger.info("highlight ai using Groq phrase=%s", phrase)
        return [phrase]

    logger.info("highlight ai falling back to heuristic selection")
    logger.info("highlight fallback tokens=%s", fallback)
    return fallback

def probe_duration(path):
    try:
        out = subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration",
                                       "-of","default=noprint_wrappers=1:nokey=1", path])
        return float(out.decode().strip())
    except Exception:
        return None


def composite_canvas_and_video(canvas_path: str, input_video_path: str, out_video_path: str, cfg: dict):
    """
    Fixed compositing that explicitly limits canvas duration to match video duration.
    """
    import tempfile, shlex, subprocess, os, math, shutil

    template_video_cfg = cfg.get("video", {})
    canvas_w = cfg.get("canvas", {}).get("width", 1080)
    vw_pct = template_video_cfg.get("width_pct", 100) / 100.0
    vw = int(canvas_w * vw_pct)

    # Get video duration FIRST
    video_duration = probe_duration(input_video_path)
    if video_duration is None:
        raise RuntimeError("Cannot determine video duration")

    # temp files
    tmp_scaled = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_scaled.close()

    base_opts = "-hide_banner -loglevel warning"

    # 1) scale input preserving aspect ratio
    cmd_scale = (
        f"ffmpeg -y {base_opts} -i {shlex.quote(input_video_path)} "
        f"-vf scale={vw}:-2 -c:v libx264 -crf 18 -preset veryfast -c:a copy {shlex.quote(tmp_scaled.name)}"
    )
    subprocess.run(shlex.split(cmd_scale), check=True)

    # compute overlay position using new positioning
    canvas_h = cfg.get("canvas", {}).get("height", 1920)
    vw_actual, vh_actual = probe_video_size(tmp_scaled.name)
    
    # NEW: Always center horizontally
    ox = (canvas_w - vw_actual)//2
    
    # Use y from config or default to center
    pos = template_video_cfg.get("position", {})
    if "y" in pos:
        # Y represents the CENTER of the video, not top-left
        center_y = int(pos["y"])
        oy = int(center_y - vh_actual // 2)
    else:
        oy = (canvas_h - vh_actual)//2

    # prepare output tmp
    tmp_with_audio = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_with_audio.close()

    # 2) Overlay with explicit duration limits
    cmd_overlay = (
        f'ffmpeg -y {base_opts} '
        f'-loop 1 -t {video_duration} -i {shlex.quote(canvas_path)} '
        f'-i {shlex.quote(tmp_scaled.name)} '
        f'-filter_complex "[0:v][1:v]overlay={ox}:{oy}:format=yuv420:shortest=1[outv]" '
        f'-map "[outv]" -map 1:a? -c:v libx264 -crf 18 -preset veryfast -c:a aac '
        f'-t {video_duration} {shlex.quote(tmp_with_audio.name)}'
    )

    try:
        subprocess.run(shlex.split(cmd_overlay), check=True)
        shutil.move(tmp_with_audio.name, out_video_path)
    except subprocess.CalledProcessError as e:
        log_path = os.path.join(tempfile.gettempdir(), "ffmpeg_overlay_error.log")
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"COMMAND: {cmd_overlay}\nERROR: {e}\n\n")
        try:
            os.unlink(tmp_with_audio.name)
        except Exception:
            pass
        raise
    finally:
        try:
            os.unlink(tmp_scaled.name)
        except Exception:
            pass

    return True


def draw_text_with_color_highlight(
    draw: ImageDraw.Draw,
    lines: List[str],
    font: ImageFont.FreeTypeFont,
    x_origin: int,
    y_origin: int,
    highlight_phrases: List[List[str]],
    highlight_color: str,
    base_color: str,
    line_origins: Optional[List[int]] = None,
    line_positions: Optional[List[int]] = None,
    line_height: Optional[int] = None,
    line_step: Optional[int] = None,
):
    """Draw text with colored words instead of background highlights."""
    line_height = line_height or get_line_height(font)
    line_step = line_step or max(1, line_height)

    if line_positions is None:
        line_positions = []
        current_y = y_origin
        for idx in range(len(lines)):
            line_positions.append(current_y)
            if idx < len(lines) - 1:
                current_y += line_step

    highlight_positions = set()
    for phrase in highlight_phrases:
        positions = find_phrase_positions(lines, phrase)
        for line_idx, start_word, end_word in positions:
            for word_idx in range(start_word, end_word + 1):
                highlight_positions.add((line_idx, word_idx))

    for line_idx, line in enumerate(lines):
        line_top = line_positions[line_idx] if line_idx < len(line_positions) else y_origin
        words = line.split()
        cursor_x = (
            line_origins[line_idx]
            if line_origins and line_idx < len(line_origins)
            else x_origin
        )

        for word_idx, word in enumerate(words):
            color = highlight_color if (line_idx, word_idx) in highlight_positions else base_color
            draw.text((cursor_x, line_top), word, font=font, fill=color)
            
            word_width = font.getlength(word)
            space_width = font.getlength(' ') if word_idx < len(words) - 1 else 0
            cursor_x += word_width + space_width


@dataclass
class TemplateRenderRequest:
    """
    Encapsulates the minimum payload required to render a template.
    Provides a structured hand-off point for CLIs, APIs, or pipelines.
    """
    input_video_path: str
    output_video_path: str
    text: str
    template_root: str
    overrides: Optional[Dict] = None
    # NEW: Optional separate text for top/bottom
    top_text: Optional[str] = None
    bottom_text: Optional[str] = None


class TemplateEngine:
    """
    Thin orchestrator around the functional helpers in this module.
    Keeps render logic isolated so multiple entrypoints can reuse it.
    """

    def __init__(self, request: TemplateRenderRequest):
        self.request = request

    def render(self) -> bool:
        root = Path(self.request.template_root)
        cfg = load_template_cfg(root)
        if self.request.overrides:
            cfg.update(self.request.overrides)

        canvas_w = int(cfg.get("canvas", {}).get("width", 1080))
        canvas_h = int(cfg.get("canvas", {}).get("height", 1920))
        image = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 255))
        draw = ImageDraw.Draw(image)

        # background
        bg = cfg.get("background", {})
        if bg.get("type") == "image":
            bg_path = root / bg.get("value", "")
            if bg_path.exists():
                bgi = Image.open(bg_path).convert("RGBA")
                bgi = bgi.resize((canvas_w, canvas_h), Image.LANCZOS)
                image.paste(bgi, (0, 0))
            else:
                draw.rectangle([0, 0, canvas_w, canvas_h], fill=bg.get("value", "#000000"))
        else:
            draw.rectangle([0, 0, canvas_w, canvas_h], fill=bg.get("value", "#000000"))

        # Probe input video to understand aspect
        vid_w, vid_h = probe_video_size(self.request.input_video_path)

        # Compute video box with NEW X,Y positioning
        video_cfg = cfg.get("video", {})
        vw = int(canvas_w * (video_cfg.get("width_pct", 100) / 100.0))
        aspect = vid_h and (vid_w / vid_h) or (16 / 9)
        vh = max(2, int(vw / aspect))
        
        # NEW: Always center horizontally
        vx = int((canvas_w - vw) // 2)  # ALWAYS CENTERED HORIZONTALLY
        
        pos = video_cfg.get("position", {})
        if "y" in pos:
            # Y represents the CENTER of the video, not top-left
            center_y = int(pos["y"])
            vy = int(center_y - vh // 2)
        else:
            # Default to center vertically
            vy = int((canvas_h - vh) // 2)
        
        logger.debug(f"Video positioning: vx={vx}, vy={vy}, vw={vw}, vh={vh}, canvas_h={canvas_h}")

        # Process top_text if enabled
        top_text_cfg = cfg.get("top_text", {})
        if top_text_cfg.get("enabled", False):
            # Use specific top_text if provided, otherwise fall back to main text
            text_content = self.request.top_text if self.request.top_text else self.request.text
            self._render_text_block(
                draw, image, root, top_text_cfg, canvas_w, canvas_h,
                vx, vy, vw, vh, "top", text_content
            )

        # Process bottom_text if enabled
        bottom_text_cfg = cfg.get("bottom_text", {})
        if bottom_text_cfg.get("enabled", False):
            # Use specific bottom_text if provided, otherwise fall back to main text
            text_content = self.request.bottom_text if self.request.bottom_text else self.request.text
            try:
                self._render_text_block(
                    draw, image, root, bottom_text_cfg, canvas_w, canvas_h,
                    vx, vy, vw, vh, "bottom", text_content
                )
            except Exception as exc:
                logger.error("bottom_text render failed: type=%s value=%r error=%s", type(text_content), text_content, exc, exc_info=True)
                raise

        # BACKWARD COMPATIBILITY: If neither top_text nor bottom_text is configured,
        # fall back to legacy "text" config
        if not top_text_cfg.get("enabled", False) and not bottom_text_cfg.get("enabled", False):
            legacy_text_cfg = cfg.get("text", {})
            if legacy_text_cfg:
                # Convert legacy config to top_text format
                legacy_as_top = {
                    "enabled": True,
                    "font": legacy_text_cfg.get("font", ""),
                    "font_size": legacy_text_cfg.get("font_size", 120),
                    "min_font_size": legacy_text_cfg.get("min_font_size", 40),
                    "max_lines": legacy_text_cfg.get("max_lines", 2),
                    "line_width_pct": legacy_text_cfg.get("line_width_pct", 90),
                    "line_spacing_factor": cfg.get("text.line_spacing_factor", legacy_text_cfg.get("line_spacing_factor", 1.35)),
                    "color": legacy_text_cfg.get("color", "#ffffff"),
                    "casing_mode": cfg.get("text.casing_mode", legacy_text_cfg.get("casing_mode", "original")),
                    "align": legacy_text_cfg.get("align", "center"),
                    "gap_px": legacy_text_cfg.get("position_relative_to_video", {}).get("gap_px", 20),
                    "highlight": legacy_text_cfg.get("highlight", {})
                }
                text_content = self.request.top_text if self.request.top_text else self.request.text
                self._render_text_block(
                    draw, image, root, legacy_as_top, canvas_w, canvas_h,
                    vx, vy, vw, vh, "top", text_content
                )

        # Header image (only if top_text is enabled)
        if top_text_cfg.get("enabled", False):
            header_cfg = cfg.get("header", {})
            if header_cfg.get("enabled", False):
                self._render_header(image, root, header_cfg, canvas_w, top_text_cfg, vx, vy)

        # Logo with NEW positioning options
        logo_cfg = cfg.get("logo", {})
        if logo_cfg.get("enabled", True):
            self._render_logo(image, root, logo_cfg, canvas_w, canvas_h, vx, vy, vw, vh)

        # Save canvas to temp file
        tmp_canvas = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_canvas.close()
        image.save(tmp_canvas.name)

        # Composite canvas and video
        composite_canvas_and_video(tmp_canvas.name, self.request.input_video_path, self.request.output_video_path, cfg)

        # Cleanup
        try:
            os.unlink(tmp_canvas.name)
        except Exception:
            pass

        return True

    def _render_text_block(self, draw, image, root, txt_cfg, canvas_w, canvas_h, vx, vy, vw, vh, position_type, text_content):
        """Render a text block (top or bottom) relative to video."""
        font_rel = txt_cfg.get("font", "")
        font_path = root / font_rel if font_rel else None
        
        # Debug logging
        logger.debug(f"_render_text_block called: position={position_type}, text='{text_content}'")
        logger.debug(f"Font path: {font_path}, exists={font_path.exists() if font_path else False}")
        
        requested_size = int(txt_cfg.get("font_size", 120))
        min_size = int(txt_cfg.get("min_font_size", 40))
        max_lines = int(txt_cfg.get("max_lines", 2))
        line_width_pct = float(txt_cfg.get("line_width_pct", 90))
        max_width = int(canvas_w * (line_width_pct / 100.0))
        
        casing_mode = txt_cfg.get("casing_mode", "original")
        render_text = apply_text_casing(text_content, casing_mode)
        
        logger.debug(f"Render text after casing: '{render_text}'")
        
        line_spacing_factor = float(txt_cfg.get("line_spacing_factor", 1.35))

        # Choose font that fits within max_lines
        if font_path and font_path.exists():
            font = choose_font(draw, font_path, requested_size, min_size, max_lines, max_width, render_text)
        else:
            font = ImageFont.load_default()

        line_height = get_line_height(font)
        line_step = max(1, int(line_height * line_spacing_factor))
        lines, text_h = measure_wrapped_lines(
            render_text,
            font,
            max_width,
            draw,
            line_spacing_factor=line_spacing_factor,
            line_height=line_height,
        )

        # Position text relative to video
        gap_px = int(txt_cfg.get("gap_px", 20))
        
        if position_type == "top":
            # Bottom of text should be gap_px above video top
            text_bottom = vy - gap_px
            text_top = int(text_bottom - text_h)
        else:  # bottom
            # Top of text should be gap_px below video bottom
            text_top = vy + vh + gap_px

        # x origin by alignment
        align = txt_cfg.get("align", "center")
        align_lower = align.lower() if isinstance(align, str) else "center"
        margin = int(canvas_w * 0.05)
        if align_lower == "center":
            x_origin = (canvas_w - max_width) // 2
        elif align_lower == "right":
            x_origin = canvas_w - margin - max_width
        else:
            x_origin = margin

        line_origins: List[int] = []
        for ln in lines:
            bbox = draw.textbbox((0, 0), ln, font=font)
            line_width = bbox[2] - bbox[0]
            if align_lower == "center":
                origin = int((canvas_w - line_width) // 2)
            elif align_lower == "right":
                origin = canvas_w - margin - line_width
            else:
                origin = margin
            line_origins.append(origin)

        line_positions: List[int] = []
        current_line_top = text_top
        for idx in range(len(lines)):
            line_positions.append(current_line_top)
            if idx < len(lines) - 1:
                current_line_top += line_step

        # Highlight configuration
        hl_cfg = txt_cfg.get("highlight", {})
        highlight_phrases = []

        if hl_cfg.get("enabled", False):
            if hl_cfg.get("mode", "ai") == "manual":
                # Manual mode: parse manual_words which can be words or phrases
                manual_words = hl_cfg.get("manual_words", [])
                highlight_phrases = parse_highlight_words(manual_words)
            else:
                # AI mode: use highlight_count from config (default 3)
                top_k = int(hl_cfg.get("highlight_count", 3))
                highlight_phrases = select_highlight_words_via_ai(render_text, top_k=top_k)

        # NEW: Check highlight type - background or color_word
        highlight_type = hl_cfg.get("type", "background")  # "background" or "color_word"
        text_color = txt_cfg.get("color", "#ffffff")

        if highlight_phrases and highlight_type == "background":
            # Draw highlights FIRST (behind text)
            draw_highlights(
                draw,
                lines,
                font,
                x_origin,
                text_top,
                highlight_phrases,
                hl_cfg.get("color", "#ffde59"),
                line_origins=line_origins,
                line_positions=line_positions,
                line_height=line_height,
                line_step=line_step,
            )
            # Draw text ON TOP of highlights
            for idx, (ln, line_top) in enumerate(zip(lines, line_positions)):
                line_x = line_origins[idx] if idx < len(line_origins) else x_origin
                draw.text((line_x, line_top), ln, font=font, fill=text_color)
        elif highlight_phrases and highlight_type == "color_word":
            # Draw text with colored words
            draw_text_with_color_highlight(
                draw,
                lines,
                font,
                x_origin,
                text_top,
                highlight_phrases,
                hl_cfg.get("color", "#ffde59"),
                text_color,
                line_origins=line_origins,
                line_positions=line_positions,
                line_height=line_height,
                line_step=line_step,
            )
        else:
            # No highlights, just draw text normally
            for idx, (ln, line_top) in enumerate(zip(lines, line_positions)):
                line_x = line_origins[idx] if idx < len(line_origins) else x_origin
                draw.text((line_x, line_top), ln, font=font, fill=text_color)

        # Store text bounds for header positioning
        if position_type == "top":
            txt_cfg["_computed_top"] = text_top
            txt_cfg["_computed_height"] = text_h

    def _render_header(self, image, root, header_cfg, canvas_w, top_text_cfg, vx, vy):
        """Render header image above top_text."""
        header_path = root / header_cfg.get("path", "")
        if not header_path.exists():
            return

        try:
            header = Image.open(header_path).convert("RGBA")
            
            # Get target width
            target_w = header_cfg.get("width", 400)
            
            # Preserve aspect ratio
            aspect_ratio = header.height / header.width if header.width else 1
            target_h = int(target_w * aspect_ratio)
            
            header = header.resize((target_w, target_h), Image.LANCZOS)
            
            # Position above top_text
            gap_px = int(header_cfg.get("gap_px", 20))
            text_top = top_text_cfg.get("_computed_top", vy)
            
            hx = (canvas_w - target_w) // 2  # Center horizontally
            hy = text_top - target_h - gap_px
            
            image.paste(header, (hx, hy), header)
        except Exception as e:
            logger.warning(f"Failed to render header: {e}")

    def _render_logo(self, image, root, logo_cfg, canvas_w, canvas_h, vx, vy, vw, vh):
        """Render logo with advanced positioning and rotation."""
        logo_path = root / logo_cfg.get("path", "")
        if not logo_path.exists():
            return

        try:
            logo = Image.open(logo_path).convert("RGBA")

            sz = logo_cfg.get("size", {})
            target_w = sz.get("width")
            target_h = sz.get("height")

            # Preserve aspect ratio
            if target_w and target_h:
                target_w = int(target_w)
                target_h = int(target_h)
            elif target_w:
                target_w = int(target_w)
                aspect_ratio = logo.height / logo.width if logo.width else 1
                target_h = int(target_w * aspect_ratio)
            elif target_h:
                target_h = int(target_h)
                aspect_ratio = logo.width / logo.height if logo.height else 1
                target_w = int(target_h * aspect_ratio)
            else:
                target_w = logo.width
                target_h = logo.height

            logo = logo.resize((target_w, target_h), Image.LANCZOS)

            # NEW: Rotation support
            rotation = logo_cfg.get("rotation", 0)
            if rotation != 0:
                logo = logo.rotate(rotation, expand=True, resample=Image.BICUBIC)
                # Update dimensions after rotation
                target_w, target_h = logo.size

            # NEW: Position modes
            position_mode = logo_cfg.get("position_mode", "canvas")  # "canvas" or "video"
            
            if position_mode == "video":
                # Position relative to video
                video_relative = logo_cfg.get("video_relative", {})
                offset_x = int(video_relative.get("x", 0))
                offset_y = int(video_relative.get("y", 0))
                lx = vx + offset_x
                ly = vy + offset_y
            else:
                # Traditional canvas positioning
                pos = logo_cfg.get("position", "top-right")
                margin = int(logo_cfg.get("margin", 36))

                # NEW: Support for x, y coordinates
                if isinstance(pos, dict):
                    lx = int(pos.get("x", margin))
                    ly = int(pos.get("y", margin))
                elif pos == "top-right":
                    lx = canvas_w - target_w - margin
                    ly = margin
                elif pos == "top-left":
                    lx = margin
                    ly = margin
                elif pos == "bottom-right":
                    lx = canvas_w - target_w - margin
                    ly = canvas_h - target_h - margin
                else:  # bottom-left
                    lx = margin
                    ly = canvas_h - target_h - margin

            image.paste(logo, (lx, ly), logo)
        except Exception as e:
            logger.warning(f"Failed to render logo: {e}")


def render_with_template(input_video_path: str, output_video_path: str, text: str, template_root: str, overrides: dict = None):
    """
    Main engine entrypoint. template_root is path to a folder containing template.json and assets.
    overrides is a dict to override keys in the template json (shallow merge).
    """
    request = TemplateRenderRequest(
        input_video_path=input_video_path,
        output_video_path=output_video_path,
        text=text,
        template_root=template_root,
        overrides=overrides,
    )
    engine = TemplateEngine(request)
    return engine.render()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run marketing spot template renderer")
    parser.add_argument("--input-video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output-video", type=str, required=True, help="Path to save output video")
    parser.add_argument("--text", type=str, required=True, help="Text content")
    parser.add_argument("--template-root", type=str, required=True, help="Folder with template.json and assets")
    parser.add_argument("--overrides", type=str, help="JSON string of any template overrides (optional)")
    args = parser.parse_args()

    overrides = None
    if args.overrides:
        overrides = json.loads(args.overrides)

    render_with_template(
        input_video_path=args.input_video,
        output_video_path=args.output_video,
        text=args.text,
        template_root=args.template_root,
        overrides=overrides
    )
