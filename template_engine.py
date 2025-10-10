"""
template_engine.py
A modular template renderer that uses a template.json and assets folder.
Works with PIL + ffmpeg subprocess. Designed to be a drop-in helper
for process_marketingspots_template in marketingspots_template.py.
"""

from dataclasses import dataclass
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json
import tempfile
import subprocess
import shlex
import os
import math
from typing import Tuple, List, Set, Optional, Dict

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

def measure_wrapped_lines(text: str, font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw.Draw):
    """Wrap text to fit within max_width. Returns (lines, total_height)"""
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        bbox = draw.textbbox((0,0), test, font=font)
        if bbox[2] <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    
    # Compute total height with proper line spacing
    if not lines:
        return [], 0
    
    line_spacing_factor = 1.12
    total_height = 0
    for i, ln in enumerate(lines):
        bbox = draw.textbbox((0,0), ln, font=font)
        line_h = bbox[3] - bbox[1]
        if i == 0:
            total_height += line_h
        else:
            total_height += int(line_h * line_spacing_factor)
    
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
    pad_px: int = 7  # <--- modular padding here
):
    """
    Draw highlight rectangles behind words/phrases—now with modular fixed padding.
    """
    if not highlight_phrases:
        return

    radius = max(4, int(font.size * 0.2))
    line_spacing = int(font.size * 0.12)  # font spacing

    highlight_positions = set()
    for phrase in highlight_phrases:
        positions = find_phrase_positions(lines, phrase)
        for line_idx, start_word, end_word in positions:
            for word_idx in range(start_word, end_word + 1):
                highlight_positions.add((line_idx, word_idx))

    y = y_origin
    for line_idx, line in enumerate(lines):
        words = line.split()
        cursor_x = (
            line_origins[line_idx]
            if line_origins and line_idx < len(line_origins)
            else x_origin
        )

        for word_idx, word in enumerate(words):
            bbox = draw.textbbox((cursor_x, y), word, font=font, anchor="la")
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
            space_width = font.getlength(" ") if word_idx < len(words) - 1 else 0
            cursor_x += word_width + space_width

        y += font.getbbox(line)[3] - font.getbbox(line)[1] + line_spacing


def select_highlight_words_via_ai(text: str, top_k: int = 3) -> List[str]:
    """
    Default AI selector: picks longest words.
    Returns list of individual words (not phrases).
    """
    cleaned = [w.strip(".,!?:;\"'()[]{}") for w in text.split()]
    cleaned = [w for w in cleaned if len(w) > 2]
    cleaned.sort(key=lambda w: -len(w))
    return cleaned[:top_k]

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

    # compute overlay position
    canvas_h = cfg.get("canvas", {}).get("height", 1920)
    vw_actual, vh_actual = probe_video_size(tmp_scaled.name)
    vertical_align = template_video_cfg.get("vertical_align", "center")
    if vertical_align == "center":
        ox = (canvas_w - vw_actual)//2
        oy = (canvas_h - vh_actual)//2
    elif vertical_align == "top":
        ox = (canvas_w - vw_actual)//2
        oy = 0
    elif vertical_align == "bottom":
        ox = (canvas_w - vw_actual)//2
        oy = canvas_h - vh_actual
    else:
        pos = template_video_cfg.get("position", {})
        ox = int(pos.get("x", (canvas_w - vw_actual)//2))
        oy = int(pos.get("y", (canvas_h - vh_actual)//2))

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

        # Compute video box
        video_cfg = cfg.get("video", {})
        vw = int(canvas_w * (video_cfg.get("width_pct", 100) / 100.0))
        aspect = vid_h and (vid_w / vid_h) or (16 / 9)
        vh = max(2, int(vw / aspect))
        vx = int((canvas_w - vw) // 2)
        vertical_align = video_cfg.get("vertical_align", "center")
        if vertical_align == "center":
            vy = int((canvas_h - vh) // 2)
        elif vertical_align == "top":
            vy = 0
        elif vertical_align == "bottom":
            vy = canvas_h - vh
        else:
            vy = int(video_cfg.get("y", (canvas_h - vh) // 2))

        # Text layout
        txt_cfg = cfg.get("text", {})
        font_rel = txt_cfg.get("font", "")
        font_path = root / font_rel if font_rel else None
        requested_size = int(txt_cfg.get("font_size", 120))
        min_size = int(txt_cfg.get("min_font_size", 40))
        max_lines = int(txt_cfg.get("max_lines", 2))
        line_width_pct = float(txt_cfg.get("line_width_pct", 90))
        max_width = int(canvas_w * (line_width_pct / 100.0))

        # Choose font that fits within max_lines
        if font_path and font_path.exists():
            font = choose_font(draw, font_path, requested_size, min_size, max_lines, max_width, self.request.text)
        else:
            font = ImageFont.load_default()

        lines, text_h = measure_wrapped_lines(self.request.text, font, max_width, draw)

        # CRITICAL FIX: Position text so BOTTOM is gap_px above video top
        gap_px = int(txt_cfg.get("position_relative_to_video", {}).get("gap_px", 20))
        text_bottom = vy - gap_px
        text_top = int(text_bottom - text_h)

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

        # Highlight configuration
        hl_cfg = txt_cfg.get("highlight", {})
        highlight_phrases = []

        if hl_cfg.get("enabled", False):
            if hl_cfg.get("mode", "ai") == "manual":
                # Manual mode: parse manual_words which can be words or phrases
                manual_words = hl_cfg.get("manual_words", [])
                highlight_phrases = parse_highlight_words(manual_words)
            else:
                # AI mode: use top_k from config (default 3)
                top_k = int(hl_cfg.get("highlight_count", 3))
                ai_words = select_highlight_words_via_ai(self.request.text, top_k=top_k)
                highlight_phrases = [[w] for w in ai_words]  # Convert to phrase format

        # Draw highlights FIRST (behind text)
        if highlight_phrases:
            draw_highlights(
                draw,
                lines,
                font,
                x_origin,
                text_top,
                highlight_phrases,
                hl_cfg.get("default_highlight_color", "#ffde59"),
                line_origins=line_origins,
            )

        # Draw text ON TOP of highlights
        y = text_top
        line_spacing_factor = 1.5
        for i, ln in enumerate(lines):
            line_x = line_origins[i] if i < len(line_origins) else x_origin
            draw.text((line_x, y), ln, font=font, fill=txt_cfg.get("color", "#ffffff"))
            bbox = draw.textbbox((0, 0), ln, font=font)
            h = bbox[3] - bbox[1]
            if i == 0:
                y += h
            else:
                y += int(h * line_spacing_factor)

        # Logo with proper aspect ratio usag
        logo_cfg = cfg.get("logo", {})
        if logo_cfg.get("enabled", True):
            logo_path = root / logo_cfg.get("path", "")
            if logo_path.exists():
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

                    pos = logo_cfg.get("position", "top-right")
                    margin = int(logo_cfg.get("margin", 36))

                    if pos == "top-right":
                        lx = canvas_w - target_w - margin
                        ly = margin
                    elif pos == "top-left":
                        lx = margin
                        ly = margin
                    elif pos == "bottom-right":
                        lx = canvas_w - target_w - margin
                        ly = canvas_h - target_h - margin
                    else:
                        lx = margin
                        ly = canvas_h - target_h - margin

                    image.paste(logo, (lx, ly), logo)
                except Exception:
                    pass  # Silent fail for logo issues

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
