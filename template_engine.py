"""
template_engine.py
A modular template renderer that uses a template.json and assets folder.
Works with PIL + ffmpeg subprocess. Designed to be a drop-in helper
for process_marketingspots_template in marketingspots_template.py.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json
import tempfile
import subprocess
import shlex
import os
import math
from typing import Tuple, List

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
    # compute total height
    heights = [draw.textbbox((0,0), ln, font=font)[3] - draw.textbbox((0,0), ln, font=font)[1] for ln in lines]
    line_spacing = int(font.size * 0.12)
    total_height = sum(heights) + max(0, (len(lines)-1) * line_spacing)
    return lines, total_height

def choose_font(draw, font_path: Path, requested_size: int, min_size: int, max_lines: int, max_width: int, text: str):
    lo = min_size
    hi = requested_size
    best_size = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            font = ImageFont.truetype(str(font_path), mid)
        except Exception:
            # fallback to default PIL font
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
    # safe rounded rectangle
    try:
        draw.rounded_rectangle(rect, radius=radius, fill=fill)
    except Exception:
        draw.rectangle(rect, fill=fill)

def draw_highlights(draw: ImageDraw.Draw, lines: List[str], font: ImageFont.FreeTypeFont, x_origin: int, y_origin: int,
                    highlight_words: List[str], highlight_color: str):
    padding = max(6, int(font.size * 0.25))
    y = y_origin
    for line in lines:
        parts = line.split(" ")
        cursor_x = x_origin
        for p in parts:
            cleaned = p.strip(".,!?:;\"'()[]{}").lower()
            bbox = draw.textbbox((0,0), p, font=font)
            w = bbox[2] - bbox[0]
            # consider trailing space width for cursor advance
            space_bbox = draw.textbbox((0,0), p + " ", font=font)
            space_w = space_bbox[2] - space_bbox[0]
            if any(cleaned == hw.lower().strip() for hw in highlight_words):
                rect = [cursor_x - padding, y - padding, cursor_x + w + padding, y + (bbox[3]-bbox[1]) + padding]
                draw_highlight_rect(draw, rect, radius=padding, fill=highlight_color)
            cursor_x += space_w
        y += int((bbox[3]-bbox[1]) * 1.08)

def select_highlight_words_via_ai(text: str, top_k: int = 3) -> List[str]:
    # Default placeholder: longest words - meant to be replaced by your AI hook.
    cleaned = [w.strip(".,!?:;\"'()[]{}") for w in text.split()]
    cleaned = [w for w in cleaned if len(w) > 2]
    cleaned.sort(key=lambda w: -len(w))
    return cleaned[:top_k]

def composite_canvas_and_video(canvas_path: str, input_video_path: str, out_video_path: str, cfg: dict):
    """
    This function overlays the input_video onto the canvas image using ffmpeg filters.
    It assumes input_video is already cropped to the area you want to place on the canvas.
    """
    # ffmpeg command approach:
    # 1. scale video to desired vw x vh using -vf scale
    # 2. overlay at desired position
    # 3. map audio from input_video (if requested)
    template_video_cfg = cfg.get("video", {})
    canvas_w = cfg.get("canvas", {}).get("width", 1080)
    vw_pct = template_video_cfg.get("width_pct", 100) / 100.0
    vw = int(canvas_w * vw_pct)
    # we'll calculate vh by probing original (preserve aspect by scale=-2)
    tmp_scaled = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_scaled.close()
    ffmpeg_params = cfg.get("render", {}).get("ffmpeg_params", "-c:v libx264 -crf 18 -preset veryfast")
    try:
        # scale while preserving aspect
        cmd_scale = f"ffmpeg -y -i {shlex.quote(input_video_path)} -vf scale={vw}:-2 -c:v libx264 -crf 18 -preset veryfast -an {shlex.quote(tmp_scaled.name)}"
        subprocess.check_call(shlex.split(cmd_scale))
        # overlay onto canvas image
        # read target overlay position
        pos = template_video_cfg.get("position", {})
        # default center horizontally and use vertical_align
        canvas_h = cfg.get("canvas", {}).get("height", 1920)
        vertical_align = template_video_cfg.get("vertical_align", "center")
        # compute vw/vh
        vw_actual, vh_actual = probe_video_size(tmp_scaled.name)
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
            ox = pos.get("x", (canvas_w - vw_actual)//2)
            oy = pos.get("y", (canvas_h - vh_actual)//2)

        # overlay command: canvas (image) as base, video as overlay, then audio map from input_video_path
        tmp_with_audio = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_with_audio.close()
        cmd_overlay = (
            f'ffmpeg -y -loop 1 -i {shlex.quote(canvas_path)} -i {shlex.quote(tmp_scaled.name)} '
            f'-filter_complex "[1:v]format=yuva420p[vid];[0:v][vid]overlay={ox}:{oy}:format=yuv420" '
            f'-map 0:v -map 1:a? -c:v libx264 -crf 18 -preset veryfast -c:a aac -shortest {shlex.quote(tmp_with_audio.name)}'
        )
        subprocess.check_call(shlex.split(cmd_overlay))
        # final move
        os.replace(tmp_with_audio.name, out_video_path)
    finally:
        for t in (tmp_scaled.name,):
            try:
                os.unlink(t)
            except Exception:
                pass

def render_with_template(input_video_path: str, output_video_path: str, text: str, template_root: str, overrides: dict = None):
    """
    Main engine entrypoint. template_root is path to a folder containing template.json and assets.
    overrides is a dict to override keys in the template json (shallow merge).
    """
    root = Path(template_root)
    cfg = load_template_cfg(root)
    if overrides:
        cfg.update(overrides)

    canvas_w = int(cfg.get("canvas", {}).get("width", 1080))
    canvas_h = int(cfg.get("canvas", {}).get("height", 1920))
    image = Image.new("RGBA", (canvas_w, canvas_h), (0,0,0,255))
    draw = ImageDraw.Draw(image)

    # background
    bg = cfg.get("background", {})
    if bg.get("type") == "image":
        bg_path = root / bg.get("value", "")
        if bg_path.exists():
            bgi = Image.open(bg_path).convert("RGBA")
            bgi = bgi.resize((canvas_w, canvas_h), Image.LANCZOS)
            image.paste(bgi, (0,0))
        else:
            draw.rectangle([0,0,canvas_w,canvas_h], fill=bg.get("value","#000000"))
    else:
        draw.rectangle([0,0,canvas_w,canvas_h], fill=bg.get("value","#000000"))

    # Probe input video to understand aspect
    vid_w, vid_h = probe_video_size(input_video_path)

    # Compute video box
    video_cfg = cfg.get("video", {})
    vw = int(canvas_w * (video_cfg.get("width_pct", 100)/100.0))
    # preserve aspect from vid_w:vid_h
    aspect = vid_h and (vid_w/vid_h) or (16/9)
    vh = max(2, int(vw / aspect))
    vx = int((canvas_w - vw)//2)
    vertical_align = video_cfg.get("vertical_align", "center")
    if vertical_align == "center":
        vy = int((canvas_h - vh)//2)
    elif vertical_align == "top":
        vy = 0
    elif vertical_align == "bottom":
        vy = canvas_h - vh
    else:
        vy = int(video_cfg.get("y", (canvas_h - vh)//2))

    # Text layout
    txt_cfg = cfg.get("text", {})
    font_rel = txt_cfg.get("font", "")
    font_path = root / font_rel if font_rel else None
    requested_size = int(txt_cfg.get("font_size", 120))
    min_size = int(txt_cfg.get("min_font_size", 40))
    max_lines = int(txt_cfg.get("max_lines", 2))
    line_width_pct = float(txt_cfg.get("line_width_pct", 90))
    max_width = int(canvas_w * (line_width_pct/100.0))
    # choose font
    if font_path and font_path.exists():
        font = choose_font(draw, font_path, requested_size, min_size, max_lines, max_width, text)
    else:
        font = ImageFont.load_default()

    lines, text_h = measure_wrapped_lines(text, font, max_width, draw)

    # compute text bottom anchored above video top by gap_px
    gap_px = int(txt_cfg.get("position_relative_to_video", {}).get("gap_px", 20))
    text_bottom = vy - gap_px
    text_top = int(text_bottom - text_h)
    # x origin by alignment
    align = txt_cfg.get("align", "center")
    if align == "center":
        x_origin = (canvas_w - max_width)//2
    elif align == "left":
        x_origin = int(canvas_w * 0.05)
    else:
        x_origin = canvas_w - int(canvas_w * 0.05) - max_width

    # highlight words
    hl_cfg = txt_cfg.get("highlight", {})
    highlight_words = []
    if hl_cfg.get("enabled", False):
        if hl_cfg.get("mode", "ai") == "manual":
            highlight_words = hl_cfg.get("manual_words", [])
        else:
            highlight_words = select_highlight_words_via_ai(text)

    # draw highlights
    if highlight_words:
        draw_highlights(draw, lines, font, x_origin, text_top, highlight_words, hl_cfg.get("default_highlight_color", "#ffde59"))

    # draw text
    y = text_top
    for ln in lines:
        draw.text((x_origin, y), ln, font=font, fill=txt_cfg.get("color", "#ffffff"))
        # approximate line height
        h = draw.textbbox((0,0), ln, font=font)[3] - draw.textbbox((0,0), ln, font=font)[1]
        y += int(h * 1.08)

    # logo
    logo_cfg = cfg.get("logo", {})
    if logo_cfg.get("enabled", True):
        logo_path = root / logo_cfg.get("path", "")
        if logo_path.exists():
            try:
                logo = Image.open(logo_path).convert("RGBA")
                # desired size
                sz = logo_cfg.get("size", {})
                target_w = int(sz.get("width", logo.width))
                target_h = int(sz.get("height", int(logo.height * (target_w/logo.width) if logo.width else logo.height)))
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
                pass

    # Save canvas image to tmp
    tmp_canvas = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_canvas.close()
    image.save(tmp_canvas.name)

    # Composite canvas and video into final video using ffmpeg
    composite_canvas_and_video(tmp_canvas.name, input_video_path, output_video_path, cfg)

    # cleanup
    try:
        os.unlink(tmp_canvas.name)
    except Exception:
        pass

    return True
