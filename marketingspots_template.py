import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
import os
import subprocess
import shutil
import random

# --- MarketingSpots Template Configuration ---
TEMPLATE_CONFIG = {
    "canvas": {
        "width": 1080,
        "height": 1920,
        "background_color": "#000000"
    },
    "logo": {
        "path": "assets/marketingspots_logo.png",
        "position": "top-right",
        "offset": {"x": -20, "y": 20},
        "max_width": 200,
        "max_height": 80
    },
    "video": {
        "alignment": "center",
        "fit_mode": "width",
        "width_percent": 100,  # 100% for landscape, can be changed for portrait
        "portrait_width_percent": 70,  # Portrait/square videos use 70% width
        "maintain_aspect": True
    },
    "text": {
        "font_family": "Anton.ttf",  # Change to your font file
        "font_size": 52,
        "color": "#FFFFFF",
        "highlight_color": "#FF0000",
        "highlight_mode": "ai",
        "position": "above_video",
        "offset": {"x": 0, "y": -50},
        "alignment": "center",
        "max_width": 1000,
        "line_spacing": 1.2,
        "weight": "bold"
    },
    "audio": {
        "source": "extracted",
        "volume": 1.0
    }
}


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def get_ai_highlight_word(text):
    """
    AI mode to determine which word to highlight.
    Simple heuristic: Pick the longest word, or a random important-looking word.
    You can replace this with actual AI/NLP logic.
    """
    words = text.split()
    if not words:
        return None
    
    # Filter out common words
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    important_words = [w for w in words if w.lower() not in common_words]
    
    if not important_words:
        important_words = words
    
    # Pick the longest word
    highlight_word = max(important_words, key=len)
    return highlight_word


def get_wrapped_lines_with_highlight(text, font, max_width, highlight_word=None):
    """
    Wrap text and track which word to highlight.
    Returns: list of tuples [(line_text, [(word, is_highlighted), ...]), ...]
    """
    draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    words = text.split()
    if not words:
        return []
    
    lines = []
    current_line_words = [words[0]]
    
    for word in words[1:]:
        test_line = " ".join(current_line_words + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] <= max_width:
            current_line_words.append(word)
        else:
            lines.append(current_line_words)
            current_line_words = [word]
    lines.append(current_line_words)
    
    # Mark highlighted word
    result = []
    for line_words in lines:
        word_data = []
        for word in line_words:
            is_highlighted = (highlight_word and word.lower() == highlight_word.lower())
            word_data.append((word, is_highlighted))
        result.append(word_data)
    
    return result


def draw_centered_text_with_highlight(draw, lines_data, font, text_color, highlight_color, 
                                     canvas_width, text_bottom_y, line_spacing):
    """
    Draw text with highlighting support (red background rectangle).
    lines_data: list of lists [(word, is_highlighted), ...]
    text_bottom_y: Y position where the BOTTOM of the text block should end
    """
    line_height = int(font.size * line_spacing)
    
    # Calculate total text height
    total_height = len(lines_data) * line_height
    
    # Start from top (text_bottom_y - total_height)
    current_y = text_bottom_y - total_height
    
    for line_words in lines_data:
        # Calculate line width
        line_text = " ".join([word for word, _ in line_words])
        bbox = draw.textbbox((0, 0), line_text, font=font)
        line_width = bbox[2]
        
        # Start x position (centered)
        current_x = (canvas_width - line_width) / 2
        
        # Draw each word
        for word, is_highlighted in line_words:
            if is_highlighted:
                # Draw red background rectangle
                word_bbox = draw.textbbox((current_x, current_y), word, font=font)
                padding = 4  # Small padding around text
                draw.rectangle(
                    [word_bbox[0] - padding, word_bbox[1] - padding, 
                     word_bbox[2] + padding, word_bbox[3] + padding],
                    fill=highlight_color
                )
                # Draw text in white on red background
                draw.text((current_x, current_y), word, font=font, fill=(255, 255, 255))
            else:
                # Normal text
                draw.text((current_x, current_y), word, font=font, fill=text_color)
            
            # Move x position for next word
            word_bbox = draw.textbbox((0, 0), word + " ", font=font)
            current_x += word_bbox[2]
        
        current_y += line_height
    
    return current_y


def load_and_resize_logo(logo_path, max_width, max_height):
    """Load logo and resize to fit within max dimensions."""
    if not os.path.exists(logo_path):
        print(f"Warning: Logo not found at {logo_path}")
        return None
    
    logo = Image.open(logo_path).convert("RGBA")
    
    # Calculate scale to fit within max dimensions
    width_scale = max_width / logo.width
    height_scale = max_height / logo.height
    scale = min(width_scale, height_scale)
    
    new_width = int(logo.width * scale)
    new_height = int(logo.height * scale)
    
    return logo.resize((new_width, new_height), Image.Resampling.LANCZOS)


def has_audio_stream(video_path):
    """Check if video has audio using ffprobe."""
    if not shutil.which("ffprobe"):
        print("Warning: ffprobe not found. Assuming video has audio.")
        return True
    
    command = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_type", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        video_path
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return len(result.stdout.strip()) > 0
    except subprocess.CalledProcessError:
        return False


def process_marketingspots_template(input_video_path, output_video_path, text_content, 
                                   config=None, logo_path_override=None):
    """
    Process video using MarketingSpots template.
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path to save output
        text_content: Text to display above video
        config: Optional custom config dict (uses TEMPLATE_CONFIG if None)
        logo_path_override: Optional path to logo (overrides config)
    """
    
    if config is None:
        config = TEMPLATE_CONFIG
    
    # Check FFmpeg
    if not shutil.which("ffmpeg"):
        print("\n" + "="*50)
        print("ERROR: FFmpeg not found.")
        print("Please install FFmpeg and add it to your system's PATH.")
        print("="*50)
        return False
    
    # Extract config values
    canvas_width = config["canvas"]["width"]
    canvas_height = config["canvas"]["height"]
    bg_color = hex_to_rgb(config["canvas"]["background_color"])
    
    text_config = config["text"]
    font_path = text_config["font_family"]
    font_size = text_config["font_size"]
    text_color = hex_to_rgb(text_config["color"])
    highlight_color = hex_to_rgb(text_config["highlight_color"])
    text_offset_y = text_config["offset"]["y"]
    max_text_width = text_config["max_width"]
    line_spacing = text_config["line_spacing"]
    
    logo_config = config["logo"]
    logo_path = logo_path_override or logo_config["path"]
    
    # Check font
    if not os.path.exists(font_path):
        print(f"Error: Font file not found: {font_path}")
        return False
    
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {input_video_path}")
        return False
    
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if math.isnan(fps) or fps <= 0:
        print("Warning: FPS not reported by source. Defaulting to 30fps.")
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input video: {video_width}x{video_height}, {fps} fps, {total_frames} frames")
    
    # Calculate video placement based on aspect ratio
    video_aspect = video_width / video_height
    
    if video_aspect < 1.0:
        # Portrait video (taller than wide): use portrait_width_percent, center both axes
        portrait_width_percent = config["video"]["portrait_width_percent"]
        target_width = int(canvas_width * (portrait_width_percent / 100))
        scale = target_width / video_width
        scaled_width = target_width
        scaled_height = int(video_height * scale)
        video_x = (canvas_width - scaled_width) // 2  # Center horizontally
        video_y = (canvas_height - scaled_height) // 2  # Center vertically
    else:
        # Square or landscape video: full width, centered vertically
        scaled_width = canvas_width
        scale = canvas_width / video_width
        scaled_height = int(video_height * scale)
        video_x = 0
        video_y = (canvas_height - scaled_height) // 2
    
    print(f"Scaled video: {scaled_width}x{scaled_height} at position ({video_x}, {video_y})")
    
    # Prepare text with highlighting
    font = ImageFont.truetype(font_path, font_size)
    
    if text_config["highlight_mode"] == "ai":
        highlight_word = get_ai_highlight_word(text_content)
        print(f"AI selected highlight word: '{highlight_word}'")
    else:
        highlight_word = None
    
    lines_data = get_wrapped_lines_with_highlight(text_content, font, max_text_width, highlight_word)
    
    # Calculate text position - bottom of text should be 20px above video
    text_bottom_y = video_y - 20  # 20px gap above video
    
    print(f"Text: {len(lines_data)} line(s), bottom positioned at y={text_bottom_y}")
    
    # Load logo
    logo = load_and_resize_logo(logo_path, logo_config["max_width"], logo_config["max_height"])
    if logo:
        # Calculate logo position (top-right with offset)
        if logo_config["position"] == "top-right":
            logo_x = canvas_width + logo_config["offset"]["x"] - logo.width
            logo_y = logo_config["offset"]["y"]
        print(f"Logo: {logo.width}x{logo.height} at ({logo_x}, {logo_y})")
    
    # Process video frames
    temp_video_path = "temp_silent_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (canvas_width, canvas_height))
    
    for frame_count in range(1, total_frames + 1):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize video frame
        resized_frame = cv2.resize(frame, (scaled_width, scaled_height))
        
        # Create canvas with background color
        canvas_np = np.full((canvas_height, canvas_width, 3), 
                           tuple(reversed(bg_color)), dtype=np.uint8)
        
        # Place video on canvas
        if video_y < 0:
            crop_top = abs(video_y)
            crop_bottom = min(scaled_height - crop_top, canvas_height)
            canvas_np[0:crop_bottom, video_x:video_x+scaled_width] = \
                resized_frame[crop_top:crop_top+crop_bottom, :]
        else:
            end_y = min(video_y + scaled_height, canvas_height)
            canvas_np[video_y:end_y, video_x:video_x+scaled_width] = \
                resized_frame[0:end_y-video_y, :]
        
        # Convert to PIL for text and logo
        canvas_pil = Image.fromarray(cv2.cvtColor(canvas_np, cv2.COLOR_BGR2RGB))
        
        # Add logo
        if logo:
            canvas_pil.paste(logo, (logo_x, logo_y), logo)
        
        # Add text with highlighting
        draw = ImageDraw.Draw(canvas_pil)
        draw_centered_text_with_highlight(draw, lines_data, font, text_color, 
                                         highlight_color, canvas_width, text_bottom_y, line_spacing)
        
        # Convert back and write
        final_frame = cv2.cvtColor(np.array(canvas_pil), cv2.COLOR_RGB2BGR)
        out.write(final_frame)
        
        print(f"Processing: {frame_count}/{total_frames}", end='\r')
    
    print("\nVideo processing complete. Running ffmpeg mux/transcode...")
    cap.release()
    out.release()

    ffmpeg_base_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        temp_video_path,
    ]

    if has_audio_stream(input_video_path):
        print("Embedding original audio and transcoding to H.264/AAC...")
        ffmpeg_cmd = ffmpeg_base_cmd + [
            "-i",
            input_video_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            output_video_path,
        ]
    else:
        print("Source audio unavailable; exporting silent H.264 video...")
        ffmpeg_cmd = ffmpeg_base_cmd + [
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-movflags",
            "+faststart",
            output_video_path,
        ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError during ffmpeg processing: {e.stderr.decode(errors='ignore')}")
        return False
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
    
    print(f"\nComplete! Output saved to: {output_video_path}")
    return True


# --- Example Usage ---
if __name__ == "__main__":
    input_video = "input.mp4"
    output_video = "output_marketingspots.mp4"
    text = "Transform Your Business Today"
    
    # Option 1: Use default config
    process_marketingspots_template(input_video, output_video, text)
    
    # Option 2: Override logo path
    # process_marketingspots_template(input_video, output_video, text, 
    #                                logo_path_override="my_custom_logo.png")
    
    # Option 3: Custom config
    # custom_config = TEMPLATE_CONFIG.copy()
    # custom_config["text"]["highlight_color"] = "#00FF00"  # Green highlight
    # process_marketingspots_template(input_video, output_video, text, config=custom_config)
