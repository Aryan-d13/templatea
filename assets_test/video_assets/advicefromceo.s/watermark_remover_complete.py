# watermark_remover_final.py
# DROP THIS FILE NEXT TO YOUR VIDEOS AND RUN

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# ====================== CONFIG (ONLY CHANGE THESE IF NEEDED) ======================
WATERMARK_PATH = "mark.png"          # Your original watermark file (any size doesn't matter!)
X_RATIO = 0.501                      # Exact center from your adding script
Y_RATIO = 0.648
WATERMARK_OPACITY = 0.70             # 70% opacity
TARGET_WIDTH = 170                   # EXACT value from your ffmpeg script → never change this
# =================================================================================

def load_and_prepare_watermark():
    wm = cv2.imread(WATERMARK_PATH, cv2.IMREAD_UNCHANGED)
    if wm is None:
        raise FileNotFoundError(f"Put your watermark file as: {WATERMARK_PATH}")

    original_w = wm.shape[1]
    if original_w != TARGET_WIDTH:
        scale_factor = TARGET_WIDTH / original_w
        new_w = TARGET_WIDTH
        new_h = int(wm.shape[0] * scale_factor)
        wm = cv2.resize(wm, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    if wm.shape[2] == 4:  # has alpha
        alpha = wm[:, :, 3] / 255.0
        wm_rgb = wm[:, :, :3]
    else:
        alpha = np.ones((wm.shape[0], wm.shape[1]), dtype=np.float32)
        wm_rgb = wm

    return wm_rgb.astype(np.float32), alpha

def remove_watermark_video(input_path: str, output_path: str):
    wm_rgb, wm_alpha = load_and_prepare_watermark()
    h_w, w_w = wm_rgb.shape[:2]

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Cannot open video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    x = int(width * X_RATIO - w_w // 2)
    y = int(height * Y_RATIO - h_w // 2)

    # Safety clip
    x = max(0, x)
    y = max(0, y)

    print(f"Removing watermark {w_w}×{h_w} @ ({x}, {y}) from {Path(input_path).name}")

    pbar = tqdm(total=total_frames, desc="Processing", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        end_x = x + w_w
        end_y = y + h_w

        if end_x > width or end_y > height:
            out.write(frame)
            pbar.update(1)
            continue

        region = frame[y:end_y, x:end_x].astype(np.float32)
        alpha_mask = wm_alpha[:, :, np.newaxis] * WATERMARK_OPACITY

        # Magic reverse blend
        cleaned = (region - alpha_mask * wm_rgb) / (1 - alpha_mask + 1e-8)
        cleaned = np.clip(cleaned, 0, 255).astype(np.uint8)

        frame[y:end_y, x:end_x] = cleaned
        out.write(frame)
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()
    print(f"Saved clean video → {output_path}\n")

def main():
    parser = argparse.ArgumentParser(description="Perfectly remove 'startups BY INDIAN' watermark")
    parser.add_argument("--input", required=True, help="Path to video file or folder with videos")
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        output_file = input_path.stem + "_clean.mp4"
        remove_watermark_video(str(input_path), output_file)
    else:
        videos = list(input_path.glob("*.mp4")) + list(input_path.glob("*.mov")) + list(input_path.glob("*.avi"))
        print(f"Found {len(videos)} videos. Starting batch removal...\n")
        for video in videos:
            if "_clean" not in video.name:  # skip already processed
                remove_watermark_video(str(video), str(video.parent / f"{video.stem}_clean.mp4"))

if __name__ == "__main__":
    main()