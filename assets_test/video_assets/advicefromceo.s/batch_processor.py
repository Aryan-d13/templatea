# batch_processor.py
# ⚡ BATCH PIPELINE FOR WATERMARK REMOVAL
# Scans subfolders, reads 'watermark_config.json', and processes videos.
# NOW INTERACTIVE: Prompts user to draw watermark if config is missing!

import cv2
import torch
import numpy as np
from pathlib import Path
import json
import sys
import queue
import threading
import time
import shutil

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
BATCH_SIZE = 32
READ_AHEAD = 120
Use_FP16 = True

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"✓ GPU Detected: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
else:
    print("❌ GPU NOT FOUND! This will be slow.")

# Load LaMa model
try:
    from simple_lama_inpainting import SimpleLama
    from PIL import Image
    simple_lama = SimpleLama(device=device)
    print("✓ LaMa Model Loaded")
except ImportError:
    print("❌ Install library: pip install simple-lama-inpainting")
    sys.exit()

# ==========================================
# INTERACTIVE ROI SELECTOR
# ==========================================
def select_roi_interactive(video_path):
    """
    Opens a window to let the user draw the watermark box.
    Returns (x, y, w, h) or None if cancelled.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("❌ Could not open video for ROI selection.")
        return None

    # Try to get a frame from the middle to ensure watermark is visible
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(100, total_frames // 2))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Could not read frame for ROI selection.")
        return None

    print("\n" + "="*50)
    print(f"🎯 CONFIGURING: {video_path.parent.name}")
    print("INSTRUCTIONS:")
    print("1. A window will pop up showing a frame.")
    print("2. Draw a box around the watermark using your mouse.")
    print("3. Press SPACE or ENTER to confirm the selection.")
    print("4. Press 'c' to cancel/skip this folder.")
    print("="*50 + "\n")

    while True:
        # Select ROI
        roi = cv2.selectROI("Select Watermark (Press SPACE to confirm, c to cancel)", frame, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
        
        x, y, w, h = roi
        
        # Check if cancelled
        if w == 0 or h == 0:
            print("⚠ Selection cancelled.")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry == 'y': continue
            else: return None
            
        print(f"\nSelected: x={x}, y={y}, w={w}, h={h}")
        
        # Visual Confirmation
        preview = frame.copy()
        cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.imshow("Confirm Selection (Press any key to close)", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        confirm = input("Are you satisfied with this selection? (y/n): ").strip().lower()
        if confirm == 'y':
            return (int(x), int(y), int(w), int(h))
        else:
            print("↺ Retrying...")

# ==========================================
# CORE LOGIC (Optimized ROI)
# ==========================================
def get_safe_roi(x, y, w, h, img_w, img_h, multiple=16, padding=32):
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)
    
    cw, ch = x2 - x1, y2 - y1
    target_w = ((cw + multiple - 1) // multiple) * multiple
    target_h = ((ch + multiple - 1) // multiple) * multiple
    
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    nx1 = max(0, cx - target_w // 2)
    ny1 = max(0, cy - target_h // 2)
    nx2 = min(img_w, nx1 + target_w)
    ny2 = min(img_h, ny1 + target_h)
    
    if (nx2 - nx1) != target_w: nx1 = max(0, nx2 - target_w)
    if (ny2 - ny1) != target_h: ny1 = max(0, ny2 - target_h)
        
    return nx1, ny1, nx2 - nx1, ny2 - ny1

def process_batch_roi(frames_rgb, roi_params, roi_mask):
    rx, ry, rw, rh = roi_params
    crops = [f[ry:ry+rh, rx:rx+rw] for f in frames_rgb]
    
    cleaned_crops = []
    try:
        for crop in crops:
            res = simple_lama(crop, roi_mask)
            if isinstance(res, Image.Image): res = np.array(res)
            cleaned_crops.append(res)
    except Exception as e:
        print(f"Batch Error: {e}")
        return frames_rgb

    results = []
    for i, frame in enumerate(frames_rgb):
        out_frame = frame.copy()
        out_frame[ry:ry+rh, rx:rx+rw] = cleaned_crops[i]
        results.append(out_frame)
    return results

def frame_reader(cap, queue_in, stop_event):
    idx = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: break
        queue_in.put((idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        idx += 1
    queue_in.put(None)

def frame_writer(out, queue_out, stop_event):
    while not stop_event.is_set():
        item = queue_out.get()
        if item is None: break
        idx, frame_bgr = item
        out.write(frame_bgr)

def process_video(video_path, output_dir, config):
    video_path = Path(video_path)
    out_path = output_dir / f"clean_{video_path.name}"
    
    if out_path.exists(): return "skipped"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return "error"

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Parse Config
    bbox = config.get("bbox")
    if not bbox:
        print("❌ Config missing 'bbox' [x, y, w, h]")
        return "error"
    
    orig_x, orig_y, orig_w, orig_h = bbox
    
    # Calculate ROI
    rx, ry, rw, rh = get_safe_roi(orig_x, orig_y, orig_w, orig_h, width, height)
    roi_params = (rx, ry, rw, rh)
    
    # Create Mask
    roi_mask = np.zeros((rh, rw), dtype=np.uint8)
    mx, my = orig_x - rx, orig_y - ry
    cv2.rectangle(roi_mask, (mx, my), (mx+orig_w, my+orig_h), 255, -1)

    # Setup Pipeline
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    q_in = queue.Queue(maxsize=READ_AHEAD)
    q_out = queue.Queue(maxsize=READ_AHEAD)
    stop_event = threading.Event()

    t_read = threading.Thread(target=frame_reader, args=(cap, q_in, stop_event), daemon=True)
    t_write = threading.Thread(target=frame_writer, args=(out, q_out, stop_event), daemon=True)
    t_read.start()
    t_write.start()

    batch_frames = []
    batch_indices = []
    processed_count = 0

    print(f"▶ Processing: {video_path.name}")
    start_time = time.time()

    try:
        while True:
            item = q_in.get()
            if item is None:
                if batch_frames:
                    res = process_batch_roi(batch_frames, roi_params, roi_mask)
                    for i, r in zip(batch_indices, res):
                        q_out.put((i, cv2.cvtColor(r, cv2.COLOR_RGB2BGR)))
                break

            idx, frame = item
            batch_frames.append(frame)
            batch_indices.append(idx)

            if len(batch_frames) >= BATCH_SIZE:
                res = process_batch_roi(batch_frames, roi_params, roi_mask)
                for i, r in zip(batch_indices, res):
                    q_out.put((i, cv2.cvtColor(r, cv2.COLOR_RGB2BGR)))
                processed_count += len(batch_frames)
                batch_frames = []
                batch_indices = []
                
                if processed_count % 60 == 0:
                    sys.stdout.write(f"\r   Frames: {processed_count}/{total_frames}")
                    sys.stdout.flush()

    except KeyboardInterrupt:
        stop_event.set()
        return "interrupted"
    except Exception as e:
        print(f"Error: {e}")
        stop_event.set()
        return "error"
    
    q_out.put(None)
    t_read.join()
    t_write.join()
    cap.release()
    out.release()
    
    elapsed = time.time() - start_time
    fps_speed = total_frames / elapsed if elapsed > 0 else 0
    print(f"\n   Done! Speed: {fps_speed:.2f} fps")
    return "success"

# ==========================================
# BATCH SCANNER
# ==========================================
def main():
    root_dir = Path(".")
    subfolders = [f for f in root_dir.iterdir() if f.is_dir()]
    
    print(f"📂 Found {len(subfolders)} folders to scan...")
    
    for folder in subfolders:
        # Skip output folders if they exist
        if folder.name == "converted_videos": continue
        
        videos = list(folder.glob("*.mp4"))
        if not videos:
            continue
            
        print(f"\n📁 Entering Folder: {folder.name}")
        
        config_file = folder / "watermark_config.json"
        config = None
        
        # 1. Check for existing config
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                print("   ✓ Loaded existing config.")
            except Exception as e:
                print(f"   ❌ Config corrupted: {e}")
        
        # 2. Interactive Setup if missing
        if not config:
            print("   ⚠ No config found. Launching Interactive Setup...")
            # Use the first video to set coordinates
            bbox = select_roi_interactive(videos[0])
            
            if bbox:
                config = {
                    "mode": "static_bbox",
                    "bbox": bbox,
                    "notes": f"Generated interactively for {folder.name}"
                }
                # Save it
                with open(config_file, "w") as f:
                    json.dump(config, f, indent=4)
                print(f"   ✓ Config saved to {config_file.name}")
            else:
                print("   ⏭ Skipping folder (No config selected).")
                continue
            
        # 3. Process Videos
        output_dir = folder / "cleaned"
        output_dir.mkdir(exist_ok=True)
        
        print(f"   Found {len(videos)} videos.")
        
        for vid in videos:
            if "clean_" in vid.name: continue
            res = process_video(vid, output_dir, config)
            if res == "interrupted":
                print("\n🛑 Batch processing stopped by user.")
                return

if __name__ == "__main__":
    main()
