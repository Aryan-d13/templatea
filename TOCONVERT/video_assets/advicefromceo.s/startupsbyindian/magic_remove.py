import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import queue
import threading
import time
import os

# ==========================================
# CONFIGURATION
# ==========================================
BATCH_SIZE = 32
READ_AHEAD = 60
PREVIEW_EXT = ".draw_mask.png"  # The extension for the image you draw on

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ Hardware: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# Load LaMa model
try:
    from simple_lama_inpainting import SimpleLama
    from PIL import Image
    simple_lama = SimpleLama(device=device)
    print("✓ LaMa Model Loaded Successfully")
except ImportError:
    print("❌ Error: Library missing. Run: pip install simple-lama-inpainting")
    sys.exit()

# ==========================================
# CORE LOGIC: DETECT DRAWING
# ==========================================
def get_middle_frame(video_path):
    """Extracts the middle frame of the video without advancing the counter permanently."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    return None

def detect_drawing_bbox(video_path, image_path):
    """
    Compares the video's actual middle frame vs the saved .png file.
    Returns the bounding box (x,y,w,h) of the area the user drew on.
    """
    original_frame = get_middle_frame(video_path)
    edited_image = cv2.imread(str(image_path))

    if original_frame is None or edited_image is None:
        return None

    # Ensure dimensions match (in case user resized)
    if original_frame.shape != edited_image.shape:
        edited_image = cv2.resize(edited_image, (original_frame.shape[1], original_frame.shape[0]))

    # 1. Calculate Difference (Absolute Diff)
    # This finds pixels that are different between the clean video and your drawing
    diff = cv2.absdiff(original_frame, edited_image)
    
    # 2. Convert to grayscale and threshold
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Threshold > 10 to ignore minor compression noise. 
    # Any pixel you colored will have a large difference.
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # 3. Find Contours of the drawing
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None # No drawing detected yet

    # 4. Get the bounding box of ALL drawn areas combined
    all_points = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(all_points)

    # Minimum area check to avoid accidental single pixel clicks
    if w < 5 or h < 5:
        return None

    return (x, y, w, h)

# ==========================================
# AI PROCESSING (Unchanged logic)
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

def run_removal_job(video_path, output_dir, bbox):
    video_path = Path(video_path)
    out_path = output_dir / f"CLEAN_{video_path.name}"
    
    if out_path.exists():
        print(f"   ⚠ Skipping {video_path.name}, already cleaned.")
        return

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    orig_x, orig_y, orig_w, orig_h = bbox
    rx, ry, rw, rh = get_safe_roi(orig_x, orig_y, orig_w, orig_h, width, height)
    roi_params = (rx, ry, rw, rh)
    
    # Create binary mask for AI based on what user drew
    roi_mask = np.zeros((rh, rw), dtype=np.uint8)
    mx, my = orig_x - rx, orig_y - ry
    cv2.rectangle(roi_mask, (mx, my), (mx+orig_w, my+orig_h), 255, -1)

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
    
    print(f"   🚀 Processing {video_path.name}...")
    
    while True:
        item = q_in.get()
        if item is None:
            if batch_frames:
                res = process_batch_roi(batch_frames, roi_params, roi_mask)
                for i, r in zip(batch_indices, res): q_out.put((i, cv2.cvtColor(r, cv2.COLOR_RGB2BGR)))
            break
        
        idx, frame = item
        batch_frames.append(frame)
        batch_indices.append(idx)

        if len(batch_frames) >= BATCH_SIZE:
            res = process_batch_roi(batch_frames, roi_params, roi_mask)
            for i, r in zip(batch_indices, res): q_out.put((i, cv2.cvtColor(r, cv2.COLOR_RGB2BGR)))
            batch_frames = []
            batch_indices = []

    q_out.put(None)
    t_read.join()
    t_write.join()
    cap.release()
    out.release()
    print(f"   ✅ Finished: {out_path.name}")

# ==========================================
# WATCHER LOOP
# ==========================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <folder_path>")
        target_dir = Path(".")
    else:
        target_dir = Path(sys.argv[1])

    if not target_dir.exists():
        print("❌ Folder not found.")
        return

    print(f"\n👀 WATCHING FOLDER: {target_dir.resolve()}")
    print(f"ℹ  I will generate '*{PREVIEW_EXT}' files.")
    print(f"ℹ  Open them, draw a box (any color) over the watermark, and SAVE.")
    print(f"ℹ  I will detect the change and process the video.\n")

    output_dir = target_dir / "cleaned_output"
    output_dir.mkdir(exist_ok=True)

    processed_files = set()

    while True:
        videos = list(target_dir.glob("*.mp4"))
        
        for vid in videos:
            # Skip processed or output files
            if "CLEAN_" in vid.name: continue
            if vid.name in processed_files: continue

            preview_path = vid.with_suffix(PREVIEW_EXT)

            # 1. If preview doesn't exist, CREATE IT
            if not preview_path.exists():
                frame = get_middle_frame(vid)
                if frame is not None:
                    cv2.imwrite(str(preview_path), frame)
                    print(f"📝 Generated preview: {preview_path.name} (Waiting for drawing...)")
                continue

            # 2. Check if user has drawn on it
            bbox = detect_drawing_bbox(vid, preview_path)
            
            if bbox:
                print(f"\n🎨 Drawing Detected on {preview_path.name}!")
                print(f"   Target Box: {bbox}")
                
                # Run the job
                run_removal_job(vid, output_dir, bbox)
                
                # Mark as done so we don't re-process forever
                processed_files.add(vid.name)
                
                # Optional: Delete the preview file to clean up?
                # os.remove(preview_path) 
                
                print("   💤 Waiting for next drawing...")

        time.sleep(2) # Check every 2 seconds

if __name__ == "__main__":
    main()