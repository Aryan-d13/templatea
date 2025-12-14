"""
Auto Processor - Watermark Removal Orchestrator
Watches a folder for videos, generates .wmask files, and processes them when ready
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import queue
import threading
import time
from wmask_utils import (
    create_wmask, load_wmask, save_wmask, 
    get_combined_bbox, shapes_to_mask
)

# ==========================================
# CONFIGURATION
# ==========================================
BATCH_SIZE = 32
READ_AHEAD = 60
WATCH_INTERVAL = 2  # seconds

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ Hardware: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Load LaMa model
try:
    from simple_lama_inpainting import SimpleLama
    from PIL import Image
    simple_lama = SimpleLama(device=device)
    print("✓ LaMa Model Loaded Successfully")
except ImportError:
    print("❌ Error: Install with: pip install simple-lama-inpainting")
    sys.exit(1)

# ==========================================
# VIDEO PROCESSING (From magic_remove.py)
# ==========================================
def get_safe_roi(x, y, w, h, img_w, img_h, multiple=16, padding=32):
    """Calculate safe ROI with padding and alignment."""
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
    """Process a batch of frames using ROI inpainting."""
    rx, ry, rw, rh = roi_params
    crops = [f[ry:ry+rh, rx:rx+rw] for f in frames_rgb]
    
    cleaned_crops = []
    try:
        for crop in crops:
            res = simple_lama(crop, roi_mask)
            if isinstance(res, Image.Image): 
                res = np.array(res)
            cleaned_crops.append(res)
    except Exception as e:
        print(f"   ⚠ Batch Error: {e}")
        return frames_rgb

    results = []
    for i, frame in enumerate(frames_rgb):
        out_frame = frame.copy()
        out_frame[ry:ry+rh, rx:rx+rw] = cleaned_crops[i]
        results.append(out_frame)
    return results


def frame_reader(cap, queue_in, stop_event):
    """Thread: Read frames from video."""
    idx = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: 
            break
        queue_in.put((idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        idx += 1
    queue_in.put(None)


def frame_writer(out, queue_out, stop_event):
    """Thread: Write frames to output video."""
    while not stop_event.is_set():
        item = queue_out.get()
        if item is None: 
            break
        idx, frame_bgr = item
        out.write(frame_bgr)


def process_video(video_path: Path, output_dir: Path, wmask_data: dict):
    """Process a single video using the mask data."""
    video_path = Path(video_path)
    out_path = output_dir / f"CLEAN_{video_path.name}"
    
    if out_path.exists():
        print(f"   ⚠ Already processed: {video_path.name}")
        return True

    # Get bbox from wmask
    bbox = wmask_data.get('bbox')
    if not bbox:
        print(f"   ❌ No bbox found in wmask for {video_path.name}")
        return False

    orig_x, orig_y, orig_w, orig_h = bbox
    
    print(f"   🚀 Processing: {video_path.name}")
    print(f"      Watermark region: x={orig_x}, y={orig_y}, w={orig_w}, h={orig_h}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"   ❌ Cannot open video: {video_path.name}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate safe ROI
    rx, ry, rw, rh = get_safe_roi(orig_x, orig_y, orig_w, orig_h, width, height)
    roi_params = (rx, ry, rw, rh)
    
    # Create binary mask from shapes
    roi_mask = np.zeros((rh, rw), dtype=np.uint8)
    mx, my = orig_x - rx, orig_y - ry
    cv2.rectangle(roi_mask, (mx, my), (mx+orig_w, my+orig_h), 255, -1)

    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    # Threading queues
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
    start_time = time.time()

    try:
        while True:
            item = q_in.get()
            if item is None:
                # Process remaining frames
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
                    elapsed = time.time() - start_time
                    fps_speed = processed_count / elapsed if elapsed > 0 else 0
                    sys.stdout.write(f"\r      Progress: {processed_count}/{total_frames} frames ({fps_speed:.1f} fps)")
                    sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n   ⚠ Processing interrupted")
        stop_event.set()
        cap.release()
        out.release()
        return False
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
        stop_event.set()
        cap.release()
        out.release()
        return False
    
    # Cleanup
    q_out.put(None)
    t_read.join()
    t_write.join()
    cap.release()
    out.release()
    
    elapsed = time.time() - start_time
    fps_speed = total_frames / elapsed if elapsed > 0 else 0
    print(f"\n      ✅ Done! ({fps_speed:.1f} fps) -> {out_path.name}")
    
    return True


# ==========================================
# WATCHER LOOP
# ==========================================
def main():
    """Main watcher loop."""
    if len(sys.argv) < 2:
        print("Usage: python auto_processor.py <folder_path>")
        target_dir = Path(".")
    else:
        target_dir = Path(sys.argv[1])

    if not target_dir.exists():
        print(f"❌ Folder not found: {target_dir}")
        return

    print(f"\n{'='*60}")
    print(f"🎬 AUTO WATERMARK PROCESSOR")
    print(f"{'='*60}")
    print(f"📂 Watching: {target_dir.resolve()}")
    print(f"\n📋 Instructions:")
    print(f"   1. I'll create .wmask files for each video")
    print(f"   2. Open .wmask files with: python watermark_marker.py <file>")
    print(f"   3. Draw rectangles/circles over watermarks and save")
    print(f"   4. I'll automatically process the videos!")
    print(f"\n{'='*60}\n")

    output_dir = target_dir / "cleaned_output"
    output_dir.mkdir(exist_ok=True)

    processed_videos = set()

    try:
        while True:
            videos = list(target_dir.glob("*.mp4"))
            
            for vid in videos:
                # Skip output files
                if "CLEAN_" in vid.name:
                    continue
                
                wmask_path = vid.with_suffix('.wmask')
                
                # STEP 1: Create .wmask if missing
                if not wmask_path.exists():
                    try:
                        create_wmask(vid, wmask_path)
                        print(f"📝 Created: {wmask_path.name}")
                        print(f"   → Open with: python watermark_marker.py {wmask_path.name}")
                    except Exception as e:
                        print(f"⚠ Failed to create wmask for {vid.name}: {e}")
                    continue
                
                # STEP 2: Check if wmask has shapes and is unprocessed
                try:
                    wmask_data = load_wmask(wmask_path)
                except Exception as e:
                    print(f"⚠ Failed to load {wmask_path.name}: {e}")
                    continue
                
                # Skip if already processed
                if wmask_data.get('processed', False):
                    continue
                
                # Skip if no shapes drawn
                if not wmask_data.get('shapes'):
                    continue
                
                # Skip if bbox not calculated
                if not wmask_data.get('bbox'):
                    print(f"⚠ {wmask_path.name} has shapes but no bbox, updating...")
                    from wmask_utils import update_wmask_bbox
                    update_wmask_bbox(wmask_path)
                    wmask_data = load_wmask(wmask_path)
                
                # Skip if we already processed this
                if vid.name in processed_videos:
                    continue
                
                # STEP 3: Process the video!
                print(f"\n🎨 Mask detected for: {vid.name}")
                success = process_video(vid, output_dir, wmask_data)
                
                if success:
                    # Mark as processed
                    wmask_data['processed'] = True
                    save_wmask(wmask_path, wmask_data)
                    processed_videos.add(vid.name)
                    print(f"   ✅ Marked {wmask_path.name} as processed\n")
                else:
                    print(f"   ❌ Processing failed for {vid.name}\n")
            
            # Wait before next scan
            time.sleep(WATCH_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user. Goodbye!")


if __name__ == "__main__":
    main()
