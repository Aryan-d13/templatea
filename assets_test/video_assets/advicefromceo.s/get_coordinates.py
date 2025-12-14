import cv2
import json
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

def select_video():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a Video to Find Watermark",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )
    return file_path

def get_coordinates():
    video_path = select_video()
    if not video_path:
        print("No video selected.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video.")
        return

    # Read a random frame (e.g., 100th frame) to ensure watermark is visible
    # or just the first frame if video is short
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(100, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Could not read frame.")
        return

    print("\nINSTRUCTIONS:")
    print("1. Draw a box around the watermark.")
    print("2. Press ENTER or SPACE to confirm.")
    print("3. Press c to cancel.")
    
    # Resize for screen if too huge (optional, but good for 4K videos)
    # For now, we assume standard 1080p/720p fits.
    
    roi = cv2.selectROI("Select Watermark", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = roi
    
    if w == 0 or h == 0:
        print("Selection cancelled.")
        return

    print(f"\n✅ Selected Coordinates: x={x}, y={y}, w={w}, h={h}")
    
    config = {
        "mode": "static_bbox",
        "bbox": [int(x), int(y), int(w), int(h)],
        "notes": f"Generated for {Path(video_path).name}"
    }
    
    json_output = json.dumps(config, indent=4)
    print("\nCopy this into your 'watermark_config.json':")
    print("-" * 40)
    print(json_output)
    print("-" * 40)
    
    # Option to save directly
    save = input("Do you want to save this to 'watermark_config.json' in the video's folder? (y/n): ")
    if save.lower().startswith('y'):
        config_path = Path(video_path).parent / "watermark_config.json"
        with open(config_path, "w") as f:
            f.write(json_output)
        print(f"Saved to: {config_path}")

if __name__ == "__main__":
    get_coordinates()
