# merge_audio.py
# Merges audio from original videos to cleaned videos in 'converted_videos' folder.

import subprocess
import sys
from pathlib import Path

def merge_audio():
    # Current directory is assumed to be the video folder
    root_dir = Path(".")
    converted_dir = root_dir / "converted_videos"
    final_dir = root_dir / "final_with_audio"
    
    if not converted_dir.exists():
        print("❌ 'converted_videos' folder not found!")
        return

    final_dir.mkdir(exist_ok=True)
    
    # Find cleaned videos
    clean_videos = list(converted_dir.glob("clean_*.mp4"))
    print(f"Found {len(clean_videos)} cleaned videos.")
    
    for clean_vid in clean_videos:
        # Extract ID from filename (e.g., clean_1.mp4 -> 1)
        # Assuming format clean_<name>.mp4 or clean_<id>.mp4
        # The user said: "original 1, 2... to clean_1, 2"
        
        filename = clean_vid.name
        if not filename.startswith("clean_"):
            continue
            
        original_name = filename.replace("clean_", "")
        original_vid = root_dir / original_name
        
        if not original_vid.exists():
            print(f"⚠ Original video not found for {filename}: {original_vid}")
            continue
            
        output_vid = final_dir / original_name
        
        print(f"Processing: {original_name}...")
        
        # FFmpeg command to copy video from clean and audio from original
        # -c copy: Fast stream copy (no re-encoding)
        # -map 0:v: Video from first input (clean)
        # -map 1:a: Audio from second input (original)
        # -shortest: Stop when shortest stream ends
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(clean_vid),      # Input 0: Clean Video (No Audio)
            "-i", str(original_vid),   # Input 1: Original Video (Has Audio)
            "-c", "copy",              # Copy streams directly
            "-map", "0:v:0",           # Use video from Input 0
            "-map", "1:a:0",           # Use audio from Input 1
            "-shortest",               # Match duration
            str(output_vid)
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"   ✓ Merged: {output_vid.name}")
        except subprocess.CalledProcessError:
            print(f"   ❌ Failed to merge: {original_name}")
        except FileNotFoundError:
            print("   ❌ FFmpeg not found! Please install FFmpeg.")
            return

    print("\nDone! Check 'final_with_audio' folder.")

if __name__ == "__main__":
    merge_audio()
