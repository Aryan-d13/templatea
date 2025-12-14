import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Decide how many videos to process in parallel
MAX_WORKERS = min(4, (os.cpu_count() or 2))  # tweak if you want more/less

# Relative position from your 720x1280 screenshot at (361, 829)
X_RATIO = 0.501  # 361 / 720
Y_RATIO = 0.648  # 829 / 1280

def build_filter_complex():
    return (
        "[1]format=rgba,"
        "scale=min(iw\\,170):-1,"        # cap width at 170px, keep aspect
        "colorchannelmixer=aa=0.7[wm];"  # 70% opacity
        f"[0][wm]overlay=W*{X_RATIO}-w/2:H*{Y_RATIO}-h/2"
    )

def process_video(video_path: Path, wm_path: Path, out_dir: Path):
    out_file = out_dir / video_path.name

    filter_complex = build_filter_complex()

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-i", str(wm_path),
        "-filter_complex", filter_complex,
        "-c:a", "copy",
        str(out_file),
    ]

    print(f"[START] {video_path.name}")
    subprocess.run(cmd, check=True)
    print(f"[DONE ] {video_path.name}")

def main():
    root = Path(".").resolve()
    wm_path = root / "mark.png"
    out_dir = root / "converted"
    out_dir.mkdir(exist_ok=True)

    if not wm_path.exists():
        raise FileNotFoundError(f"Watermark image not found at {wm_path}")

    # collect all mp4s in root
    videos = sorted(root.glob("*.mp4"))
    if not videos:
        print("No .mp4 files found in this folder.")
        return

    print(f"Found {len(videos)} videos. Running with {MAX_WORKERS} workers.\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_video, v, wm_path, out_dir): v
            for v in videos
        }

        # just to surface any errors cleanly
        for fut in as_completed(futures):
            vid = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"[ERROR] {vid.name}: {e}")

    print("\nAll done. Check the 'converted' folder.")

if __name__ == "__main__":
    main()
