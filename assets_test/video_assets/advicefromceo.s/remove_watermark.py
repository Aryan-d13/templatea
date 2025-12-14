# Jaitika fixed it for her man ♡ – removes ONLY the tiny @advicefromceo.s text
import cv2
import numpy as np
from pathlib import Path

# Perfect box for 720×1280 videos
X, Y, W, H = 280, 821, 158, 52

# Create folder
Path("converted videos").mkdir(exist_ok=True)

print("♡ Starting removal – this time 100% perfect")

for video in Path(".").glob("*.mp4"):
    if video.name.startswith(".") or "_clean" in video.name.lower():
        continue

    print(f"♡ Cleaning {video.name} ...")

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path("converted videos") / video.name
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mask ONLY the tiny watermark area
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (X, Y), (X + W, Y + H), 255, -1)   # white = inpaint here
        clean_frame = cv2.inpaint(frame, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

        out.write(clean_frame)

    cap.release()
    out.release()
    print(f"♡ Done → converted videos/{video.name}")

print("\nAll clean and beautiful now my love ❤️ Check the 'converted videos' folder")
input("Press Enter when you're happy ♡")