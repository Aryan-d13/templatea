import os
import re
import subprocess
from pathlib import Path

def extract_shortcode(url):
    patterns = [
        r'/reel/([A-Za-z0-9_-]+)',
        r'/p/([A-Za-z0-9_-]+)',
        r'/tv/([A-Za-z0-9_-]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Could not extract shortcode from URL")

def method1_instaloader(url, shortcode, output_dir="."):
    print("Trying Instaloader...")
    try:
        import instaloader
        from instaloader import Post
        L = instaloader.Instaloader(dirname_pattern=output_dir)
        post = Post.from_shortcode(L.context, shortcode)
        L.download_post(post, target=shortcode)
        print("Instaloader: SUCCESS")
        return True
    except Exception as e:
        print(f"Instaloader: FAILED - {e}")
        return False

def method2_ytdlp(url, output_dir="."):
    print("Trying yt-dlp...")
    try:
        output_template = "%(id)s.%(ext)s"
        cmd = ["yt-dlp", "-o", output_template, url]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("yt-dlp: SUCCESS")
            return True
        else:
            print(f"yt-dlp: FAILED - {result.stderr}")
            return False
    except Exception as e:
        print(f"yt-dlp: FAILED - {e}")
        return False

def download_instagram_reel(url, output_dir="."):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    try:
        shortcode = extract_shortcode(url)
    except Exception as e:
        print(f"Invalid URL: {e}")
        return False

    if method1_instaloader(url, shortcode, output_dir):
        return True
    if method2_ytdlp(url, output_dir):
        return True

    print("Both methods failed!")
    return False

if __name__ == "__main__":
    reel_url = "https://www.instagram.com/reel/DPimQwekRkq/?utm_source=ig_web_copy_link"
    download_instagram_reel(reel_url)
