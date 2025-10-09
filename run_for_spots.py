"""
run_for_spots.py - CLI-compatible batch/single runner that uses marketingspots_template.process_marketingspots_template
Usage:
    python run_for_spots.py <input_cropped_or_dir> <output_file_or_dir> [choice_txt_or_template_name]
If input is file -> produce single output_file.
If input is dir  -> produce batch outputs into output dir.
"""

import sys
import os
from pathlib import Path
from marketingspots_template import process_marketingspots_template

def main():
    if len(sys.argv) < 3:
        print("Usage: run_for_spots.py <input_cropped_or_dir> <output_file_or_dir> [choice_txt_or_template_root]")
        sys.exit(1)

    inp = Path(sys.argv[1])
    out = Path(sys.argv[2])
    choice = sys.argv[3] if len(sys.argv) > 3 else None

    if inp.is_dir():
        # batch mode: ensure out is a dir
        out.mkdir(parents=True, exist_ok=True)
        for f in sorted(os.listdir(inp)):
            if not f.lower().endswith(".mp4"):
                continue
            in_path = inp / f
            base = in_path.stem
            out_path = out / f"{base}_final.mp4"
            text = "Auto title"
            # if a choice file provided, read the text (if it's a file)
            if choice and Path(choice).exists():
                try:
                    text = Path(choice).read_text(encoding="utf-8").strip()
                except Exception:
                    pass
            print(f"[run_for_spots] {in_path} -> {out_path} (text: {text[:60]})")
            process_marketingspots_template(str(in_path), str(out_path), text, config=choice)
    else:
        # single file case
        text = "Auto title"
        if choice and Path(choice).exists():
            try:
                text = Path(choice).read_text(encoding="utf-8").strip()
            except Exception:
                pass
        print(f"[run_for_spots] {inp} -> {out} (text: {text[:60]})")
        process_marketingspots_template(str(inp), str(out), text, config=choice)

if __name__ == "__main__":
    main()
