# orchestrator.py
"""
Minimal orchestrator — idempotent, file-specific, writes step.status JSON files.

Usage:
  python orchestrator.py --url "<REEL_URL>" [--auto]
  python orchestrator.py --scan-workspace [--auto]

Notes:
- Expects workspace/<id>/meta.json (or instagram_downloader.py to create it).
- Calls existing CLIs/scripts:
    - python instagram_downloader.py "<URL>"   (creates workspace)
    - python video_detector.py <in> <out> <threshold>
    - python gemini_ocr_batch.py <in> --out-dir <outdir>
    - uses marketingspots_template.process_marketingspots_template if importable else runs run_for_spots.py
- Writes workspace/<id>/<step>.status JSON with {"status","ts","error","retries"}
"""

import argparse, subprocess, json, time, os
from pathlib import Path
from datetime import datetime
from typing import Optional
import re
_EMOJI_RE = re.compile(
    "["                                    # character ranges
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed chars
    "]+",
    flags=re.UNICODE
)

def clean_text_for_render(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return _EMOJI_RE.sub("", text).strip()

try:
    from api.template_registry import get_renderer_func, TemplateRegistryError
except Exception:
    get_renderer_func = None
    TemplateRegistryError = RuntimeError

# Config
WORKSPACE_BASE = Path("workspace")
DETECTOR_THRESHOLD = "60"
RETRY_MAX = 9999
RETRY_BACKOFF = 2  # multiplier

def ts():
    return datetime.utcnow().isoformat() + "Z"

def write_status(ws: Path, step: str, status: str, error=None, retries=0):
    s = {"status": status, "ts": ts(), "error": None if error is None else str(error), "retries": retries}
    p = ws / f"{step}.status"
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf8") as f:
        json.dump(s, f)
    os.replace(str(tmp), str(p))


def read_meta(ws: Path) -> dict:
    meta_path = ws / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8-sig") as handle:
            try:
                return json.load(handle)
            except json.JSONDecodeError:
                return {}
    return {}


def write_meta(ws: Path, meta: dict) -> None:
    meta_path = ws / "meta.json"
    tmp = meta_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    os.replace(str(tmp), str(meta_path))

def run_cmd(cmd, cwd=None, timeout=600):
    """Run cmd (list). Return (retcode, stdout, stderr)."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as e:
        return 999, "", str(e)

def acquire_lock(ws: Path):
    lock = ws / ".lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False

def release_lock(ws: Path):
    lock = ws / ".lock"
    try:
        lock.unlink()
    except Exception:
        pass

def find_workspaces(scan_dir=WORKSPACE_BASE):
    if not scan_dir.exists():
        return []
    return sorted([p for p in scan_dir.iterdir() if p.is_dir()])

def ensure_detector(ws: Path):
    out_dir = ws / "01_detector"
    out_dir.mkdir(exist_ok=True)
    cropped = out_dir / "cropped.mp4"
    if cropped.exists():
        write_status(ws, "01_detector", "success", retries=0)
        return True
    src = ws / "00_raw" / "raw_source.mp4"
    if not src.exists():
        write_status(ws, "01_detector", "failed", error="raw_source.mp4 missing")
        return False

    # try with retries
    retries = 0
    while retries < RETRY_MAX:
        cmd = ["python", "video_detector.py", str(src), str(cropped), DETECTOR_THRESHOLD]
        code, out, err = run_cmd(cmd)
        if code == 0 and cropped.exists():
            write_status(ws, "01_detector", "success", retries=retries)
            return True
        retries += 1
        write_status(ws, "01_detector", "retrying", error=err, retries=retries)
        time.sleep((RETRY_BACKOFF ** retries))
    write_status(ws, "01_detector", "failed", error="max retries")
    return False

def ensure_ocr(ws: Path):
    ocr_dir = ws / "02_ocr"
    ocr_dir.mkdir(exist_ok=True)
    ocr_txt = ocr_dir / "ocr.txt"
    ai_json = ocr_dir / "ai_copies.json"

    if ocr_txt.exists() and ai_json.exists():
        write_status(ws, "02_ocr", "success")
        return True

    # Prefer the full raw video for OCR+cross-checking (per your note)
    raw_video = ws / "00_raw" / "raw_source.mp4"
    cropped = ws / "01_detector" / "cropped.mp4"
    if raw_video.exists():
        video_input = raw_video
    elif cropped.exists():
        video_input = cropped
    else:
        write_status(ws, "02_ocr", "failed", error="no raw_source.mp4 or cropped.mp4 found")
        return False

    # caption file available for Perplexity cross-check
    caption_file = ws / "00_raw" / "raw_caption.txt"

    retries = 0
    while retries < RETRY_MAX:
        json_report = ocr_dir / "gemini_report.json"
        # Build correct gemini_ocr_batch invocation (no --out-dir)
        cmd = ["python", "gemini_ocr_batch.py", str(video_input), "--json-report", str(json_report)]

        # If downloader caption exists, tell gemini_ocr_batch where to look
        if caption_file.exists():
            cmd.extend(["--caption-dir", str(ws / "00_raw")])
            cmd.extend(["--caption-extension", ".txt"])

        code, out, err = run_cmd(cmd, timeout=300)

        # write logs for debugging
        logs_dir = ws / "logs"
        logs_dir.mkdir(exist_ok=True)
        with open(logs_dir / "ocr.stdout.txt", "w", encoding="utf8") as f:
            f.write(out or "")
        with open(logs_dir / "ocr.stderr.txt", "w", encoding="utf8") as f:
            f.write(err or "")
        with open(logs_dir / "ocr.log", "w", encoding="utf8") as f:
            f.write(f"CMD: {' '.join(cmd)}\nRETURN: {code}\n\nSTDOUT:\n{out}\n\nSTDERR:\n{err}\n")

        # If gemini wrote the JSON report, parse it
        if code == 0 and json_report.exists():
            try:
                with open(json_report, "r", encoding="utf-8-sig") as f:
                    report = json.load(f)
                if isinstance(report, list) and len(report) > 0:
                    result = report[0]
                    # effective_caption set by gemini script; fallback to 'text'
                    caption_text = result.get("effective_caption") or result.get("text") or ""
                    # write OCR text (effective caption)
                    with open(ocr_txt, "w", encoding="utf8") as f:
                        f.write(caption_text)

                    # ai recommended copies (gemini script produces ai_recommended_copies as list of strings)
                    # ai recommended copies (expect list of either strings or dicts)
                    ai_copies_raw = result.get("ai_recommended_copies", []) or []
                    ai_copies_data = []

                    # Try to read a global reported source from the report if present
                    reported_source = result.get("ai_copy_source") or result.get("ai_copy_provider") or None

                    for idx, item in enumerate(ai_copies_raw, start=1):
                        if isinstance(item, dict):
                            text = item.get("text") or item.get("caption") or item.get("one_liner") or ""
                            source = item.get("source") or reported_source or "groq_direct"
                        else:
                            text = str(item)
                            source = reported_source or "groq_direct"
                        text = text.strip()
                        if not text:
                            continue
                        ai_copies_data.append({
                            "id": f"ai-copy-{idx}",
                            "text": text,
                            "source": source
                        })

                    # If no AI copies but we have caption, create a minimal fallback
                    if not ai_copies_data and caption_text:
                        ai_copies_data.append({"id": "fallback-1", "text": caption_text[:140], "source": "ocr_fallback"})
                    with open(ai_json, "w", encoding="utf8") as f:
                        json.dump(ai_copies_data, f, indent=2)
                    write_status(ws, "02_ocr", "success", retries=retries)
                    return True
            except Exception as e:
                write_status(ws, "02_ocr", "failed", error=f"Failed to parse report: {e}")

        # fallback: use downloader caption if available
        raw_caption = ws / "00_raw" / "raw_caption.txt"
        if raw_caption.exists():
            try:
                text = raw_caption.read_text(encoding="utf-8-sig").strip()
                with open(ocr_txt, "w", encoding="utf8") as f:
                    f.write(text)
                ac = [{"id": "fallback-1", "text": text[:140] if len(text) > 140 else text, "source": "raw_caption_fallback"}]
                with open(ai_json, "w", encoding="utf8") as f:
                    json.dump(ac, f, indent=2)
                write_status(ws, "02_ocr", "fallback_caption", retries=retries)
                return True
            except Exception as e:
                write_status(ws, "02_ocr", "failed", error=str(e))
                return False

        retries += 1
        write_status(ws, "02_ocr", "retrying", error=err, retries=retries)
        time.sleep((RETRY_BACKOFF ** retries))

    write_status(ws, "02_ocr", "failed", error="max retries")
    return False


def ensure_choice(ws: Path, auto=False):
    choice_dir = ws / "03_choice"
    choice_dir.mkdir(exist_ok=True)
    choice_file = choice_dir / "choice.txt"
    manual_file = choice_dir / "manual.txt"
    status_file = choice_dir / "03_choice.status"
    if choice_file.exists():
        write_status(ws, "03_choice", "success")
        return True
    # manual override
    if manual_file.exists():
        try:
            with open(manual_file, "r", encoding="utf-8-sig") as f:
                text = f.read().strip()
            with open(choice_file, "w", encoding="utf8") as f:
                f.write(text)
            write_status(ws, "03_choice", "manual_selected")
            return True
        except Exception as e:
            write_status(ws, "03_choice", "failed", error=str(e))
            return False
    # auto
    if auto:
        ai_json = ws / "02_ocr" / "ai_copies.json"
        if ai_json.exists():
            try:
                with open(ai_json, "r", encoding="utf-8-sig") as f:
                    ac = json.load(f)
                chosen = ac[0]["text"] if isinstance(ac, list) and len(ac) > 0 and "text" in ac[0] else str(ac[0])
                with open(choice_file, "w", encoding="utf8") as f:
                    f.write(chosen)
                write_status(ws, "03_choice", "auto_selected")
                return True
            except Exception as e:
                write_status(ws, "03_choice", "failed", error=str(e))
                return False
        else:
            write_status(ws, "03_choice", "pending_choice", error="no ai_copies", retries=0)
            return False
    # not auto -> pending
    write_status(ws, "03_choice", "pending_choice")
    return False

def ensure_render(ws: Path, template_id: Optional[str] = None):
    render_dir = ws / "04_render"
    render_dir.mkdir(exist_ok=True)
    final = render_dir / "final_1080x1920.mp4"
    if final.exists():
        write_status(ws, "04_render", "success")
        return True
    choice = ws / "03_choice" / "choice.txt"
    cropped = ws / "01_detector" / "cropped.mp4"
    if not cropped.exists():
        write_status(ws, "04_render", "failed", error="cropped missing")
        return False
    if not choice.exists():
        write_status(ws, "04_render", "failed", error="choice missing")
        return False

    meta = read_meta(ws)
    effective_template_id = template_id or meta.get("template_id")
    template_options = dict(meta.get("template_options", {}))
    text_value = choice.read_text(encoding="utf-8-sig").strip()

    text_value = clean_text_for_render(choice.read_text(encoding="utf-8-sig"))


    if effective_template_id and get_renderer_func:
        try:
            renderer = get_renderer_func(effective_template_id)
            ok = renderer(str(cropped), str(final), text_value, template_options)
            if ok is None or ok is True:
                write_status(ws, "04_render", "success")
                return True
            write_status(ws, "04_render", "failed", error="renderer returned False")
            return False
        except TemplateRegistryError as tre:
            write_status(ws, "04_render", "failed", error=str(tre))
            return False
        except Exception as exc:
            write_status(ws, "04_render", "failed", error=str(exc))
            return False

    # fallback legacy renderer
    try:
        from marketingspots_template import process_marketingspots_template
        ok = process_marketingspots_template(str(cropped), str(final), text_value)
        if ok is None or ok is True:
            write_status(ws, "04_render", "success")
            return True
        write_status(ws, "04_render", "failed", error="templater returned False")
        return False
    except Exception:
        cmd = ["python", "run_for_spots.py", str(cropped), str(final), str(choice)]
        code, out, err = run_cmd(cmd, timeout=300)
        if code == 0 and final.exists():
            write_status(ws, "04_render", "success")
            return True
        write_status(ws, "04_render", "failed", error=err)
        return False

def process_single_workspace(ws: Path, auto=False, template_id: Optional[str] = None):
    """Run all steps for single workspace path. Returns dict summary."""
    summary = {
        "id": ws.name,
        "downloaded": False,
        "detected": False,
        "ocr": False,
        "choice": False,
        "rendered": False,
    }
    meta = read_meta(ws)
    if template_id and meta.get("template_id") != template_id:
        meta["template_id"] = template_id
        write_meta(ws, meta)
    effective_template = meta.get("template_id")
    # download already handled by instagram_downloader or creator; if meta missing and url provided, caller should have run downloader.
    # Acquire lock
    if not acquire_lock(ws):
        return {"id": ws.name, "error": "locked"}
    try:
        # DETECTOR
        d = ensure_detector(ws)
        summary["detected"] = d
        # OCR
        o = ensure_ocr(ws)
        summary["ocr"] = o
        # CHOICE
        c = ensure_choice(ws, auto=auto)
        summary["choice"] = c
        # RENDER
        r = ensure_render(ws, template_id=effective_template) if c else False
        summary["rendered"] = r
        return summary
    finally:
        release_lock(ws)

def build_workspace_payload(ws_id: str) -> dict:
    ws_path = WORKSPACE_BASE / ws_id
    meta = read_meta(ws_path)
    payload = {"id": ws_id, "path": str(ws_path.resolve())}
    if meta.get("template_id"):
        payload["template_id"] = meta["template_id"]
    return payload


def emit_workspace_info(target: str, ws_id: str) -> None:
    payload = build_workspace_payload(ws_id)
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    if target in ("-", "stdout"):
        print(data)
        return
    out_path = Path(target)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf8") as handle:
        handle.write(data)
    os.replace(tmp, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="Instagram reel URL to download and process")
    ap.add_argument("--scan-workspace", action="store_true", help="Scan workspace directory for existing meta folders")
    ap.add_argument("--auto", action="store_true", help="Auto select AI copy when choice missing")
    ap.add_argument("--template-id", help="Template identifier to use for rendering")
    ap.add_argument("--emit-json-workspace", help="Path (or '-' for stdout) to emit workspace metadata JSON")
    args = ap.parse_args()

    summaries = []
    workspace_to_emit: Optional[str] = None
    pre_existing_ids = set()

    if args.url:
        # call downloader which creates workspace/<id>
        print("Calling downloader for URL...")
        pre_existing_ids = {p.name for p in find_workspaces()}
        code, out, err = run_cmd(["python", "instagram_downloader.py", args.url], timeout=600)
        if code != 0:
            print("Downloader failed:", err)
            # still attempt scan for newly created workspaces
        # short sleep to allow downloader to finish creating workspace
        time.sleep(1)
        post_ids = {p.name for p in find_workspaces()}
        new_ids = sorted(post_ids - pre_existing_ids)
        if new_ids:
            workspace_to_emit = new_ids[-1]

    if args.scan_workspace or args.url:
        wss = find_workspaces()
        if not wss:
            print("No workspaces found in", WORKSPACE_BASE)
        targets = wss
        if args.url and not args.scan_workspace:
            post_ids = {p.name for p in wss}
            new_candidates = sorted(post_ids - pre_existing_ids)
            if new_candidates:
                targets = [ws for ws in wss if ws.name in set(new_candidates)]
        for ws in targets:
            print("Processing", ws.name)
            res = process_single_workspace(ws, auto=args.auto, template_id=args.template_id)
            summaries.append(res)
            if workspace_to_emit is None:
                workspace_to_emit = ws.name

    # write report
    rep = {"ts": ts(), "workspaces": summaries}
    with open("orchestrator_report.json", "w", encoding="utf8") as f:
        json.dump(rep, f, indent=2)
    # compact print
    for s in summaries:
        print(f"{s.get('id')} | detected:{s.get('detected')} ocr:{s.get('ocr')} choice:{s.get('choice')} render:{s.get('rendered')}")
    print("Report written to orchestrator_report.json")
    if args.emit_json_workspace and workspace_to_emit:
        emit_workspace_info(args.emit_json_workspace, workspace_to_emit)

if __name__ == "__main__":
    main()
