# Quick Start Guide - Watermark Removal

## 🚀 Quick Setup

### 1. Start the Auto-Processor

```bash
cd e:\Code\Templatea\TOCONVERT\video_assets\advicefromceo.s\startupsbyindian
..\..\..\..\venv\Scripts\python.exe auto_processor.py .
```

This will:

- Scan for all `.mp4` files
- Create `.wmask` files (one per video)
- Wait for you to mark watermarks

**Leave this running** - it watches for changes!

---

## 🎨 2. Mark Watermarks

Open a `.wmask` file in the GUI:

```bash
..\..\..\..\venv\Scripts\python.exe watermark_marker.py 1.wmask
```

### Drawing Tools

- **Press `R`** - Rectangle tool (best for most watermarks)
- **Press `C`** - Circle tool
- **Press `P`** - Polygon tool (for complex shapes)

### Steps

1. Select Rectangle tool (R)
2. Click and drag over the watermark
3. Release mouse button
4. Press **Ctrl+S** to save
5. Close the window

---

## ✅ 3. Processing Happens Automatically

Once you save, check **Terminal 1** (auto_processor):

```
🎨 Mask detected for: 1.mp4
   🚀 Processing: 1.mp4
      Progress: 120/300 frames (8.5 fps)
      ✅ Done! (9.2 fps) -> CLEAN_1.mp4
```

Cleaned video saved to: `cleaned_output/CLEAN_1.mp4`

---

## 📁 Files Explained

| File | Purpose |
|------|---------|
| `1.mp4` | Your original video |
| `1.wmask` | Mask data (JSON with thumbnail) |
| `cleaned_output/CLEAN_1.mp4` | Watermark removed! |

---

## 🔄 Workflow Summary

```
Run auto_processor.py
       ↓
Open .wmask in GUI
       ↓
Draw rectangle over watermark
       ↓
Save (Ctrl+S)
       ↓
Auto-processor detects & processes
       ↓
Get CLEAN_*.mp4 in cleaned_output/
```

---

## 💡 Tips

- **Keep auto_processor running** - It watches continuously
- **Draw with margin** - Include a bit of padding around watermarks
- **Multiple shapes** - Draw as many rectangles/circles as needed
- **Undo mistakes** - Press Ctrl+Z to remove last shape
- **Clear all** - Press Delete to start over

---

## 🎯 Your Turn

1. Start auto_processor (Terminal 1)
2. Open `1.wmask` (Terminal 2)
3. Draw a rectangle over the watermark
4. Save with Ctrl+S
5. Watch the magic happen! ✨
