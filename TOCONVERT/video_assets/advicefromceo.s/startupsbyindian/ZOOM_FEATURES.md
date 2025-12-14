# Zoom & Pan Features - Quick Guide

## 🔍 New Zoom Features Added

I've added comprehensive zoom and pan capabilities to the watermark marker GUI.

### Mouse Controls

| Action | Method |
|--------|--------|
| **Zoom In/Out** | Scroll mouse wheel 🖱️ |
| **Pan Image** | Middle-click + drag 🖱️ |
| **Draw Shapes** | Left-click + drag (as before) |
| **Finish Polygon** | Right-click (as before) |

### Toolbar Controls

- **➕ Button** - Zoom in (1.2x)
- **➖ Button** - Zoom out (1.2x)
- **🔍 Fit Button** - Reset zoom to fit window
- **Zoom %** - Shows current zoom level (10% to 1000%)

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl` + `+` | Zoom in |
| `Ctrl` + `-` | Zoom out |
| `Ctrl` + `0` | Reset zoom (fit to window) |

### Zoom Range

- **Minimum**: 10% (0.1x)
- **Maximum**: 1000% (10x)
- **Default**: 100% (fit to window)

---

## 💡 How to Use

### For Precise Watermark Marking

1. **Open a .wmask file** in the GUI
2. **Scroll to zoom in** on the watermark area
3. **Middle-click + drag** to pan if needed
4. **Draw your rectangle/circle** with precision
5. **Scroll out** or press `Ctrl+0` to see full image
6. **Save** (`Ctrl+S`)

### Example Workflow

```
1. Load 1.wmask
2. Scroll wheel up 3-4 times (zoom to ~200%)
3. Middle-click drag to center watermark
4. Press 'R' for rectangle tool
5. Draw precise box around watermark
6. Ctrl+0 to reset view and verify
7. Ctrl+S to save
```

---

## 🎯 Benefits

✅ **Pixel-perfect accuracy** - Zoom in to see exactly where watermark edges are  
✅ **Small watermarks** - Easier to mark tiny logos or text  
✅ **Complex shapes** - Precise polygon points for irregular watermarks  
✅ **Full control** - Pan to any part of the image while zoomed

---

## Technical Details

- **Smooth zooming** - 1.2x increments (20% per step)
- **Coordinate mapping** - Correctly translates zoom/pan to original image coordinates
- **Performance** - Image is resized only when zoom changes
- **Visual feedback** - Zoom percentage always visible in toolbar

---

## Status Bar Hint

The status bar now shows:
> `No file loaded | Scroll to zoom, Middle-click to pan`

This reminds you of the zoom/pan controls!
