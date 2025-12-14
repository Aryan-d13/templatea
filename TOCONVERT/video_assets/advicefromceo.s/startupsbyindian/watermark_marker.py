"""
Watermark Marker - GUI Application with Zoom & Pan
Interactive tool for drawing watermark masks on video frames
"""

import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
from wmask_utils import load_wmask, save_wmask, base64_to_frame, update_wmask_bbox


class WatermarkMarkerApp:
    def __init__(self, root, wmask_path=None):
        self.root = root
        self.root.title("Watermark Marker - Zoom Enabled")
        self.root.geometry("1200x800")
        
        # Data
        self.wmask_path = None
        self.wmask_data = None
        self.thumbnail_cv = None
        self.thumbnail_pil = None
        self.display_image = None
        self.photo = None
        
        # Drawing state
        self.current_tool = 'rectangle'
        self.drawing = False
        self.start_x = None
        self.start_y = None
        self.temp_shape = None
        self.polygon_points = []
        
        # Zoom and Pan state
        self.zoom_level = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        
        # Scale factor
        self.scale = 1.0
        self.base_scale = 1.0
        
        # Setup UI
        self.setup_ui()
        
        # Load wmask if provided
        if wmask_path:
            self.load_wmask_file(Path(wmask_path))
    
    def setup_ui(self):
        """Create the user interface."""
        # Menu Bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open .wmask", command=self.open_file)
        file_menu.add_command(label="Save", command=self.save_file, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Toolbar
        toolbar = ttk.Frame(self.root, padding=5)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Label(toolbar, text="Tool:").pack(side=tk.LEFT, padx=5)
        
        self.tool_var = tk.StringVar(value='rectangle')
        ttk.Radiobutton(toolbar, text="Rectangle (R)", variable=self.tool_var, 
                        value='rectangle', command=self.change_tool).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(toolbar, text="Circle (C)", variable=self.tool_var, 
                        value='circle', command=self.change_tool).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(toolbar, text="Polygon (P)", variable=self.tool_var, 
                        value='polygon', command=self.change_tool).pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        ttk.Button(toolbar, text="Clear All (Del)", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Undo Last", command=self.undo_last).pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
       
        # Zoom controls
        ttk.Label(toolbar, text="Zoom:").pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="➕", command=self.zoom_in, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="➖", command=self.zoom_out, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🔍 Fit", command=self.zoom_reset, width=6).pack(side=tk.LEFT, padx=2)
        
        self.zoom_label = ttk.Label(toolbar, text="100%", width=6)
        self.zoom_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        ttk.Button(toolbar, text="💾 Save", command=self.save_file,
                   style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        
        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = ttk.Label(status_frame, text="No file loaded | Scroll to zoom, Middle-click to pan", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=2)
        
        # Canvas
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(canvas_frame, bg='gray20', cursor='cross')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind events - Drawing
        self.canvas.bind('<Button-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.canvas.bind('<Button-3>', self.on_right_click)
        
        # Bind events - Zoom and Pan
        self.canvas.bind('<MouseWheel>', self.on_mouse_wheel)
        self.canvas.bind('<Button-4>', self.on_mouse_wheel)  # Linux scroll up
        self.canvas.bind('<Button-5>', self.on_mouse_wheel)  # Linux scroll down
        self.canvas.bind('<Button-2>', self.on_pan_start)
        self.canvas.bind('<B2-Motion>', self.on_pan_drag)
        self.canvas.bind('<ButtonRelease-2>', self.on_pan_end)
        
        # Keyboard shortcuts
        self.root.bind('<r>', lambda e: self.set_tool('rectangle'))
        self.root.bind('<c>', lambda e: self.set_tool('circle'))
        self.root.bind('<p>', lambda e: self.set_tool('polygon'))
        self.root.bind('<Delete>', lambda e: self.clear_all())
        self.root.bind('<Control-s>', lambda e: self.save_file())
        self.root.bind('<Control-z>', lambda e: self.undo_last())
        self.root.bind('<Control-plus>', lambda e: self.zoom_in())
        self.root.bind('<Control-equal>', lambda e: self.zoom_in())
        self.root.bind('<Control-minus>', lambda e: self.zoom_out())
        self.root.bind('<Control-0>', lambda e: self.zoom_reset())
    
    # Tool Management
    def change_tool(self):
        """Change the current drawing tool."""
        self.current_tool = self.tool_var.get()
        self.polygon_points = []
        self.update_status(f"Tool: {self.current_tool.capitalize()}")
    
    def set_tool(self, tool):
        """Set tool via keyboard shortcut."""
        self.tool_var.set(tool)
        self.change_tool()
    
    # File Operations
    def open_file(self):
        """Open a .wmask file."""
        filepath = filedialog.askopenfilename(
            title="Open .wmask file",
            filetypes=[("Watermark Mask", "*.wmask"), ("All Files", "*.*")]
        )
        if filepath:
            self.load_wmask_file(Path(filepath))
    
    def load_wmask_file(self, wmask_path: Path):
        """Load a .wmask file and display it."""
        try:
            self.wmask_path = wmask_path
            self.wmask_data = load_wmask(wmask_path)
            
            # Decode thumbnail
            self.thumbnail_cv = base64_to_frame(self.wmask_data['thumbnail'])
            self.thumbnail_pil = Image.fromarray(cv2.cvtColor(self.thumbnail_cv, cv2.COLOR_BGR2RGB))
            
            # Reset zoom/pan
            self.zoom_level = 1.0
            self.pan_offset_x = 0
            self.pan_offset_y = 0
            
            self.render_canvas()
            self.update_status(f"Loaded: {wmask_path.name} | Video: {self.wmask_data['video_file']}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")
    
    # Rendering
    def render_canvas(self):
        """Render the thumbnail with overlaid shapes, with zoom and pan."""
        if self.thumbnail_pil is None:
            return
        
        # Create a copy to draw on
        img = self.thumbnail_pil.copy()
        draw = ImageDraw.Draw(img, 'RGBA')
        
        # Draw existing shapes
        for shape in self.wmask_data.get('shapes', []):
            self.draw_shape_on_pil(draw, shape, outline='red', fill=(255, 0, 0, 80))
        
        # Draw temporary shape during drag
        if self.temp_shape:
            self.draw_shape_on_pil(draw, self.temp_shape, outline='yellow', fill=(255, 255, 0, 60))
        
        # Draw polygon points in progress
        if self.polygon_points:
            for px, py in self.polygon_points:
                draw.ellipse([px-3, py-3, px+3, py+3], fill='lime', outline='white')
        
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        # Calculate base scale to fit window
        img_w, img_h = self.thumbnail_pil.width, self.thumbnail_pil.height
        scale_w = canvas_width / img_w
        scale_h = canvas_height / img_h
        self.base_scale = min(scale_w, scale_h)
        
        # Apply zoom level
        self.scale = self.base_scale * self.zoom_level
        
        # Calculate new image size
        new_w = int(img_w * self.scale)
        new_h = int(img_h * self.scale)
        
        # Resize image
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Calculate image position with pan offset
        img_x = (canvas_width - new_w) // 2 + self.pan_offset_x
        img_y = (canvas_height - new_h) // 2 + self.pan_offset_y
        
        # Store for coordinate conversion
        self.display_image = img_resized
        self.photo = ImageTk.PhotoImage(img_resized)
        
        # Clear and draw
        self.canvas.delete('all')
        self.canvas.create_image(img_x, img_y, image=self.photo, anchor=tk.NW)
        
        # Update zoom label
        self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%")
    
    def draw_shape_on_pil(self, draw, shape, outline='red', fill=None):
        """Draw a shape on PIL ImageDraw."""
        shape_type = shape.get('type')
        
        if shape_type == 'rectangle':
            x, y, w, h = shape['x'], shape['y'], shape['width'], shape['height']
            draw.rectangle([x, y, x+w, y+h], outline=outline, fill=fill, width=2)
            
        elif shape_type == 'circle':
            cx, cy, r = shape['cx'], shape['cy'], shape['radius']
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=outline, fill=fill, width=2)
            
        elif shape_type == 'polygon':
            points = [(p[0], p[1]) for p in shape['points']]
            if len(points) >= 3:
                draw.polygon(points, outline=outline, fill=fill)
    
    # Coordinate Conversion
    def canvas_to_image_coords(self, canvas_x, canvas_y):
        """Convert canvas coordinates to original image coordinates."""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if self.display_image and self.thumbnail_pil:
            img_w, img_h = self.display_image.size
            
            # Calculate image position
            img_x = (canvas_width - img_w) // 2 + self.pan_offset_x
            img_y = (canvas_height - img_h) // 2 + self.pan_offset_y
            
            # Convert to image coordinates
            local_x = canvas_x - img_x
            local_y = canvas_y - img_y
            
            # Scale to original image size
            orig_x = int(local_x / self.scale)
            orig_y = int(local_y / self.scale)
            
            return orig_x, orig_y
        return canvas_x, canvas_y
    
    # Mouse Events - Drawing
    def on_mouse_down(self, event):
        """Handle mouse button press."""
        if self.wmask_data is None:
            return
        
        self.start_x, self.start_y = self.canvas_to_image_coords(event.x, event.y)
        
        if self.current_tool == 'polygon':
            self.polygon_points.append((self.start_x, self.start_y))
            self.render_canvas()
        else:
            self.drawing = True
    
    def on_mouse_drag(self, event):
        """Handle mouse drag."""
        if not self.drawing or self.wmask_data is None:
            return
        
        curr_x, curr_y = self.canvas_to_image_coords(event.x, event.y)
        
        if self.current_tool == 'rectangle':
            w = curr_x - self.start_x
            h = curr_y - self.start_y
            self.temp_shape = {
                'type': 'rectangle',
                'x': min(self.start_x, curr_x),
                'y': min(self.start_y, curr_y),
                'width': abs(w),
                'height': abs(h)
            }
        elif self.current_tool == 'circle':
            dx = curr_x - self.start_x
            dy = curr_y - self.start_y
            radius = int(np.sqrt(dx**2 + dy**2))
            self.temp_shape = {
                'type': 'circle',
                'cx': self.start_x,
                'cy': self.start_y,
                'radius': radius
            }
        
        self.render_canvas()
    
    def on_mouse_up(self, event):
        """Handle mouse button release."""
        if not self.drawing or self.wmask_data is None:
            return
        
        if self.temp_shape:
            # Add shape to list
            if self.temp_shape['type'] == 'rectangle' and self.temp_shape['width'] > 5 and self.temp_shape['height'] > 5:
                self.wmask_data['shapes'].append(self.temp_shape)
            elif self.temp_shape['type'] == 'circle' and self.temp_shape['radius'] > 5:
                self.wmask_data['shapes'].append(self.temp_shape)
            
            self.temp_shape = None
            self.render_canvas()
        
        self.drawing = False
    
    def on_right_click(self, event):
        """Finish polygon on right-click."""
        if self.current_tool == 'polygon' and len(self.polygon_points) >= 3:
            self.wmask_data['shapes'].append({
                'type': 'polygon',
                'points': self.polygon_points.copy()
            })
            self.polygon_points = []
            self.render_canvas()
    
    # Zoom and Pan Events
    def on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming."""
        if self.thumbnail_pil is None:
            return
        
        # Determine zoom direction
        if event.num == 4 or event.delta > 0:  # Scroll up
            self.zoom_in()
        elif event.num == 5 or event.delta < 0:  # Scroll down
            self.zoom_out()
    
    def on_pan_start(self, event):
        """Start panning with middle click."""
        self.panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.canvas.config(cursor='fleur')
    
    def on_pan_drag(self, event):
        """Pan the image."""
        if self.panning:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            self.pan_offset_x += dx
            self.pan_offset_y += dy
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self.render_canvas()
    
    def on_pan_end(self, event):
        """End panning."""
        self.panning = False
        self.canvas.config(cursor='cross')
    
    # Zoom Controls
    def zoom_in(self):
        """Zoom in (increase zoom level)."""
        if self.zoom_level < self.max_zoom:
            self.zoom_level *= 1.2
            if self.zoom_level > self.max_zoom:
                self.zoom_level = self.max_zoom
            self.render_canvas()
    
    def zoom_out(self):
        """Zoom out (decrease zoom level)."""
        if self.zoom_level > self.min_zoom:
            self.zoom_level /= 1.2
            if self.zoom_level < self.min_zoom:
                self.zoom_level = self.min_zoom
            self.render_canvas()
    
    def zoom_reset(self):
        """Reset zoom to fit window."""
        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.render_canvas()
    
    # Shape Management
    def clear_all(self):
        """Clear all shapes."""
        if self.wmask_data:
            if messagebox.askyesno("Clear All", "Remove all drawn shapes?"):
                self.wmask_data['shapes'] = []
                self.polygon_points = []
                self.temp_shape = None
                self.render_canvas()
    
    def undo_last(self):
        """Remove the last drawn shape."""
        if self.wmask_data and self.wmask_data.get('shapes'):
            self.wmask_data['shapes'].pop()
            self.render_canvas()
    
    # Save
    def save_file(self):
        """Save the .wmask file."""
        if self.wmask_path and self.wmask_data:
            try:
                # Update bbox
                update_wmask_bbox(self.wmask_path)
                
                # Reload to get updated bbox
                self.wmask_data = load_wmask(self.wmask_path)
                
                messagebox.showinfo("Saved", f"Saved to:\n{self.wmask_path.name}")
                self.update_status(f"Saved! Shapes: {len(self.wmask_data['shapes'])}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save:\n{e}")
        else:
            messagebox.showwarning("No File", "No .wmask file is loaded.")
    
    def update_status(self, message):
        """Update status bar."""
        self.status_label.config(text=message)


def main():
    root = tk.Tk()
    
    # Get wmask path from command line
    wmask_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    app = WatermarkMarkerApp(root, wmask_path)
    root.mainloop()


if __name__ == "__main__":
    main()
