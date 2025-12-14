"""
Watermark Mask (.wmask) Utilities
Handles creation, loading, and manipulation of .wmask files
"""

import json
import base64
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def extract_middle_frame(video_path: Path) -> Optional[np.ndarray]:
    """Extract the middle frame from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None
    
    # Go to middle frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    cap.release()
    
    return frame if ret else None


def frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Convert OpenCV frame to base64-encoded JPEG string."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')


def base64_to_frame(b64_string: str) -> np.ndarray:
    """Convert base64-encoded JPEG string to OpenCV frame."""
    img_data = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def create_wmask(video_path: Path, output_path: Optional[Path] = None) -> Path:
    """
    Create a new .wmask file for a video.
    
    Args:
        video_path: Path to the video file
        output_path: Optional custom output path. If None, uses video_name.wmask
    
    Returns:
        Path to the created .wmask file
    """
    if output_path is None:
        output_path = video_path.with_suffix('.wmask')
    
    # Extract middle frame
    frame = extract_middle_frame(video_path)
    if frame is None:
        raise ValueError(f"Could not extract frame from {video_path}")
    
    # Create wmask data structure
    wmask_data = {
        "video_file": video_path.name,
        "video_path": str(video_path.resolve()),
        "thumbnail": frame_to_base64(frame),
        "frame_width": frame.shape[1],
        "frame_height": frame.shape[0],
        "shapes": [],
        "bbox": None,
        "processed": False
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(wmask_data, f, indent=2)
    
    return output_path


def load_wmask(wmask_path: Path) -> Dict:
    """Load and parse a .wmask file."""
    with open(wmask_path, 'r') as f:
        return json.load(f)


def save_wmask(wmask_path: Path, data: Dict) -> None:
    """Save wmask data to file."""
    with open(wmask_path, 'w') as f:
        json.dump(data, f, indent=2)


def get_combined_bbox(shapes: List[Dict]) -> Optional[Tuple[int, int, int, int]]:
    """
    Calculate the bounding box that encompasses all shapes.
    
    Args:
        shapes: List of shape dictionaries
    
    Returns:
        (x, y, width, height) or None if no shapes
    """
    if not shapes:
        return None
    
    min_x = float('inf')
    min_y = float('inf')
    max_x = 0
    max_y = 0
    
    for shape in shapes:
        shape_type = shape.get('type')
        
        if shape_type == 'rectangle':
            x, y, w, h = shape['x'], shape['y'], shape['width'], shape['height']
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)
            
        elif shape_type == 'circle':
            cx, cy, r = shape['cx'], shape['cy'], shape['radius']
            min_x = min(min_x, cx - r)
            min_y = min(min_y, cy - r)
            max_x = max(max_x, cx + r)
            max_y = max(max_y, cy + r)
            
        elif shape_type == 'polygon':
            points = shape['points']
            for px, py in points:
                min_x = min(min_x, px)
                min_y = min(min_y, py)
                max_x = max(max_x, px)
                max_y = max(max_y, py)
    
    if min_x == float('inf'):
        return None
    
    return (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))


def shapes_to_mask(shapes: List[Dict], width: int, height: int) -> np.ndarray:
    """
    Convert shapes to a binary mask image.
    
    Args:
        shapes: List of shape dictionaries
        width: Mask width
        height: Mask height
    
    Returns:
        Binary mask (0 or 255) as uint8 numpy array
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for shape in shapes:
        shape_type = shape.get('type')
        
        if shape_type == 'rectangle':
            x, y, w, h = shape['x'], shape['y'], shape['width'], shape['height']
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            
        elif shape_type == 'circle':
            cx, cy, r = shape['cx'], shape['cy'], shape['radius']
            cv2.circle(mask, (cx, cy), r, 255, -1)
            
        elif shape_type == 'polygon':
            points = np.array(shape['points'], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
    
    return mask


def update_wmask_bbox(wmask_path: Path) -> None:
    """Update the bbox field in a .wmask file based on current shapes."""
    data = load_wmask(wmask_path)
    data['bbox'] = get_combined_bbox(data['shapes'])
    save_wmask(wmask_path, data)
