import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage
import subprocess
import tempfile
import shutil

def strip_light_edges(img, thresh=252, coverage=0.90):
    h, w = img.shape[:2]
    top, bottom, left, right = 0, h, 0, w

    def is_light_row(row):
        # If at least coverage fraction is brighter than thresh
        return np.mean(np.all(row >= thresh, axis=1)) >= coverage

    def is_light_col(col):
        return np.mean(np.all(col >= thresh, axis=1)) >= coverage

    # Top
    while top < bottom-1 and is_light_row(img[top:top+1, :, :]):
        top += 1
    # Bottom
    while bottom-1 > top and is_light_row(img[bottom-1:bottom, :, :]):
        bottom -= 1
    # Left
    while left < right-1 and is_light_col(img[:, left:left+1, :]):
        left += 1
    # Right
    while right-1 > left and is_light_col(img[:, right-1:right, :]):
        right -= 1

    return img[top:bottom, left:right]



def remove_background_border(frame, bbox, background_thresh=250):
    x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
    # Defend against out-of-bounds and minimum size
    while h > 1 and np.all(frame[y, x:x+w] > background_thresh):     # Top edge
        y += 1
        h -= 1
    while h > 1 and np.all(frame[y+h-1, x:x+w] > background_thresh): # Bottom edge
        h -= 1
    while w > 1 and np.all(frame[y:y+h, x] > background_thresh):     # Left edge
        x += 1
        w -= 1
    while w > 1 and np.all(frame[y:y+h, x+w-1] > background_thresh): # Right edge
        w -= 1
    return {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}


def detect_and_crop_video(input_path, output_path, confidence_threshold=60.0):
    """
    Detects and extracts inner video from video-in-video format.
    
    Args:
        input_path: Path to input video file
        output_path: Path where cropped video will be saved
        confidence_threshold: Minimum confidence to perform cropping (0-100)
    
    Returns:
        dict: Detection results with keys:
            - detected: bool
            - confidence: float
            - output: str (path to output file)
            - bbox: dict or None
            - original_dimensions: dict
            - cropped_dimensions: dict or None
    """
    
    # ========================
    # 1. LOAD VIDEO
    # ========================
    cap = cv2.VideoCapture(str(input_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    
    print(f"Video: {width}x{height}, {fps:.2f}fps, {duration:.2f}s, {total_frames} frames")
    
    # ========================
    # 2. SAMPLE FRAMES
    # ========================
    # Calculate sampling interval
    if duration < 5:
        sample_interval = 0.25
    elif duration < 30:
        sample_interval = 0.5
    else:
        sample_interval = 1.0
    
    sample_frame_interval = max(1, int(sample_interval * fps))
    
    # Exclude first and last second
    start_frame = int(fps)
    end_frame = total_frames - int(fps)
    
    sampled_frames = []
    sample_indices = []
    
    for frame_idx in range(start_frame, end_frame, sample_frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            sampled_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            sample_indices.append(frame_idx)
        
        # Limit to 100 samples max
        if len(sampled_frames) >= 100:
            break
    
    print(f"Sampled {len(sampled_frames)} frames for analysis")
    
    if len(sampled_frames) < 10:
        cap.release()
        print("Not enough frames sampled. Returning original video.")
        shutil.copy(input_path, output_path)
        return {
            'detected': False,
            'confidence': 0.0,
            'output': str(output_path),
            'bbox': None,
            'original_dimensions': {'width': width, 'height': height},
            'cropped_dimensions': None
        }
    
    # ========================
    # 3. TEMPORAL VARIANCE ANALYSIS
    # ========================
    print("Computing temporal variance...")
    
    # Convert to grayscale
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in sampled_frames]
    frame_stack = np.stack(gray_frames, axis=-1)
    
    # Compute variance across time
    variance_map = np.var(frame_stack, axis=-1)
    
    # Normalize
    variance_normalized = cv2.normalize(variance_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Smooth
    variance_smoothed = cv2.GaussianBlur(variance_normalized, (5, 5), 0)
    
    # ========================
    # 4. IDENTIFY DYNAMIC REGION
    # ========================
    print("Identifying dynamic region...")
    
    # Threshold at 30th percentile
    threshold = np.percentile(variance_smoothed.flatten(), 30)
    binary_mask = (variance_smoothed > threshold).astype(np.uint8)
    
    # Morphological operations
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    if num_labels <= 1:  # Only background
        cap.release()
        print("No dynamic region detected. Returning original video.")
        subprocess.run(['cp', str(input_path), str(output_path)], check=True)
        return {
            'detected': False,
            'confidence': 0.0,
            'output': str(output_path),
            'bbox': None,
            'original_dimensions': {'width': width, 'height': height},
            'cropped_dimensions': None
        }
    
    # Analyze regions (skip label 0 which is background)
    regions = []
    total_pixels = height * width
    
    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        
        aspect_ratio = w / h if h > 0 else 0
        coverage = area / total_pixels
        rectangularity = area / (w * h) if (w * h) > 0 else 0
        
        # Score the region
        score = 0.0
        
        # Size score (prefer 30-90% coverage)
        if 0.30 <= coverage <= 0.90:
            score += 30.0 * (1 - abs(coverage - 0.60) / 0.30)
        
        # Aspect ratio score (standard video ratios)
        standard_ratios = [16/9, 9/16, 4/3, 3/4, 1/1, 21/9, 9/21]
        min_ratio_diff = min([abs(aspect_ratio - r) for r in standard_ratios])
        if min_ratio_diff < 0.1:
            score += 30.0
        elif min_ratio_diff < 0.2:
            score += 15.0
        
        # Rectangularity score
        if rectangularity > 0.85:
            score += 20.0 * rectangularity
        
        # Centering score
        region_center_x = x + w / 2
        region_center_y = y + h / 2
        frame_center_x = width / 2
        frame_center_y = height / 2
        
        center_offset_x = abs(region_center_x - frame_center_x) / frame_center_x
        center_offset_y = abs(region_center_y - frame_center_y) / frame_center_y
        
        if center_offset_x < 0.1 and center_offset_y < 0.1:
            score += 20.0
        
        regions.append({
            'bbox': {'x': x, 'y': y, 'w': w, 'h': h},
            'area': area,
            'aspect_ratio': aspect_ratio,
            'coverage': coverage,
            'rectangularity': rectangularity,
            'score': score
        })
    
    if not regions:
        cap.release()
        print("No valid regions found. Returning original video.")
        subprocess.run(['cp', str(input_path), str(output_path)], check=True)
        return {
            'detected': False,
            'confidence': 0.0,
            'output': str(output_path),
            'bbox': None,
            'original_dimensions': {'width': width, 'height': height},
            'cropped_dimensions': None
        }
    
    best_region = max(regions, key=lambda r: r['score'])
    
    if best_region['score'] < 40.0:
        cap.release()
        print(f"Best region score too low ({best_region['score']:.1f}). Returning original video.")
        subprocess.run(['cp', str(input_path), str(output_path)], check=True)
        return {
            'detected': False,
            'confidence': best_region['score'],
            'output': str(output_path),
            'bbox': best_region['bbox'],
            'original_dimensions': {'width': width, 'height': height},
            'cropped_dimensions': None
        }
    
    initial_confidence = min(best_region['score'], 100.0)
    bbox = best_region['bbox']
    
    print(f"Initial detection - Confidence: {initial_confidence:.1f}%")
    print(f"Bbox: x={bbox['x']}, y={bbox['y']}, w={bbox['w']}, h={bbox['h']}")
    
    # ========================
    # 5. REFINE BOUNDARY
    # ========================
    print("Refining boundaries...")
    
    mid_frame = sampled_frames[len(sampled_frames) // 2]
    gray_mid = cv2.cvtColor(mid_frame, cv2.COLOR_RGB2GRAY)
    
    # Edge detection on multiple frames
    edge_maps = []
    for i in range(0, len(sampled_frames), 5):
        gray = cv2.cvtColor(sampled_frames[i], cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_maps.append(edges)
    
    edge_persistence = np.mean(edge_maps, axis=0)
    edge_persistent_binary = (edge_persistence > 0.5).astype(np.uint8)
    
    # Refine boundaries by looking for strong edges
    margin = 20
    
    def find_edge_in_region(region, axis):
        density = np.sum(region, axis=axis)
        if len(density) == 0 or np.max(density) == 0:
            return None
        threshold = np.max(density) * 0.4
        peaks = np.where(density > threshold)[0]
        if len(peaks) > 0:
            return peaks[len(peaks) // 2]  # Take median peak
        return None
    
    # Refine top
    y_min = bbox['y']
    search_top = edge_persistent_binary[max(0, y_min - margin):y_min + margin, bbox['x']:bbox['x'] + bbox['w']]
    top_edge = find_edge_in_region(search_top, axis=1)
    if top_edge is not None:
        y_min = max(0, y_min - margin) + top_edge
    
    # Refine bottom
    y_max = bbox['y'] + bbox['h']
    search_bottom = edge_persistent_binary[y_max - margin:min(height, y_max + margin), bbox['x']:bbox['x'] + bbox['w']]
    bottom_edge = find_edge_in_region(search_bottom, axis=1)
    if bottom_edge is not None:
        y_max = (y_max - margin) + bottom_edge
    
    # Refine left
    x_min = bbox['x']
    search_left = edge_persistent_binary[y_min:y_max, max(0, x_min - margin):x_min + margin]
    left_edge = find_edge_in_region(search_left, axis=0)
    if left_edge is not None:
        x_min = max(0, x_min - margin) + left_edge
    
    # Refine right
    x_max = bbox['x'] + bbox['w']
    search_right = edge_persistent_binary[y_min:y_max, x_max - margin:min(width, x_max + margin)]
    right_edge = find_edge_in_region(search_right, axis=0)
    if right_edge is not None:
        x_max = (x_max - margin) + right_edge
    
    # Ensure bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)
    
    refined_bbox = {
        'x': int(x_min),
        'y': int(y_min),
        'w': int(x_max - x_min),
        'h': int(y_max - y_min)
    }
    
    print(f"Refined bbox: x={refined_bbox['x']}, y={refined_bbox['y']}, w={refined_bbox['w']}, h={refined_bbox['h']}")
    # === Fix to remove stray static background edge ===
    refined_bbox = remove_background_border(mid_frame, refined_bbox, background_thresh=250)
    print(f"Final bbox after background strip removal: x={refined_bbox['x']}, y={refined_bbox['y']}, w={refined_bbox['w']}, h={refined_bbox['h']}")

    
    # ========================
    # 6. TEMPORAL CONSISTENCY CHECK
    # ========================
    print("Validating temporal consistency...")
    
    first_frame = sampled_frames[0]
    last_frame = sampled_frames[-1]
    mid_frame = sampled_frames[len(sampled_frames) // 2]
    
    ssim_scores = []
    
    # Check top region
    if refined_bbox['y'] > 10:
        first_top = first_frame[0:refined_bbox['y'], :]
        last_top = last_frame[0:refined_bbox['y'], :]
        mid_top = mid_frame[0:refined_bbox['y'], :]
        
        if first_top.size > 0 and last_top.size > 0:
            score1 = ssim(cv2.cvtColor(first_top, cv2.COLOR_RGB2GRAY), 
                         cv2.cvtColor(last_top, cv2.COLOR_RGB2GRAY))
            score2 = ssim(cv2.cvtColor(first_top, cv2.COLOR_RGB2GRAY), 
                         cv2.cvtColor(mid_top, cv2.COLOR_RGB2GRAY))
            ssim_scores.extend([score1, score2])
    
    # Check bottom region
    if refined_bbox['y'] + refined_bbox['h'] < height - 10:
        first_bottom = first_frame[refined_bbox['y'] + refined_bbox['h']:, :]
        last_bottom = last_frame[refined_bbox['y'] + refined_bbox['h']:, :]
        mid_bottom = mid_frame[refined_bbox['y'] + refined_bbox['h']:, :]
        
        if first_bottom.size > 0 and last_bottom.size > 0:
            score1 = ssim(cv2.cvtColor(first_bottom, cv2.COLOR_RGB2GRAY), 
                         cv2.cvtColor(last_bottom, cv2.COLOR_RGB2GRAY))
            score2 = ssim(cv2.cvtColor(first_bottom, cv2.COLOR_RGB2GRAY), 
                         cv2.cvtColor(mid_bottom, cv2.COLOR_RGB2GRAY))
            ssim_scores.extend([score1, score2])
    
    # Check left region
    if refined_bbox['x'] > 10:
        first_left = first_frame[refined_bbox['y']:refined_bbox['y'] + refined_bbox['h'], 0:refined_bbox['x']]
        last_left = last_frame[refined_bbox['y']:refined_bbox['y'] + refined_bbox['h'], 0:refined_bbox['x']]
        mid_left = mid_frame[refined_bbox['y']:refined_bbox['y'] + refined_bbox['h'], 0:refined_bbox['x']]
        
        if first_left.size > 0 and last_left.size > 0:
            score1 = ssim(cv2.cvtColor(first_left, cv2.COLOR_RGB2GRAY), 
                         cv2.cvtColor(last_left, cv2.COLOR_RGB2GRAY))
            score2 = ssim(cv2.cvtColor(first_left, cv2.COLOR_RGB2GRAY), 
                         cv2.cvtColor(mid_left, cv2.COLOR_RGB2GRAY))
            ssim_scores.extend([score1, score2])
    
    # Check right region
    if refined_bbox['x'] + refined_bbox['w'] < width - 10:
        first_right = first_frame[refined_bbox['y']:refined_bbox['y'] + refined_bbox['h'], 
                                  refined_bbox['x'] + refined_bbox['w']:]
        last_right = last_frame[refined_bbox['y']:refined_bbox['y'] + refined_bbox['h'], 
                               refined_bbox['x'] + refined_bbox['w']:]
        mid_right = mid_frame[refined_bbox['y']:refined_bbox['y'] + refined_bbox['h'], 
                             refined_bbox['x'] + refined_bbox['w']:]
        
        if first_right.size > 0 and last_right.size > 0:
            score1 = ssim(cv2.cvtColor(first_right, cv2.COLOR_RGB2GRAY), 
                         cv2.cvtColor(last_right, cv2.COLOR_RGB2GRAY))
            score2 = ssim(cv2.cvtColor(first_right, cv2.COLOR_RGB2GRAY), 
                         cv2.cvtColor(mid_right, cv2.COLOR_RGB2GRAY))
            ssim_scores.extend([score1, score2])
    
    # Calculate consistency boost
    if ssim_scores:
        avg_ssim = np.mean(ssim_scores)
        if avg_ssim > 0.98:
            consistency_boost = 30.0
        elif avg_ssim > 0.95:
            consistency_boost = 20.0
        elif avg_ssim > 0.90:
            consistency_boost = 10.0
        else:
            consistency_boost = 0.0
        print(f"Static region SSIM: {avg_ssim:.4f} -> Boost: {consistency_boost:.1f}%")
    else:
        consistency_boost = 0.0
        print("No static regions to validate")
    
    final_confidence = min(initial_confidence + consistency_boost, 100.0)
    
    print(f"\nFinal confidence: {final_confidence:.1f}%")
    print(f"Cropped dimensions: {refined_bbox['w']}x{refined_bbox['h']}")
    print(f"Aspect ratio: {refined_bbox['w']/refined_bbox['h']:.2f}:1")
    
    # ========================
    # 7. DECISION AND EXTRACTION
    # ========================
    if final_confidence < confidence_threshold:
        cap.release()
        print(f"Confidence ({final_confidence:.1f}%) below threshold ({confidence_threshold}%). Returning original video.")
        shutil.copy(input_path, output_path)
        return {
            'detected': False,
            'confidence': final_confidence,
            'output': str(output_path),
            'bbox': refined_bbox,
            'original_dimensions': {'width': width, 'height': height},
            'cropped_dimensions': None
        }
    
    # ========================
    # 8. EXTRACT VIDEO
    # ========================
    print("Extracting inner video...")
    
    # Reset video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Get codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create temporary output without audio
    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_output_path = temp_output.name
    temp_output.close()
    
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, 
                         (refined_bbox['w'], refined_bbox['h']))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop frame
        cropped = frame[refined_bbox['y']:refined_bbox['y'] + refined_bbox['h'],
                       refined_bbox['x']:refined_bbox['x'] + refined_bbox['w']]
        
        # --- After cropping the frame ---
        cropped = frame[refined_bbox['y']:refined_bbox['y'] + refined_bbox['h'],
                        refined_bbox['x']:refined_bbox['x'] + refined_bbox['w']]

        # Create grayscale and threshold to find inner contour (assume nearly white background)
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)

        # Find largest contour
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            best_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [best_contour], -1, 255, thickness=-1)
            # Mask the cropped image
            cropped = cv2.bitwise_and(cropped, cropped, mask=mask)
        # Now write as usual
        cropped = strip_light_edges(cropped, thresh=252, coverage=0.90)
        out.write(cropped)
      
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    cap.release()
    out.release()
    
    print(f"Video extraction complete: {frame_count} frames written")
    
    # Final pass through ffmpeg to produce a streaming-friendly MP4
    print("Finalizing cropped video...")
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-i', temp_output_path,
        '-i', str(input_path),
        '-map', '0:v:0',
        '-map', '1:a?',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-profile:v', 'high',
        '-level', '4.1',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-shortest',
        '-map_metadata', '1',
        str(output_path)
    ]
    temp_path = Path(temp_output_path)
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        print("FFmpeg re-encode failed. Leaving temporary file for inspection:", temp_output_path)
        raise exc
    else:
        if temp_path.exists():
            temp_path.unlink()
    
    print(f"Successfully extracted video to: {output_path}")
    
    return {
        'detected': True,
        'confidence': final_confidence,
        'output': str(output_path),
        'bbox': refined_bbox,
        'original_dimensions': {'width': width, 'height': height},
        'cropped_dimensions': {'width': refined_bbox['w'], 'height': refined_bbox['h']}
    }


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python video_detector.py <input_video> <output_video> [confidence_threshold]")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_video = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
    
    result = detect_and_crop_video(input_video, output_video, threshold)
    
    print("\n" + "="*50)
    print("RESULTS:")
    print("="*50)
    print(f"Detected: {result['detected']}")
    print(f"Confidence: {result['confidence']:.1f}%")
    print(f"Output: {result['output']}")
    if result['bbox']:
        print(f"Bounding box: {result['bbox']}")
    print(f"Original: {result['original_dimensions']}")
    if result['cropped_dimensions']:
        print(f"Cropped: {result['cropped_dimensions']}")
