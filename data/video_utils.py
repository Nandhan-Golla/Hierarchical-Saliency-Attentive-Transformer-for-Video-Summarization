import cv2
import numpy as np
import torch
from torchvision import transforms

def extract_frames(video_path, fps_target=2, target_size=(224, 224)):
    """
    Extract frames from a video at a specified frame rate.
    Applies standard ImageNet normalization.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    if fps_original <= 0:
        fps_original = 30.0 # fallback
        
    frame_interval = int(round(fps_original / fps_target))
    if frame_interval < 1:
        frame_interval = 1
        
    frames_rgb = []
    original_frames = [] # For visualization later
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frames.append(cv2.resize(frame_rgb, target_size))
            
            # Apply transform
            tensor_frame = transform(frame_rgb)
            frames_rgb.append(tensor_frame)
            
        frame_idx += 1
        
    cap.release()
    
    if not frames_rgb:
        raise ValueError(f"No frames extracted from {video_path}")
        
    frames_tensor = torch.stack(frames_rgb) # (N, 3, 224, 224)
    return frames_tensor, original_frames

def compute_color_histogram(frame_rgb, bins=32):
    """Compute normalized 3D color histogram for a frame."""
    hist = cv2.calcHist([frame_rgb], [0, 1, 2], None, [bins, bins, bins], 
                        [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def compute_chi_square_distance(hist1, hist2):
    """Compute chi-square distance between two histograms."""
    # Add a small epsilon to avoid division by zero
    eps = 1e-10
    return np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + eps))

def detect_shot_boundaries(original_frames, threshold=0.5):
    """
    Detect shot boundaries using color histograms and chi-square distance.
    Returns list of frame indices where a new shot begins.
    Always includes 0 as the first boundary.
    """
    if not original_frames:
        return []
        
    boundaries = [0]
    
    # Compute histograms
    hists = [compute_color_histogram(f) for f in original_frames]
    
    # Compute distances between consecutive frames
    distances = []
    for i in range(len(hists) - 1):
        dist = compute_chi_square_distance(hists[i], hists[i+1])
        distances.append(dist)
        
    if not distances:
        return boundaries
        
    # Adaptive thresholding logic could also be added,
    # but for simplicity we rely on a fixed/relative threshold.
    # A standard way is to use mean + alpha * std
    dists_arr = np.array(distances)
    dynamic_threshold = dists_arr.mean() + 3.0 * dists_arr.std()
    
    # Use max of fixed and dynamic
    final_thresh = max(threshold, dynamic_threshold)
    
    for i, dist in enumerate(distances):
        if dist > final_thresh:
            boundaries.append(i + 1)
            
    return boundaries
