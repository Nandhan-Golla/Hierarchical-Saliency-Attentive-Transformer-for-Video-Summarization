import cv2
import json
import numpy as np

def assemble_summary(video_path, importance_scores, kts_segments, selected_indices, out_path="summary_video.mp4", scores_path="scores.json"):
    """
    Reads the original video, selectively writes frames belonging to 
    selected KTS segments based on knapsack optimization, and saves scores.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video.")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    # Flatten selected segments into a set of frame indices
    selected_frames = set()
    for tr in selected_indices:
        # Avoid out of bounds
        if tr < len(kts_segments):
            start, end = kts_segments[tr]
            for i in range(start, end):
                selected_frames.add(i)
                
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx in selected_frames:
            out.write(frame)
            
        frame_idx += 1
        
    cap.release()
    out.release()
    
    # Save scores
    with open(scores_path, "w") as f:
        json.dump({
            "scores": [float(s) for s in importance_scores],
            "selected_segments": selected_indices,
            "kts_segments": kts_segments
        }, f)
        
    return out_path, scores_path
