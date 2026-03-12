import numpy as np

def kts_segmentation(features, max_segments=None):
    """
    Proxy for Kernel Temporal Segmentation (KTS).
    Creates segments based on feature distance differences.
    Returns:
        change_points: list of [start_frame, end_frame]
    """
    N = len(features)
    if N == 0:
        return []
    
    # Calculate pairwise distances (or consecutive distances)
    # Simple approach: consecutive distance peaks
    # features: (N, D)
    diffs = np.linalg.norm(features[1:] - features[:-1], axis=1)
    
    # Smooth diffs slightly
    window = 5
    smoothed = np.convolve(diffs, np.ones(window)/window, mode='same')
    
    # Find local maxima as change points
    cps = [0]
    threshold = smoothed.mean() + smoothed.std()
    
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
            if smoothed[i] > threshold:
                cps.append(i + 1)
                
    cps.append(N)
    
    # Ensure segments are within reasonable sizes
    final_cps = []
    min_length = 15 # frames
    
    last_cp = 0
    for cp in cps[1:]:
        if cp - last_cp >= min_length or cp == N:
            final_cps.append([last_cp, cp])
            last_cp = cp
            
    # If not enough changes, fake some segments
    if len(final_cps) == 0:
        step = max(N // 10, 1)
        final_cps = [[i, min(i+step, N)] for i in range(0, N, step)]
        
    return final_cps
