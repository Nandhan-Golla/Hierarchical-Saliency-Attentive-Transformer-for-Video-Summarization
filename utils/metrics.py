import numpy as np

def compute_f_score(pred_scores, target_scores, shot_boundaries, max_budget_ratio=0.15):
    """
    Computes Precision, Recall, and F-score for video summarization.
    This is a simplified rank-based proxy.
    
    Args:
        pred_scores: (N,) frame scores [0, 1]
        target_scores: (N,) ground truth frame scores
        shot_boundaries: list of shot start indices
        max_budget_ratio: proportion of frames to select
        
    Returns:
        f_score: float
    """
    N = len(pred_scores)
    if N == 0:
        return 0.0
        
    budget = int(N * max_budget_ratio)
    
    # Simple top-K selection for frame-level evaluation
    # More advanced methods use KTS and Knapsack to select whole shots
    
    # Sort indices
    pred_indices = np.argsort(pred_scores)[::-1][:budget]
    target_indices = np.argsort(target_scores)[::-1][:budget]
    
    # Intersection
    intersection = len(set(pred_indices).intersection(set(target_indices)))
    
    # Precision and Recall
    precision = intersection / budget if budget > 0 else 0
    recall = intersection / budget if budget > 0 else 0
    
    if precision + recall > 0:
        f_score = 2 * precision * recall / (precision + recall)
    else:
        f_score = 0.0
        
    return f_score
