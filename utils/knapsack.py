def knapsack_ortools(weights, values, capacity):
    """
    Solves 0/1 knapsack using dynamic programming.
    This selects segments to maximize value within the max length budget.
    
    Args:
        weights: list of weights (segment lengths)
        values: list of values (segment importance scores)
        capacity: max budget (e.g. 15% of video length in frames)
        
    Returns:
        selected_indices: List of selected item indices
    """
    n = len(values)
    W = int(capacity)
    
    # dp[i][w] array
    K = [[0 for _ in range(W + 1)] for _ in range(n + 1)]
    
    # Ensure weights are integers
    wt = [int(w) for w in weights]
    val = values
    
    # Build table K[][]
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]],  K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]
                
    # Find selected items
    res = K[n][W]
    w = W
    selected = []
    
    for i in range(n, 0, -1):
        if res <= 0:
            break
        if res == K[i - 1][w]:
            continue
        else:
            selected.append(i - 1)
            res = res - val[i - 1]
            w = w - wt[i - 1]
            
    return selected[::-1]
