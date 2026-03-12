import torch
import torch.nn as nn

class ImportancePredictor(nn.Module):
    """
    Predicts frame-level importance scores and adaptive summary budget.
    """
    def __init__(self, d_model=512, dropout=0.3):
        super(ImportancePredictor, self).__init__()
        
        # Frame-level importance MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model + 1, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Adaptive Budget Predictor
        self.budget_predictor = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(self, h_temporal, s_scores):
        """
        Args:
            h_temporal: (B, N, d_model) feature sequence
            s_scores: (B, N) scalar saliency scores per frame
            
        Returns:
            importance_scores: (B, N) frame importance scores
            budget_ratio: (B, 1) predicted optimal budget ratio
        """
        B, N, _ = h_temporal.shape
        
        # 1. Importance Prediction
        # Saliency conditioning: concat global feature with scalar saliency score
        s_expanded = s_scores.unsqueeze(-1) # (B, N, 1)
        h_concat = torch.cat([h_temporal, s_expanded], dim=-1) # (B, N, d_model + 1)
        
        importance_scores = self.mlp(h_concat).squeeze(-1) # (B, N)
        
        # 2. Adaptive Budget Prediction
        # Global average pool
        h_global = h_temporal.mean(dim=1) # (B, d_model)
        # Scaled to be centered around 15% (could adjust scale/bias if needed)
        # We'll just output a raw sigmoid value and let the loss function or post-processing map it.
        # But generally, it should learn to output ~0.15. 
        budget_ratio = self.budget_predictor(h_global) # (B, 1)
        
        return importance_scores, budget_ratio
