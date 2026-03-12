import torch
import torch.nn as nn
import math

from .ssfb import SSFB
from .htpe import HTPE
from .predictor import ImportancePredictor
from .saliency_encoder import SaliencyFeatureEncoder

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """
    def __init__(self, d_model=512, max_len=10000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class HiSAT(nn.Module):
    """
    Hierarchical Saliency-Attentive Transformer (HiSAT) for Video Summarization.
    """
    def __init__(self, sem_dim=1024, sal_dim=256, d_model=512, n_heads=8, 
                 local_window=8, n_local=2, n_shot=2, n_scene=1, dropout=0.3, gamma_init=0.5):
        super(HiSAT, self).__init__()
        
        # 0. Saliency Encoder (Optional, if inputs are spatial maps)
        self.sal_encoder = SaliencyFeatureEncoder(sal_dim)
        
        # 1. Saliency-Semantic Fusion Bridge
        self.ssfb = SSFB(sem_dim, sal_dim, d_model, n_heads)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Hierarchical Temporal Pyramid Encoder
        self.htpe = HTPE(d_model, n_heads, sal_dim, local_window, n_local, n_shot, n_scene, gamma_init)
        
        # 4. Importance Predictor & Budget Estimator
        self.predictor = ImportancePredictor(d_model, dropout)
        
    def forward(self, F_sem, F_sal, S_scores, shot_boundaries):
        """
        Args:
            F_sem: Semantic features (Batch=1, N, sem_dim)
            F_sal: Saliency features (Batch=1, N, sal_dim)
            S_scores: Scalar saliency scores (Batch=1, N)
            shot_boundaries: List of ints representing start frame of each shot.
        
        Returns:
            importance_scores: (Batch=1, N) predicted importance in [0, 1]
            budget_ratio: (Batch=1, 1) predicted optimal summary budget
            h_temporal: (Batch=1, N, d_model) internal representations useful for contrastive loss
        """
        # Step 0: Encode saliency maps if F_sal is spatial maps
        if F_sal.dim() == 5:
            F_sal = self.sal_encoder(F_sal)
            
        # Step 1: Feature Fusion
        F_fused = self.ssfb(F_sem, F_sal)  # (1, N, d_model)
        
        # Step 2: Positional Encoding
        F_fused = self.pos_encoder(F_fused)
        
        # Step 3: Hierarchical Temporal Pyramid
        # Pass raw F_sal as it has dimension sal_dim which DPSA projects nicely
        h_temporal = self.htpe(F_fused, F_sal, shot_boundaries)  # (1, N, d_model)
        
        # Step 4: Importance Score & Budget Prediction
        importance_scores, budget_ratio = self.predictor(h_temporal, S_scores)
        
        return importance_scores, budget_ratio, h_temporal
