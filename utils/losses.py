import torch
import torch.nn as nn
import torch.nn.functional as F

class HisatLoss(nn.Module):
    def __init__(self, lambda_div=0.1, lambda_sp=0.01, alpha=0.5):
        super(HisatLoss, self).__init__()
        self.lambda_div = lambda_div
        self.lambda_sp = lambda_sp
        self.alpha = alpha
        
    def forward(self, pred, target, h_temporal, s_scores):
        """
        pred: (1, N) importance scores
        target: (1, N) ground truth scores
        h_temporal: (1, N, d_model) features
        s_scores: (1, N) saliency scores
        """
        L_imp = self.saliency_weighted_mse(pred, target, s_scores)
        L_div = self.contrastive_diversity_loss(h_temporal, pred)
        L_sp = self.sparsity_loss(pred)
        
        return L_imp + self.lambda_div * L_div + self.lambda_sp * L_sp, L_imp, L_div, L_sp
        
    def saliency_weighted_mse(self, pred, target, s_scores):
        """
        MSE where salient frames have higher weight in the loss.
        weight = 1 + alpha * s_scores
        """
        weight = 1.0 + self.alpha * s_scores
        loss = weight * (pred - target) ** 2
        return loss.mean()
        
    def contrastive_diversity_loss(self, h_temporal, pred):
        """
        Encourages diversity. We use a repelling loss on the chosen summary frames.
        Wait, for pure contrastive as per design:
        Push apart temporally adjacent similar frames.
        """
        # A simple diversity proxy: minimize cosine similarity among highly predicted frames
        # Weight by prediction scores so only "selected" frames repel each other
        B, N, D = h_temporal.shape
        h_norm = F.normalize(h_temporal[0], p=2, dim=1) # (N, D)
        sim_matrix = torch.matmul(h_norm, h_norm.t()) # (N, N)
        
        # We only care about pairs that are highly predicted
        p = pred[0].unsqueeze(1) # (N, 1)
        joint_pred = torch.matmul(p, p.t()) # (N, N)
        
        # Mask out diagonal since self-similarity is always 1
        mask = torch.eye(N, device=sim_matrix.device)
        
        # Maximize diversity = minimize similarity for high pred pairs
        div_loss = (sim_matrix * joint_pred * (1 - mask)).sum() / (N * (N - 1) + 1e-8)
        
        return div_loss
        
    def sparsity_loss(self, pred):
        """
        Encourages the model to only select a few frames.
        """
        return pred.mean()
