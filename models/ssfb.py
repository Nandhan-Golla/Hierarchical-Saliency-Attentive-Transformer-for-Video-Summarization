import torch
import torch.nn as nn
import torch.nn.functional as F

class SSFB(nn.Module):
    """
    Saliency-Semantic Fusion Bridge (SSFB).
    Aligns saliency and semantic features through bidirectional cross-attention
    and merges them via a learned gating mechanism.
    """
    def __init__(self, sem_dim=1024, sal_dim=256, d_model=512, n_heads=8):
        super(SSFB, self).__init__()
        
        self.d_model = d_model
        
        # 1. Projections to common dimension
        self.proj_sem = nn.Linear(sem_dim, d_model)
        self.proj_sal = nn.Linear(sal_dim, d_model)
        
        # 2. Bidirectional Cross-Attention
        # PyTorch MultiheadAttention takes (Query, Key, Value)
        self.attn_sem2sal = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.attn_sal2sem = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        
        # 3. Gated Fusion
        self.gate_fc = nn.Linear(d_model * 2, d_model)
        
    def forward(self, F_sem, F_sal):
        """
        Args:
            F_sem: Semantic features (batch_size, seq_len, sem_dim)
            F_sal: Saliency features (batch_size, seq_len, sal_dim)
        Returns:
            F_fused: Fused features (batch_size, seq_len, d_model)
        """
        # Step 1: Projection
        F_sem_proj = self.proj_sem(F_sem)  # (B, N, 512)
        F_sal_proj = self.proj_sal(F_sal)  # (B, N, 512)
        
        # Step 2: Bidirectional Cross-attention
        # Sem->Sal: Query=Sem, Key=Sal, Value=Sal
        # Sal->Sem: Query=Sal, Key=Sem, Value=Sem
        F_sem2sal, _ = self.attn_sem2sal(query=F_sem_proj, key=F_sal_proj, value=F_sal_proj)
        F_sal2sem, _ = self.attn_sal2sem(query=F_sal_proj, key=F_sem_proj, value=F_sem_proj)
        
        # Step 3: Gated Fusion
        concat_features = torch.cat([F_sem2sal, F_sal2sem], dim=-1) # (B, N, 1024)
        gate = torch.sigmoid(self.gate_fc(concat_features))         # (B, N, 512)
        
        F_fused = gate * F_sem2sal + (1 - gate) * F_sal2sem         # (B, N, 512)
        
        return F_fused
