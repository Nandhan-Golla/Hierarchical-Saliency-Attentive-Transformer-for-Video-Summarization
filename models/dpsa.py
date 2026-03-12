import torch
import torch.nn as nn
import torch.nn.functional as F

class DPSA(nn.Module):
    """
    Dual-Path Saliency-Attentive (DPSA) Attention.
    Computes parallel attention maps for semantic and saliency features,
    fusing them before softmax.
    """
    def __init__(self, d_model=512, n_heads=8, d_sal=512, gamma_init=0.5):
        super(DPSA, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Semantic projections
        self.W_q_sem = nn.Linear(d_model, d_model)
        self.W_k_sem = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Saliency projections
        self.W_q_sal = nn.Linear(d_sal, d_model)
        self.W_k_sal = nn.Linear(d_sal, d_model)
        
        # Learnable fusion parameter
        self.gamma = nn.Parameter(torch.tensor(gamma_init))
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, X_sem, X_sal, mask=None):
        """
        Args:
            X_sem: Semantic features of shape (batch_size, seq_len, d_model)
            X_sal: Saliency features of shape (batch_size, seq_len, d_sal)
            mask: Optional attention mask of shape (batch_size, 1, seq_len, seq_len)
        """
        batch_size, seq_len, _ = X_sem.size()
        
        # 1. Project to Q, K, V for semantic
        Q_sem = self.W_q_sem(X_sem).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K_sem = self.W_k_sem(X_sem).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(X_sem).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. Project to Q, K for saliency
        Q_sal = self.W_q_sal(X_sal).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K_sal = self.W_k_sal(X_sal).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. Compute pre-softmax attention scores
        A_sem = torch.matmul(Q_sem, K_sem.transpose(-2, -1)) / (self.d_k ** 0.5)
        A_sal = torch.matmul(Q_sal, K_sal.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # 4. Fuse attention scores
        A_combined = A_sem + self.gamma * A_sal
        
        # Apply mask if provided
        if mask is not None:
            A_combined = A_combined.masked_fill(mask == 0, -1e9)
            
        # 5. Softmax
        A_final = F.softmax(A_combined, dim=-1)
        
        # 6. Apply attention to V
        out = torch.matmul(A_final, V)
        
        # 7. Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.W_o(out)
        
        return out

class DPSABlock(nn.Module):
    """
    A full Transformer encoder block incorporating DPSA.
    """
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.3, d_sal=512, gamma_init=0.5):
        super(DPSABlock, self).__init__()
        self.dpsa = DPSA(d_model, n_heads, d_sal, gamma_init)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, X_sem, X_sal, mask=None):
        out_dpsa = self.dpsa(X_sem, X_sal, mask)
        X_out = self.norm1(X_sem + self.dropout1(out_dpsa))
        out_ffn = self.ffn(X_out)
        X_out = self.norm2(X_out + self.dropout2(out_ffn))
        return X_out
