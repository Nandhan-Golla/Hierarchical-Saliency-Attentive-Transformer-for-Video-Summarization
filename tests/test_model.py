import torch
import pytest
import os
import sys

# Setup imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dpsa import DPSA, DPSABlock
from models.ssfb import SSFB
from models.htpe import HTPE
from models.hisat import HiSAT

def test_ssfb():
    sem_dim, sal_dim, d_model = 1024, 256, 512
    ssfb = SSFB(sem_dim, sal_dim, d_model, n_heads=8)
    
    B, N = 2, 50
    F_sem = torch.rand(B, N, sem_dim)
    F_sal = torch.rand(B, N, sal_dim)
    
    out = ssfb(F_sem, F_sal)
    assert out.shape == (B, N, d_model), f"SSFB output shape {out.shape} does not match expected ({B}, {N}, {d_model})"

def test_dpsa():
    d_model, d_sal = 512, 128
    dpsa = DPSABlock(d_model=d_model, n_heads=8, d_sal=d_sal)
    
    B, N = 2, 50
    X_sem = torch.rand(B, N, d_model)
    X_sal = torch.rand(B, N, d_sal)
    
    out = dpsa(X_sem, X_sal)
    assert out.shape == (B, N, d_model)

def test_htpe():
    B, N, d_model, d_sal = 1, 100, 512, 256
    htpe = HTPE(d_model=d_model, d_sal=d_sal)
    
    F_fused = torch.rand(B, N, d_model)
    F_sal = torch.rand(B, N, d_sal)
    shot_boundaries = [0, 20, 45, 80, 100]
    
    out = htpe(F_fused, F_sal, shot_boundaries)
    assert out.shape == (B, N, d_model), f"HTPE output shape {out.shape} doesn't match ({B}, {N}, {d_model})"

def test_hisat_gradient():
    B, N = 1, 80
    sem_dim, sal_dim = 1024, 256
    model = HiSAT(sem_dim=sem_dim, sal_dim=sal_dim)
    
    F_sem = torch.rand(B, N, sem_dim)
    F_sal = torch.rand(B, N, sal_dim)
    S_scores = torch.rand(B, N)
    shot_boundaries = [0, 40, 80]
    
    # Forward pass
    importance_scores, budget_ratio, h_temporal = model(F_sem, F_sal, S_scores, shot_boundaries)
    
    # Check shapes
    assert importance_scores.shape == (B, N)
    assert budget_ratio.shape == (B, 1)
    
    # Backprop check
    loss = importance_scores.sum() + budget_ratio.sum()
    loss.backward()
    
    # Verify gradients are flowing through the predictor and SSFB
    assert model.predictor.mlp[0].weight.grad is not None, "Predictor gradients not found"
    assert model.ssfb.gate_fc.weight.grad is not None, "SSFB gradients not found"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
