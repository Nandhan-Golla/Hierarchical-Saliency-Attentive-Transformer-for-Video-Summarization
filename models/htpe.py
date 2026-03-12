import torch
import torch.nn as nn
from .dpsa import DPSABlock

def pytorch_kmeans(x, n_clusters, max_iter=100):
    """
    Fast simple k-means clustering in PyTorch.
    x: (N, D)
    Returns:
        centroids: (C, D)
        labels: (N,)
    """
    N, D = x.shape
    if N <= n_clusters:
        return x.clone(), torch.arange(N, device=x.device)
    
    # Randomly initialize centroids
    indices = torch.randperm(N, device=x.device)[:n_clusters]
    centroids = x[indices].clone()
    
    for _ in range(max_iter):
        # Compute distances (N, C)
        distances = torch.cdist(x, centroids, p=2)
        # Assign to nearest centroid
        labels = torch.argmin(distances, dim=1)
        
        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        counts = torch.bincount(labels, minlength=n_clusters)
        
        # Prevent division by zero
        counts = torch.clamp(counts, min=1).view(-1, 1).float()
        
        for c in range(n_clusters):
            mask = (labels == c).float().unsqueeze(1) # (N, 1)
            new_centroids[c] = (x * mask).sum(dim=0) / counts[c]
            
        # Check for convergence
        if torch.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids
        
    return centroids, labels

class LocalDPSA(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_sal=512, window_size=8, n_layers=2, gamma_init=0.5):
        super().__init__()
        self.window_size = window_size
        self.layers = nn.ModuleList([
            DPSABlock(d_model=d_model, n_heads=n_heads, d_sal=d_sal, gamma_init=gamma_init) for _ in range(n_layers)
        ])
        
    def forward(self, x_sem, x_sal):
        """
        x_sem: (B, N, d_model)
        x_sal: (B, N, d_sal)
        Uses overlapping windows or chunked windows. We will pad to multiple of window_size,
        reshape to treat windows as batch dimension, process, and reshape back.
        """
        B, N, D_sem = x_sem.shape
        _, _, D_sal = x_sal.shape
        
        pad_len = (self.window_size - (N % self.window_size)) % self.window_size
        if pad_len > 0:
            x_sem_padded = torch.cat([x_sem, x_sem[:, -1:].repeat(1, pad_len, 1)], dim=1)
            x_sal_padded = torch.cat([x_sal, x_sal[:, -1:].repeat(1, pad_len, 1)], dim=1)
        else:
            x_sem_padded, x_sal_padded = x_sem, x_sal
            
        N_padded = x_sem_padded.size(1)
        num_windows = N_padded // self.window_size
        
        # Reshape: (B*num_windows, W, D)
        x_sem_w = x_sem_padded.view(B * num_windows, self.window_size, D_sem)
        x_sal_w = x_sal_padded.view(B * num_windows, self.window_size, D_sal)
        
        out_sem = x_sem_w
        for layer in self.layers:
            out_sem = layer(out_sem, x_sal_w)
            
        out_sem = out_sem.view(B, N_padded, D_sem)
        
        # Remove padding
        if pad_len > 0:
            out_sem = out_sem[:, :N, :]
            
        return out_sem

class HTPE(nn.Module):
    """
    Hierarchical Temporal Pyramid Encoder.
    Models at Frame (local), Shot (segment), and Scene (global) levels.
    """
    def __init__(self, d_model=512, n_heads=8, d_sal=512, local_window=8, 
                 n_local=2, n_shot=2, n_scene=1, gamma_init=0.5):
        super(HTPE, self).__init__()
        
        self.d_model = d_model
        
        # Level 1: Frame-Level Local DPSA
        self.local_encoder = LocalDPSA(d_model, n_heads, d_sal, local_window, n_local, gamma_init)
        
        # Level 2: Shot-Level Segment DPSA
        self.shot_encoder = nn.ModuleList([
            DPSABlock(d_model, n_heads, d_sal, gamma_init=gamma_init) for _ in range(n_shot)
        ])
        
        # Level 3: Scene-Level Global DPSA
        self.scene_encoder = nn.ModuleList([
            DPSABlock(d_model, n_heads, d_sal, gamma_init=gamma_init) for _ in range(n_scene)
        ])
        
        # Multi-scale Fusion
        self.fusion_linear = nn.Linear(d_model * 3, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        
    def forward(self, F_fused, S_features, shot_boundaries):
        """
        F_fused: (Batch=1, N, d_model) - semantic
        S_features: (Batch=1, N, d_sal) - saliency
        shot_boundaries: List of ints indicating start index of shots. e.g. [0, 45, 90, 150]
        We assume Batch=1 for video processing.
        """
        B, N, D = F_fused.shape
        assert B == 1, "HTPE expects batch size of 1 for variable length video sequences."
        
        # -- LEVEL 1: Local --
        h_local = self.local_encoder(F_fused, S_features) # (1, N, d_model)
        
        # -- LEVEL 2: Shot-Level --
        # Pool frames into shots
        shot_tokens_list = []
        sal_tokens_list = []
        
        num_shots = len(shot_boundaries)
        # Handle case where shot_boundaries does not end with N
        boundaries = list(shot_boundaries)
        if boundaries[-1] != N:
            boundaries.append(N)
        
        frame_to_shot_idx = torch.zeros(N, dtype=torch.long, device=F_fused.device)
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i+1]
            if start == end: continue # empty shot fallback
            shot_rep = h_local[0, start:end, :].mean(dim=0)
            sal_rep = S_features[0, start:end, :].mean(dim=0)
            shot_tokens_list.append(shot_rep)
            sal_tokens_list.append(sal_rep)
            frame_to_shot_idx[start:end] = len(shot_tokens_list) - 1
            
        if len(shot_tokens_list) == 0:
            # Fallback if no valid shots
            shot_tokens_list = [h_local[0].mean(dim=0)]
            sal_tokens_list = [S_features[0].mean(dim=0)]
            frame_to_shot_idx[:] = 0

        shot_tokens = torch.stack(shot_tokens_list).unsqueeze(0) # (1, S, d_model)
        shot_sal = torch.stack(sal_tokens_list).unsqueeze(0)     # (1, S, d_sal)
        
        # Encoder DPSA on shots
        h_shot_encoded = shot_tokens
        for layer in self.shot_encoder:
            h_shot_encoded = layer(h_shot_encoded, shot_sal)
            
        # Broadcast back to frames
        # frame_to_shot_idx maps each frame to its shot index
        h_shot_broadcast = h_shot_encoded[0, frame_to_shot_idx, :].unsqueeze(0) # (1, N, d_model)
        
        # -- LEVEL 3: Scene-Level --
        # K-means clustering on shot tokens
        num_valid_shots = shot_tokens.size(1)
        C = min(max(1, N // 50), 10) # As per specification
        C = min(C, num_valid_shots)  # Cannot have more clusters than shots
        
        scene_tokens, shot_to_scene_labels = pytorch_kmeans(
            shot_tokens[0], n_clusters=C, max_iter=30
        )
        # Re-compute saliency centroids
        scene_sal = torch.zeros(C, S_features.size(-1), device=S_features.device)
        for c in range(C):
            mask = (shot_to_scene_labels == c)
            if mask.sum() > 0:
                scene_sal[c] = shot_sal[0][mask].mean(dim=0)
            else:
                scene_sal[c] = shot_sal[0].mean(dim=0) # Fallback
                
        scene_tokens = scene_tokens.unsqueeze(0) # (1, C, d_model)
        scene_sal = scene_sal.unsqueeze(0)       # (1, C, d_sal)
        
        # Encoder DPSA on scenes
        h_scene_encoded = scene_tokens
        for layer in self.scene_encoder:
            h_scene_encoded = layer(h_scene_encoded, scene_sal)
            
        # Broadcast back to frames (frame -> shot -> scene)
        frame_to_scene_idx = shot_to_scene_labels[frame_to_shot_idx]
        h_scene_broadcast = h_scene_encoded[0, frame_to_scene_idx, :].unsqueeze(0) # (1, N, d_model)
        
        # -- MULTI-SCALE FUSION --
        h_concat = torch.cat([h_local, h_shot_broadcast, h_scene_broadcast], dim=-1) # (1, N, 1536)
        h_temporal = self.fusion_linear(h_concat)
        h_temporal = self.fusion_norm(h_temporal)
        
        return h_temporal
