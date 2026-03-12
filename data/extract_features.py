import torch
import torchvision.models as models
from torchvision.models import GoogLeNet_Weights
import numpy as np

class FeatureExtractor:
    """
    Extracts semantic and saliency features from video frames.
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # GoogLeNet (Semantic Setup)
        self.googlenet = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
        # We need the pooling output (R^1024), so replace the final FC layer with Identity
        self.googlenet.fc = torch.nn.Identity()
        self.googlenet.eval()
        self.googlenet.to(self.device)
        
    def extract_semantic(self, frames_tensor, batch_size=32):
        """
        Extract R^1024 semantic features from frames.
        frames_tensor: (N, 3, 224, 224) - expected to be normalized for ImageNet
        """
        N = frames_tensor.size(0)
        features = []
        
        with torch.no_grad():
            for i in range(0, N, batch_size):
                batch = frames_tensor[i:i+batch_size].to(self.device)
                out = self.googlenet(batch)
                features.append(out.cpu())
                
        return torch.cat(features, dim=0) # (N, 1024)
        
    def extract_saliency(self, frames_tensor):
        """
        IMPORTANT: This is a placeholder for the TranSalNet / DeepGaze II model.
        Setting up the full custom model for saliency requires external weights
        which are not included in PyTorch by default.
        
        Instead, we generate a mock center-biased "saliency" map proxy which will
        allow the rest of the HiSAT pipeline to run and train seamlessly out of the box.
        
        If you have TranSalNet set up, replace this function to pass frames through it.
        
        Returns:
            spatial_maps: (N, 1, 56, 56) tensor
            s_scores: (N,) tensor
        """
        N = frames_tensor.size(0)
        
        # Center bias mask
        y, x = torch.meshgrid(torch.linspace(-1, 1, 56), torch.linspace(-1, 1, 56), indexing='ij')
        center_bias = torch.exp(-(x**2 + y**2) / 0.5)
        
        # Add some variation based on frame intensity
        frame_means = frames_tensor.mean(dim=(1, 2, 3)).view(N, 1, 1, 1) # (N, 1, 1, 1)
        spatial_maps = center_bias.unsqueeze(0).unsqueeze(0).repeat(N, 1, 1, 1) * (0.5 + frame_means * 0.5)
        
        # Normalize maps to [0, 1] per frame
        maps_min = spatial_maps.view(N, -1).min(dim=1)[0].view(N, 1, 1, 1)
        maps_max = spatial_maps.view(N, -1).max(dim=1)[0].view(N, 1, 1, 1)
        spatial_maps = (spatial_maps - maps_min) / (maps_max - maps_min + 1e-6)
        
        # Saliency scalar score is the mean of the map
        s_scores = spatial_maps.mean(dim=(1, 2, 3))
        
        return spatial_maps, s_scores
