import torch
import torch.nn as nn
import torch.nn.functional as F

class SaliencyFeatureEncoder(nn.Module):
    """
    Lightweight CNN to encode spatial saliency maps into 256-d feature vectors.
    Input: Saliency map of shape (Batch, 1, 56, 56) or (Batch, SeqLen, 1, 56, 56)
    Output: Saliency features of shape (Batch, 256) or (Batch, SeqLen, 256)
    """
    def __init__(self, out_dim=256):
        super(SaliencyFeatureEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        """
        x: (B, N, 1, H, W) or (B, 1, H, W)
        """
        has_seq = False
        if x.dim() == 5:
            has_seq = True
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        
        out = F.relu(self.conv3(out))
        out = self.pool3(out)
        
        out = out.view(out.size(0), -1) # (B*N, 128)
        out = self.fc(out)              # (B*N, 256)
        
        if has_seq:
            out = out.view(B, N, -1)
            
        return out
