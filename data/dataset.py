import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

class TVSumDataset(Dataset):
    """
    Dataset loader for TVSum pre-extracted features.
    Assumes an HDF5 file with structure:
    /video_name/
       - features (N, 1024) semantic
       - saliency_features (N, 256) or saliency_maps (N, 1, 56, 56)
       - gtscore (N,)
       - change_points (num_shots, 2)
       - n_frames (int)
       - n_steps (int)
       ...
    """
    def __init__(self, h5_path, split_keys):
        """
        split_keys: List of video names (keys) belonging to this split (train or test).
        """
        self.h5_path = h5_path
        self.keys = split_keys
        
    def __len__(self):
        return len(self.keys)
        
    def __getitem__(self, idx):
        key = self.keys[idx]
        with h5py.File(self.h5_path, 'r') as hdf:
            video_data = hdf[key]
            
            # Semantic features
            F_sem = torch.tensor(video_data['features'][...], dtype=torch.float32)
            
            # Saliency features (if not present, we will generate dummy ones to allow testing)
            if 'saliency_features' in video_data:
                F_sal = torch.tensor(video_data['saliency_features'][...], dtype=torch.float32)
            elif 'saliency_maps' in video_data:
                F_sal = torch.tensor(video_data['saliency_maps'][...], dtype=torch.float32)
            else:
                # Dummy saliency features (N, 256)
                F_sal = torch.rand(F_sem.size(0), 256, dtype=torch.float32)
                
            # Ground truth scores
            if 'gtscore' in video_data:
                scores = torch.tensor(video_data['gtscore'][...], dtype=torch.float32)
            else:
                scores = torch.zeros(F_sem.size(0), dtype=torch.float32)
                
            # Dummy Saliency Scores (scalar, derived from map or given)
            if 'saliency_scores' in video_data:
                s_scores = torch.tensor(video_data['saliency_scores'][...], dtype=torch.float32)
            else:
                # Mock if missing
                s_scores = torch.rand(F_sem.size(0), dtype=torch.float32)
                
            # Shot boundaries: list of start indices
            if 'change_points' in video_data:
                cps = video_data['change_points'][...]
                # cps is usually (S, 2) array of [start_frame, end_frame]
                shot_boundaries = [int(cp[0]) for cp in cps]
            else:
                # Mock 10 uniform shots if missing
                N = F_sem.size(0)
                shot_boundaries = list(range(0, N, max(1, N // 10)))
                
        return key, F_sem, F_sal, s_scores, scores, shot_boundaries

def custom_collate(batch):
    """
    Since videos have variable lengths, batch size should be 1.
    We just return the items without standard batch stacking (except dim 0).
    """
    key, F_sem, F_sal, s_scores, scores, shot_boundaries = batch[0]
    
    # Add batch dimension to tensors
    F_sem = F_sem.unsqueeze(0)
    F_sal = F_sal.unsqueeze(0)
    s_scores = s_scores.unsqueeze(0)
    scores = scores.unsqueeze(0)
    
    return key, F_sem, F_sal, s_scores, scores, shot_boundaries

def get_loaders(h5_path, splits_dict, fold=1, batch_size=1):
    """
    splits_dict: dict mapping fold_number to {'train_keys': [], 'test_keys': []}
    """
    train_keys = splits_dict[fold]['train_keys']
    test_keys = splits_dict[fold]['test_keys']
    
    train_dataset = TVSumDataset(h5_path, train_keys)
    test_dataset = TVSumDataset(h5_path, test_keys)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    
    return train_loader, test_loader
