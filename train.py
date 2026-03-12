import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np

# Set up relative imports for submodules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hisat import HiSAT
from data.dataset import get_loaders
from utils.losses import HisatLoss
from utils.metrics import compute_f_score

def generate_mock_splits(n_videos=50):
    """
    Generate mock 5-fold splits as placeholders for TVSum videos.
    TVSum has 50 videos. 
    Returns dictionary: fold_idx -> {'train_keys': [...], 'test_keys': [...]}
    """
    keys = [f"video_{i}" for i in range(1, n_videos + 1)]
    folds = {}
    fold_size = n_videos // 5
    for i in range(5):
        test_keys = keys[i*fold_size : (i+1)*fold_size]
        train_keys = list(set(keys) - set(test_keys))
        folds[i+1] = {'train_keys': train_keys, 'test_keys': test_keys}
    return folds

def create_mock_h5(h5_path, n_videos=50):
    """
    Create a mock HDF5 file so the script can run right away without actual TVSum downloaded.
    This demonstrates the architecture handles variable length videos + training loop perfectly.
    """
    import h5py
    if not os.path.exists(h5_path):
        print(f"Creating mock TVSum dataset at {h5_path}...")
        with h5py.File(h5_path, 'w') as f:
            for i in range(1, n_videos + 1):
                N = np.random.randint(100, 300) # random video length
                grp = f.create_group(f"video_{i}")
                grp.create_dataset('features', data=np.random.rand(N, 1024).astype(np.float32))
                grp.create_dataset('saliency_features', data=np.random.rand(N, 256).astype(np.float32))
                grp.create_dataset('gtscore', data=np.random.rand(N).astype(np.float32))
                # Create random shot boundaries
                cps = []
                for j in range(0, N, max(1, N // 10)):
                    cps.append([j, min(j+max(1, N//10), N)])
                grp.create_dataset('change_points', data=np.array(cps, dtype=np.int32))
    return True

def train():
    parser = argparse.ArgumentParser(description="Train HiSAT on TVSum")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--data", type=str, default="mock_tvsum.h5", help="Path to TVSum h5 features")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (fewer epochs)")
    args = parser.parse_args()

    # Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Mock data setup if file doesn't exist
    if not os.path.exists(args.data):
        create_mock_h5(args.data)
        
    splits_dict = generate_mock_splits()
    
    # Loss criterion
    criterion = HisatLoss(
        lambda_div=config['loss']['lambda_diversity'],
        lambda_sp=config['loss']['lambda_sparsity'],
        alpha=config['loss']['saliency_weight_alpha']
    ).to(device)

    epochs = config['training']['epochs'] if not args.debug else 2
    
    for fold in range(1, config['evaluation']['n_folds'] + 1):
        print(f"\\n{'='*20} Fold {fold} {'='*20}")
        
        train_loader, test_loader = get_loaders(args.data, splits_dict, fold=fold, batch_size=config['training']['batch_size'])
        
        # Initialize Model
        model = HiSAT(
            sem_dim=config['model']['sem_feat_dim'],
            sal_dim=config['model']['sal_feat_dim'],
            d_model=config['model']['d_model'],
            n_heads=config['model']['n_heads'],
            local_window=config['model']['local_window_size'],
            n_local=config['model']['n_local_layers'],
            n_shot=config['model']['n_shot_layers'],
            n_scene=config['model']['n_scene_layers'],
            dropout=config['model']['dropout'],
            gamma_init=config['model']['gamma_init']
        ).to(device)
        
        optimizer = Adam(
            model.parameters(), 
            lr=float(config['training']['learning_rate']),
            weight_decay=float(config['training']['weight_decay']),
            betas=tuple(config['training']['betas'])
        )
        
        scheduler = StepLR(
            optimizer, 
            step_size=config['training']['lr_step_size'], 
            gamma=config['training']['lr_gamma']
        )
        
        best_f_score = 0.0
        
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            
            for keys, F_sem, F_sal, s_scores, gt_scores, shot_boundaries in train_loader:
                F_sem = F_sem.to(device)
                F_sal = F_sal.to(device)
                s_scores = s_scores.to(device)
                gt_scores = gt_scores.to(device)
                
                optimizer.zero_grad()
                
                # Forward Pass
                pred_scores, pred_budget, h_temporal = model(F_sem, F_sal, s_scores, shot_boundaries)
                
                # Compute Loss
                total_loss, l_imp, l_div, l_sp = criterion(pred_scores, gt_scores, h_temporal, s_scores)
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config['training']['gradient_clip']))
                
                optimizer.step()
                epoch_loss += total_loss.item()
                
            scheduler.step()
            
            # Validation every 5 epochs or in debug mode
            if epoch % 5 == 0 or args.debug:
                model.eval()
                f_scores = []
                with torch.no_grad():
                    for keys, F_sem, F_sal, s_scores, gt_scores, shot_boundaries in test_loader:
                        F_sem = F_sem.to(device)
                        F_sal = F_sal.to(device)
                        s_scores = s_scores.to(device)
                        
                        pred_scores, _, _ = model(F_sem, F_sal, s_scores, shot_boundaries)
                        
                        f = compute_f_score(
                            pred_scores[0].cpu().numpy(), 
                            gt_scores[0].numpy(), 
                            shot_boundaries, 
                            max_budget_ratio=config['evaluation']['summary_proportion']
                        )
                        f_scores.append(f)
                        
                mean_f = np.mean(f_scores)
                print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f} | F-score: {mean_f:.4f}")
                
                if mean_f > best_f_score:
                    best_f_score = mean_f
                    os.makedirs('checkpoints', exist_ok=True)
                    torch.save(model.state_dict(), f"checkpoints/best_model_fold{fold}.pth")
                    
        print(f"Fold {fold} Best F-score: {best_f_score:.4f}")

if __name__ == "__main__":
    train()
