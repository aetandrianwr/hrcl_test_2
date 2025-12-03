import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pickle
import random
import os
from tqdm import tqdm
import json

from hierarchical_s2_model_v4 import HierarchicalS2ModelV4
from dataset import create_dataloaders
from metrics import evaluate_model

# Set random seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prob).sum(dim=1).mean()
        return loss


def train_epoch(model, train_loader, optimizer, device, loss_weights, epoch, use_label_smoothing=True):
    model.train()
    
    total_loss = 0
    total_loss_l11 = 0
    total_loss_l13 = 0
    total_loss_l14 = 0
    total_loss_X = 0
    
    if use_label_smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch in pbar:
        l11_seq = batch['l11_seq'].to(device)
        l13_seq = batch['l13_seq'].to(device)
        l14_seq = batch['l14_seq'].to(device)
        X_seq = batch['X_seq'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        
        l11_y = batch['l11_y'].to(device)
        l13_y = batch['l13_y'].to(device)
        l14_y = batch['l14_y'].to(device)
        X_y = batch['X_y'].to(device)
        
        # Forward pass
        outputs = model(l11_seq, l13_seq, l14_seq, X_seq, padding_mask)
        
        # Compute losses
        loss_l11 = criterion(outputs['logits_l11'], l11_y)
        loss_l13 = criterion(outputs['logits_l13'], l13_y)
        loss_l14 = criterion(outputs['logits_l14'], l14_y)
        loss_X = criterion(outputs['logits_X'], X_y)
        
        # Weighted loss - prioritize exact location heavily
        loss = (loss_weights['l11'] * loss_l11 + 
                loss_weights['l13'] * loss_l13 + 
                loss_weights['l14'] * loss_l14 + 
                loss_weights['X'] * loss_X)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_loss_l11 += loss_l11.item()
        total_loss_l13 += loss_l13.item()
        total_loss_l14 += loss_l14.item()
        total_loss_X += loss_X.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'X': f'{loss_X.item():.4f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    return {
        'total': avg_loss,
        'l11': total_loss_l11 / len(train_loader),
        'l13': total_loss_l13 / len(train_loader),
        'l14': total_loss_l14 / len(train_loader),
        'X': total_loss_X / len(train_loader),
    }


def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        return
    
    # Model configuration - optimized within 700k budget
    config = {
        'vocab_l11': 315,
        'vocab_l13': 675,
        'vocab_l14': 930,
        'vocab_X': 1190,
        'd_model': 128,    # Maximum within budget
        'nhead': 4,
        'dropout': 0.3,
    }
    
    # Create model
    model = HierarchicalS2ModelV4(config).to(device)
    num_params = model.count_parameters()
    print(f"\nModel parameters: {num_params:,} / 700,000")
    assert num_params <= 700000, f"Model has {num_params:,} parameters, exceeding 700k limit!"
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=256, num_workers=4)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Load hierarchy mapping
    with open('s2_hierarchy_mapping.pkl', 'rb') as f:
        hierarchy_map = pickle.load(f)
    
    # Loss weights - heavily prioritize exact location
    loss_weights = {
        'l11': 0.05,
        'l13': 0.1,
        'l14': 0.2,
        'X': 0.65,
    }
    
    # Optimizer with higher learning rate
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01, betas=(0.9, 0.999))
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-6)
    
    # Training loop
    num_epochs = 150
    best_val_acc = 0
    best_epoch = 0
    patience = 20
    patience_counter = 0
    
    print("\nStarting training with V4 model...")
    for epoch in range(1, num_epochs + 1):
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, device, loss_weights, epoch, use_label_smoothing=True)
        
        # Validate
        val_perf = evaluate_model(model, val_loader, device, hierarchy_map, use_hierarchy=False)
        
        # Update scheduler
        scheduler.step()
        
        # Print results
        print(f"\nEpoch {epoch}/{num_epochs} | LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"Train Loss: {train_losses['total']:.4f} (X: {train_losses['X']:.4f})")
        print(f"Val acc@1: X={val_perf['X']['acc@1']:.2f}% | "
              f"l11={val_perf['l11']['acc@1']:.2f}% | "
              f"l13={val_perf['l13']['acc@1']:.2f}% | "
              f"l14={val_perf['l14']['acc@1']:.2f}%")
        print(f"Val acc@5: X={val_perf['X']['acc@5']:.2f}% | MRR: X={val_perf['X']['mrr']:.2f}%")
        
        # Save best model
        val_acc_X = val_perf['X']['acc@1']
        if val_acc_X > best_val_acc:
            best_val_acc = val_acc_X
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_perf': val_perf,
                'config': config,
            }, 'best_model_v4.pt')
            print(f"✓ Saved best model with val acc@1 (X) = {val_acc_X:.2f}%")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
        
        # Check if target achieved
        if val_acc_X >= 50.0:
            print(f"\n🎉 TARGET ACHIEVED! Val acc@1 (X) = {val_acc_X:.2f}% >= 50%")
            break
    
    # Load best model and evaluate on test set
    print(f"\n{'='*60}")
    print(f"Loading best model from epoch {best_epoch}")
    checkpoint = torch.load('best_model_v4.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nEvaluating on test set...")
    test_perf = evaluate_model(model, test_loader, device, hierarchy_map, use_hierarchy=False)
    
    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS (V4 Model)")
    print(f"{'='*60}")
    for level in ['l11', 'l13', 'l14', 'X']:
        print(f"\nLevel {level}:")
        print(f"  acc@1:  {test_perf[level]['acc@1']:.2f}%")
        print(f"  acc@5:  {test_perf[level]['acc@5']:.2f}%")
        print(f"  acc@10: {test_perf[level]['acc@10']:.2f}%")
        print(f"  MRR:    {test_perf[level]['mrr']:.2f}%")
        print(f"  NDCG:   {test_perf[level]['ndcg']:.2f}%")
    
    print(f"\n{'='*60}")
    print(f"Model parameters: {num_params:,} / 700,000")
    print(f"Best validation acc@1 (X): {best_val_acc:.2f}%")
    print(f"Test acc@1 (X): {test_perf['X']['acc@1']:.2f}%")
    
    if test_perf['X']['acc@1'] >= 50.0:
        print(f"✅ SUCCESS: Achieved {test_perf['X']['acc@1']:.2f}% >= 50% target!")
    else:
        print(f"❌ Target not achieved: {test_perf['X']['acc@1']:.2f}% < 50%")
        print("\nWill continue with further improvements...")
    
    # Save test results
    with open('test_results_v4.json', 'w') as f:
        json.dump(test_perf, f, indent=2)
    
    return test_perf['X']['acc@1']


if __name__ == '__main__':
    final_acc = main()
