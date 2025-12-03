import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import pickle
import random
from tqdm import tqdm
import json

from hierarchical_s2_model_v5 import HierarchicalS2ModelV5
from dataset import create_dataloaders
from metrics import evaluate_model

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


def train_epoch(model, train_loader, optimizer, device, loss_weights, epoch, scheduler=None):
    model.train()
    
    total_loss = 0
    total_loss_X = 0
    
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
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
        
        # Weighted loss
        loss = (loss_weights['l11'] * loss_l11 + 
                loss_weights['l13'] * loss_l13 + 
                loss_weights['l14'] * loss_l14 + 
                loss_weights['X'] * loss_X)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        total_loss_X += loss_X.item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'X': f'{loss_X.item():.4f}'})
    
    return total_loss / len(train_loader), total_loss_X / len(train_loader)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA required but not available!")
        return 0.0
    
    # Best config from testing
    config = {
        'vocab_l11': 315,
        'vocab_l13': 675,
        'vocab_l14': 930,
        'vocab_X': 1190,
        'd_model': 96,
        'nhead': 4,
        'num_layers': 2,
        'factor_dim': 48,
        'dropout': 0.25,  # Reduced dropout for better performance
    }
    
    # Create model
    model = HierarchicalS2ModelV5(config).to(device)
    num_params = model.count_parameters()
    print(f"\nModel V5 parameters: {num_params:,} / 700,000")
    assert num_params <= 700000, f"Exceeded budget: {num_params:,}"
    
    # Dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=128, num_workers=4)
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    # Load hierarchy
    with open('s2_hierarchy_mapping.pkl', 'rb') as f:
        hierarchy_map = pickle.load(f)
    
    # Loss weights - heavily focus on exact location
    loss_weights = {
        'l11': 0.05,
        'l13': 0.1,
        'l14': 0.15,
        'X': 0.7,
    }
    
    # Optimizer - use OneCycleLR for better convergence
    num_epochs = 100
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.003,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    best_val_acc = 0
    best_epoch = 0
    patience = 20
    patience_counter = 0
    
    print("\n" + "="*60)
    print("TRAINING V5 MODEL")
    print("="*60)
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_loss_X = train_epoch(
            model, train_loader, optimizer, device, loss_weights, epoch, scheduler
        )
        
        # Validate
        val_perf = evaluate_model(model, val_loader, device, hierarchy_map)
        
        val_acc_X = val_perf['X']['acc@1']
        
        # Print
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} (X: {train_loss_X:.4f})")
        print(f"Val acc@1: X={val_acc_X:.2f}% | l11={val_perf['l11']['acc@1']:.2f}% | "
              f"l13={val_perf['l13']['acc@1']:.2f}% | l14={val_perf['l14']['acc@1']:.2f}%")
        print(f"Val acc@5: X={val_perf['X']['acc@5']:.2f}% | MRR: {val_perf['X']['mrr']:.2f}%")
        
        # Save best
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
            }, 'best_model_v5.pt')
            print(f"✓ NEW BEST: {val_acc_X:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
        
        # Check target
        if val_acc_X >= 50.0:
            print(f"\n🎉 TARGET ACHIEVED! {val_acc_X:.2f}% >= 50%")
            break
    
    # Test evaluation
    print(f"\n{'='*60}")
    print(f"Loading best model from epoch {best_epoch}")
    checkpoint = torch.load('best_model_v5.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Evaluating on test set...")
    test_perf = evaluate_model(model, test_loader, device, hierarchy_map)
    
    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS (V5 Model)")
    print(f"{'='*60}")
    for level in ['X', 'l11', 'l13', 'l14']:
        print(f"\n{level}:")
        print(f"  acc@1:  {test_perf[level]['acc@1']:.2f}%")
        print(f"  acc@5:  {test_perf[level]['acc@5']:.2f}%")
        print(f"  acc@10: {test_perf[level]['acc@10']:.2f}%")
        print(f"  MRR:    {test_perf[level]['mrr']:.2f}%")
        print(f"  NDCG:   {test_perf[level]['ndcg']:.2f}%")
    
    print(f"\n{'='*60}")
    print(f"Parameters: {num_params:,} / 700,000")
    print(f"Best Val acc@1 (X): {best_val_acc:.2f}%")
    print(f"Test acc@1 (X): {test_perf['X']['acc@1']:.2f}%")
    print(f"{'='*60}")
    
    if test_perf['X']['acc@1'] >= 50.0:
        print(f"✅ SUCCESS!")
    else:
        print(f"Target not reached. Gap: {50.0 - test_perf['X']['acc@1']:.2f}%")
    
    with open('test_results_v5.json', 'w') as f:
        json.dump(test_perf, f, indent=2)
    
    return test_perf['X']['acc@1']


if __name__ == '__main__':
    final_acc = main()
