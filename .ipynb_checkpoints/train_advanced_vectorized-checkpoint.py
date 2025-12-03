#!/usr/bin/env python3
"""
Train Advanced model (vectorized) and verify it achieves 44.43% acc@1 on test set
This is the optimized version with vectorized operations for faster training
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import json
import time

from dataset import create_dataloaders
from hierarchical_s2_model_advanced import AdvancedHierarchicalS2Model
from metrics import calculate_correct_total_prediction, get_performance_dict

import random
import numpy as np

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, optimizer, loss_weights, device, scheduler=None):
    model.train()
    total_loss = 0
    total_loss_X = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        l11_seq = batch['l11_seq'].to(device)
        l13_seq = batch['l13_seq'].to(device)
        l14_seq = batch['l14_seq'].to(device)
        X_seq = batch['X_seq'].to(device)
        user_seq = batch['user_seq'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        
        # Targets (next location)
        target_l11 = batch['l11_y'].to(device)
        target_l13 = batch['l13_y'].to(device)
        target_l14 = batch['l14_y'].to(device)
        target_X = batch['X_y'].to(device)
        
        optimizer.zero_grad()
        
        # Forward (no filtering during training)
        outputs = model(l11_seq, l13_seq, l14_seq, X_seq, user_seq, 
                       padding_mask, hierarchy_map=None, use_filtering=False)
        
        # Compute hierarchical losses
        loss_l11 = nn.CrossEntropyLoss()(outputs['logits_l11'], target_l11)
        loss_l13 = nn.CrossEntropyLoss()(outputs['logits_l13'], target_l13)
        loss_l14 = nn.CrossEntropyLoss()(outputs['logits_l14'], target_l14)
        loss_X = nn.CrossEntropyLoss()(outputs['logits_X'], target_X)
        
        # Weighted combination
        loss = (loss_weights['l11'] * loss_l11 + 
                loss_weights['l13'] * loss_l13 +
                loss_weights['l14'] * loss_l14 + 
                loss_weights['X'] * loss_X)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        total_loss_X += loss_X.item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'X': f'{loss_X.item():.4f}'})
    
    return total_loss / len(train_loader), total_loss_X / len(train_loader)


def validate(model, val_loader, device, hierarchy_map=None, use_filtering=False):
    model.eval()
    
    all_correct = None
    all_true_y = []
    all_top1 = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for batch in pbar:
            l11_seq = batch['l11_seq'].to(device)
            l13_seq = batch['l13_seq'].to(device)
            l14_seq = batch['l14_seq'].to(device)
            X_seq = batch['X_seq'].to(device)
            user_seq = batch['user_seq'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            target_X = batch['X_y'].to(device)
            
            outputs = model(l11_seq, l13_seq, l14_seq, X_seq, user_seq, 
                          padding_mask, hierarchy_map, use_filtering)
            
            correct, true_y, top1 = calculate_correct_total_prediction(
                outputs['logits_X'], target_X
            )
            
            if all_correct is None:
                all_correct = correct
            else:
                all_correct += correct
            
            all_true_y.extend(true_y.numpy())
            all_top1.extend(top1.numpy())
    
    # Calculate F1 score
    from sklearn.metrics import f1_score
    f1 = f1_score(all_true_y, all_top1, average='weighted')
    
    return_dict = {
        'correct@1': all_correct[0],
        'correct@3': all_correct[1],
        'correct@5': all_correct[2],
        'correct@10': all_correct[3],
        'rr': all_correct[4],
        'ndcg': all_correct[5],
        'f1': f1,
        'total': all_correct[6],
    }
    
    perf = get_performance_dict(return_dict)
    return perf


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA required!")
        return 0.0
    
    # Advanced config with all features
    config = {
        'vocab_l11': 315,
        'vocab_l13': 675,
        'vocab_l14': 930,
        'vocab_X': 1190,
        'num_users': 50,
        'd_model': 96,
        'nhead': 4,
        'dropout': 0.25,
    }
    
    # Create model
    model = AdvancedHierarchicalS2Model(config).to(device)
    num_params = model.count_parameters()
    
    print("\n" + "="*80)
    print("ADVANCED MODEL (VECTORIZED) - VERIFICATION RUN")
    print("="*80)
    print(f"Parameters: {num_params:,} / 700,000")
    print(f"Budget remaining: {700000 - num_params:,}")
    print("\nOptimizations:")
    print("  ✓ Vectorized frequency computation (no loops)")
    print("  ✓ Vectorized recency computation (no loops)")
    print("  ✓ Vectorized filtering (torch.where)")
    print("\nFeatures:")
    print("  ✓ User embeddings (personalization)")
    print("  ✓ Frequency & recency features")
    print("  ✓ Multi-head copy attention")
    print("  ✓ Copy-generate gate")
    print("\nTarget: 44.43% acc@1 on test set (no filtering)")
    print("="*80)
    
    assert num_params <= 700000
    
    # Dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=64, num_workers=4)
    
    # Load hierarchy
    with open('s2_hierarchy_mapping.pkl', 'rb') as f:
        hierarchy_map = pickle.load(f)
    
    # Loss weights
    loss_weights = {
        'l11': 0.1,
        'l13': 0.15,
        'l14': 0.2,
        'X': 0.55
    }
    
    # Optimizer with warmup
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    total_steps = len(train_loader) * 30
    warmup_steps = len(train_loader) * 2
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return max(0.1, 0.5 * (1 + np.cos(np.pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training
    print("\nTraining for 30 epochs...")
    best_acc = 0.0
    patience = 8
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(1, 31):
        train_loss, train_loss_X = train_epoch(
            model, train_loader, optimizer, loss_weights, device, scheduler
        )
        
        # Validate WITHOUT filtering (this is our target metric)
        val_perf = validate(model, val_loader, device, hierarchy_map, use_filtering=False)
        
        print(f"\nEpoch {epoch:3d} | Loss: {train_loss:.4f} | "
              f"Val acc@1 (X): {val_perf['acc@1']:.2f}% | "
              f"acc@5: {val_perf['acc@5']:.2f}% | "
              f"MRR: {val_perf['mrr']:.2f}%")
        
        # Save best model
        if val_perf['acc@1'] > best_acc:
            best_acc = val_perf['acc@1']
            torch.save(model.state_dict(), 'best_model_advanced_vectorized.pt')
            patience_counter = 0
            print(f"✓ NEW BEST: {best_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    training_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Training completed in {training_time/60:.1f} minutes")
    print(f"Best validation acc@1: {best_acc:.2f}%")
    
    # Load best model and test
    print("\n" + "="*80)
    print("TESTING BEST MODEL (NO FILTERING)")
    print("="*80)
    model.load_state_dict(torch.load('best_model_advanced_vectorized.pt'))
    
    test_perf = validate(model, test_loader, device, hierarchy_map, use_filtering=False)
    
    print(f"\nTest Results (NO filtering):")
    print(f"  acc@1:  {test_perf['acc@1']:.2f}%")
    print(f"  acc@5:  {test_perf['acc@5']:.2f}%")
    print(f"  acc@10: {test_perf['acc@10']:.2f}%")
    print(f"  MRR:    {test_perf['mrr']:.2f}%")
    print(f"  NDCG:   {test_perf['ndcg']:.2f}%")
    
    # Save results
    results = {
        'no_filtering': test_perf,
        'training_time_minutes': training_time / 60,
        'best_val_acc': best_acc,
        'num_parameters': num_params,
    }
    
    with open('test_results_advanced_vectorized.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("VERIFICATION STATUS:")
    target_acc = 44.43
    achieved_acc = test_perf['acc@1']
    tolerance = 0.5  # Allow 0.5% difference due to randomness
    
    if abs(achieved_acc - target_acc) <= tolerance:
        print(f"✓ SUCCESS! Achieved {achieved_acc:.2f}% (target: {target_acc:.2f}%)")
        print(f"  Difference: {achieved_acc - target_acc:+.2f}%")
    else:
        print(f"⚠ WARNING: Achieved {achieved_acc:.2f}% (target: {target_acc:.2f}%)")
        print(f"  Difference: {achieved_acc - target_acc:+.2f}%")
        print(f"  Note: Small variations are normal due to randomness")
    
    print(f"{'='*80}\n")
    
    return test_perf['acc@1']


if __name__ == '__main__':
    final_acc = main()
    print(f"Final test acc@1: {final_acc:.2f}%")
