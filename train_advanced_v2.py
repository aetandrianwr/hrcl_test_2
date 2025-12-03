#!/usr/bin/env python3
"""
Training script for Advanced V2 Model (Transformer-based)

This is exactly the same as train_advanced.py but uses the V2 model with Transformer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import pickle
import random
import json
from tqdm import tqdm

from hierarchical_s2_model_advanced_v2 import AdvancedHierarchicalS2ModelV2
from dataset import create_dataloaders
from metrics import calculate_correct_total_prediction, get_performance_dict


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, optimizer, scheduler, criterion, loss_weights, device, hierarchy_map):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(train_loader, desc='Training', leave=False):
        l11_seq = batch['l11_seq'].to(device)
        l13_seq = batch['l13_seq'].to(device)
        l14_seq = batch['l14_seq'].to(device)
        X_seq = batch['X_seq'].to(device)
        user_seq = batch['user_seq'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        
        l11_y = batch['l11_y'].to(device)
        l13_y = batch['l13_y'].to(device)
        l14_y = batch['l14_y'].to(device)
        X_y = batch['X_y'].to(device)
        
        # Forward pass (no filtering during training)
        outputs = model(l11_seq, l13_seq, l14_seq, X_seq, user_seq, 
                       padding_mask, hierarchy_map=hierarchy_map, use_filtering=False)
        
        # Compute losses
        loss_l11 = criterion(outputs['logits_l11'], l11_y)
        loss_l13 = criterion(outputs['logits_l13'], l13_y)
        loss_l14 = criterion(outputs['logits_l14'], l14_y)
        loss_X = criterion(outputs['logits_X'], X_y)
        
        # Weighted sum
        loss = (loss_weights['l11'] * loss_l11 + 
                loss_weights['l13'] * loss_l13 + 
                loss_weights['l14'] * loss_l14 + 
                loss_weights['X'] * loss_X)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate(model, val_loader, device, hierarchy_map, use_filtering=True):
    """Validate model"""
    model.eval()
    
    metrics = {
        'correct@1': 0, 'correct@3': 0, 'correct@5': 0, 'correct@10': 0,
        'rr': 0, 'ndcg': 0, 'f1': 0, 'total': 0
    }
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating', leave=False):
            l11_seq = batch['l11_seq'].to(device)
            l13_seq = batch['l13_seq'].to(device)
            l14_seq = batch['l14_seq'].to(device)
            X_seq = batch['X_seq'].to(device)
            user_seq = batch['user_seq'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            X_y = batch['X_y'].to(device)
            
            outputs = model(l11_seq, l13_seq, l14_seq, X_seq, user_seq, 
                           padding_mask, hierarchy_map=hierarchy_map, use_filtering=use_filtering)
            
            result, _, _ = calculate_correct_total_prediction(outputs['logits_X'], X_y)
            
            metrics['correct@1'] += result[0]
            metrics['correct@3'] += result[1]
            metrics['correct@5'] += result[2]
            metrics['correct@10'] += result[3]
            metrics['rr'] += result[4]
            metrics['ndcg'] += result[5]
            metrics['total'] += result[6]
    
    perf = get_performance_dict(metrics)
    return perf


def main():
    # Setup
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("TRAINING ADVANCED MODEL V2 (TRANSFORMER)")
    print("="*80)
    print(f"Device: {device}")
    print(f"Random seed: 42")
    print()
    
    # Model configuration
    config = {
        'vocab_l11': 315,
        'vocab_l13': 675,
        'vocab_l14': 930,
        'vocab_X': 1190,
        'num_users': 50,
        'd_model': 80,  # Adjusted to fit budget
        'nhead': 4,
        'num_layers': 2,
        'dropout': 0.25,
    }
    
    # Create model
    model = AdvancedHierarchicalS2ModelV2(config).to(device)
    num_params = model.count_parameters()
    
    print(f"Model: Advanced V2 (Transformer)")
    print(f"Parameters: {num_params:,} / 700,000")
    assert num_params <= 700000, f"Model exceeds parameter budget: {num_params}"
    print()
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=64, num_workers=4)
    
    with open('s2_hierarchy_mapping.pkl', 'rb') as f:
        hierarchy_map = pickle.load(f)
    print()
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.005)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.002,
        epochs=150,
        steps_per_epoch=len(train_loader),
        pct_start=0.25,
        div_factor=20.0,
        final_div_factor=1000.0
    )
    
    # Loss function and weights
    criterion = nn.CrossEntropyLoss()
    loss_weights = {'l11': 0.05, 'l13': 0.1, 'l14': 0.15, 'X': 0.7}
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    patience = 30
    
    print("Starting training...")
    print("="*80)
    
    log_file = open('training_advanced_v2_log.txt', 'w')
    
    for epoch in range(1, 151):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, 
                                criterion, loss_weights, device, hierarchy_map)
        
        # Validate (with filtering for early stopping)
        val_perf = validate(model, val_loader, device, hierarchy_map, use_filtering=True)
        val_acc = val_perf['acc@1']
        
        # Log
        log_msg = (f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                  f"Val acc@1 (X): {val_acc:.2f}% | acc@5: {val_perf['acc@5']:.2f}% | "
                  f"MRR: {val_perf['mrr']:.2f}%")
        print(log_msg)
        log_file.write(log_msg + '\n')
        log_file.flush()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config,
            }, 'best_model_advanced_v2.pt')
            
            print(f"✓ NEW BEST: {val_acc:.2f}%")
            log_file.write(f"✓ NEW BEST: {val_acc:.2f}%\n")
            log_file.flush()
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            log_file.write(f"Early stopping at epoch {epoch}\n")
            break
    
    log_file.close()
    
    # Load best model for testing
    print("\n" + "="*80)
    print("TESTING BEST MODEL")
    print("="*80)
    
    checkpoint = torch.load('best_model_advanced_v2.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Best model from epoch {checkpoint['epoch']}")
    print(f"Best validation acc@1: {checkpoint['val_acc']:.2f}%")
    print()
    
    # Test WITHOUT filtering
    print("Testing WITHOUT hierarchical filtering:")
    test_perf_no = validate(model, test_loader, device, hierarchy_map, use_filtering=False)
    
    print(f"  acc@1:  {test_perf_no['acc@1']:.2f}%")
    print(f"  acc@5:  {test_perf_no['acc@5']:.2f}%")
    print(f"  acc@10: {test_perf_no['acc@10']:.2f}%")
    print(f"  MRR:    {test_perf_no['mrr']:.2f}%")
    print(f"  NDCG:   {test_perf_no['ndcg']:.2f}%")
    print()
    
    # Test WITH filtering
    print("Testing WITH hierarchical filtering:")
    test_perf_yes = validate(model, test_loader, device, hierarchy_map, use_filtering=True)
    
    print(f"  acc@1:  {test_perf_yes['acc@1']:.2f}%")
    print(f"  acc@5:  {test_perf_yes['acc@5']:.2f}%")
    print(f"  acc@10: {test_perf_yes['acc@10']:.2f}%")
    print(f"  MRR:    {test_perf_yes['mrr']:.2f}%")
    print(f"  NDCG:   {test_perf_yes['ndcg']:.2f}%")
    print()
    
    # Save results
    results = {
        'model': 'Advanced V2 (Transformer)',
        'parameters': num_params,
        'best_epoch': checkpoint['epoch'],
        'without_filtering': {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
            for k, v in test_perf_no.items()
        },
        'with_filtering': {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
            for k, v in test_perf_yes.items()
        },
        'config': config,
    }
    
    with open('test_results_advanced_v2.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best model saved to: best_model_advanced_v2.pt")
    print(f"Results saved to: test_results_advanced_v2.json")
    print(f"Log saved to: training_advanced_v2_log.txt")
    print()
    print(f"Final test acc@1 (without filtering): {test_perf_no['acc@1']:.2f}%")
    print(f"Final test acc@1 (with filtering):    {test_perf_yes['acc@1']:.2f}%")
    print("="*80)


if __name__ == '__main__':
    main()
