#!/usr/bin/env python3
"""
Automated Script to Reproduce All Results

This script trains and evaluates all models to verify the reported results:
- Model V1 (Baseline): 31.70% test acc@1
- Model Advanced (Without Filtering): 44.43% test acc@1  
- Model Advanced (With Filtering): 36.75% test acc@1

Usage:
    conda activate mlenv
    python reproduce_results.py [--train] [--eval-only]

Options:
    --train        Train models from scratch (default: use saved checkpoints)
    --eval-only    Only evaluate existing models without training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import pickle
import random
import argparse
import json
from tqdm import tqdm
import sys

from hierarchical_s2_model import HierarchicalS2Model
from hierarchical_s2_model_advanced import AdvancedHierarchicalS2Model
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


def train_baseline(device, train_loader, val_loader, hierarchy_map):
    """Train Model V1 (Baseline)"""
    print("\n" + "="*80)
    print("TRAINING MODEL V1 (BASELINE)")
    print("="*80)
    
    config = {
        'vocab_l11': 315,
        'vocab_l13': 675,
        'vocab_l14': 930,
        'vocab_X': 1190,
        'd_model': 80,
        'nhead': 4,
        'num_layers': 2,
        'dropout': 0.2,
    }
    
    model = HierarchicalS2Model(config).to(device)
    num_params = model.count_parameters()
    print(f"Parameters: {num_params:,} / 700,000")
    assert num_params <= 700000, f"Model exceeds parameter budget: {num_params}"
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizer, max_lr=0.001, epochs=100,
        steps_per_epoch=len(train_loader),
        pct_start=0.3, div_factor=25.0
    )
    
    # Loss weights
    loss_weights = {'l11': 0.1, 'l13': 0.2, 'l14': 0.2, 'X': 0.5}
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    patience_counter = 0
    patience = 20
    
    for epoch in range(1, 101):
        # Train
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            l11_seq = batch['l11_seq'].to(device)
            l13_seq = batch['l13_seq'].to(device)
            l14_seq = batch['l14_seq'].to(device)
            X_seq = batch['X_seq'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            
            l11_y = batch['l11_y'].to(device)
            l13_y = batch['l13_y'].to(device)
            l14_y = batch['l14_y'].to(device)
            X_y = batch['X_y'].to(device)
            
            outputs = model(l11_seq, l13_seq, l14_seq, X_seq, padding_mask)
            
            loss_l11 = criterion(outputs['logits_l11'], l11_y)
            loss_l13 = criterion(outputs['logits_l13'], l13_y)
            loss_l14 = criterion(outputs['logits_l14'], l14_y)
            loss_X = criterion(outputs['logits_X'], X_y)
            
            loss = (loss_weights['l11'] * loss_l11 + 
                   loss_weights['l13'] * loss_l13 + 
                   loss_weights['l14'] * loss_l14 + 
                   loss_weights['X'] * loss_X)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        # Validate
        model.eval()
        val_metrics = {'correct@1': 0, 'correct@3': 0, 'correct@5': 0, 
                       'correct@10': 0, 'rr': 0, 'ndcg': 0, 'f1': 0, 'total': 0}
        
        with torch.no_grad():
            for batch in val_loader:
                l11_seq = batch['l11_seq'].to(device)
                l13_seq = batch['l13_seq'].to(device)
                l14_seq = batch['l14_seq'].to(device)
                X_seq = batch['X_seq'].to(device)
                padding_mask = batch['padding_mask'].to(device)
                X_y = batch['X_y'].to(device)
                
                outputs = model(l11_seq, l13_seq, l14_seq, X_seq, padding_mask)
                result, _, _ = calculate_correct_total_prediction(outputs['logits_X'], X_y)
                
                val_metrics['correct@1'] += result[0]
                val_metrics['correct@3'] += result[1]
                val_metrics['correct@5'] += result[2]
                val_metrics['correct@10'] += result[3]
                val_metrics['rr'] += result[4]
                val_metrics['ndcg'] += result[5]
                val_metrics['total'] += result[6]
        
        val_perf = get_performance_dict(val_metrics)
        val_acc = val_perf['acc@1']
        
        print(f"Epoch {epoch}: Val acc@1 = {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config,
            }, 'best_model.pt')
            print(f"  ✓ New best: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    return 'best_model.pt'


def train_advanced(device, train_loader, val_loader, hierarchy_map):
    """Train Advanced Model"""
    print("\n" + "="*80)
    print("TRAINING ADVANCED MODEL")
    print("="*80)
    
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
    
    model = AdvancedHierarchicalS2Model(config).to(device)
    num_params = model.count_parameters()
    print(f"Parameters: {num_params:,} / 700,000")
    assert num_params <= 700000, f"Model exceeds parameter budget: {num_params}"
    
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.005)
    scheduler = OneCycleLR(
        optimizer, max_lr=0.002, epochs=150,
        steps_per_epoch=len(train_loader),
        pct_start=0.25, div_factor=20.0, final_div_factor=1000.0
    )
    
    loss_weights = {'l11': 0.05, 'l13': 0.1, 'l14': 0.15, 'X': 0.7}
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    patience_counter = 0
    patience = 30
    
    for epoch in range(1, 151):
        # Train
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
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
            
            outputs = model(l11_seq, l13_seq, l14_seq, X_seq, user_seq, 
                          padding_mask, hierarchy_map=hierarchy_map, use_filtering=False)
            
            loss_l11 = criterion(outputs['logits_l11'], l11_y)
            loss_l13 = criterion(outputs['logits_l13'], l13_y)
            loss_l14 = criterion(outputs['logits_l14'], l14_y)
            loss_X = criterion(outputs['logits_X'], X_y)
            
            loss = (loss_weights['l11'] * loss_l11 + 
                   loss_weights['l13'] * loss_l13 + 
                   loss_weights['l14'] * loss_l14 + 
                   loss_weights['X'] * loss_X)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        # Validate
        model.eval()
        val_metrics = {'correct@1': 0, 'correct@3': 0, 'correct@5': 0, 
                       'correct@10': 0, 'rr': 0, 'ndcg': 0, 'f1': 0, 'total': 0}
        
        with torch.no_grad():
            for batch in val_loader:
                l11_seq = batch['l11_seq'].to(device)
                l13_seq = batch['l13_seq'].to(device)
                l14_seq = batch['l14_seq'].to(device)
                X_seq = batch['X_seq'].to(device)
                user_seq = batch['user_seq'].to(device)
                padding_mask = batch['padding_mask'].to(device)
                X_y = batch['X_y'].to(device)
                
                outputs = model(l11_seq, l13_seq, l14_seq, X_seq, user_seq, 
                              padding_mask, hierarchy_map=hierarchy_map, use_filtering=True)
                result, _, _ = calculate_correct_total_prediction(outputs['logits_X'], X_y)
                
                val_metrics['correct@1'] += result[0]
                val_metrics['correct@3'] += result[1]
                val_metrics['correct@5'] += result[2]
                val_metrics['correct@10'] += result[3]
                val_metrics['rr'] += result[4]
                val_metrics['ndcg'] += result[5]
                val_metrics['total'] += result[6]
        
        val_perf = get_performance_dict(val_metrics)
        val_acc = val_perf['acc@1']
        
        print(f"Epoch {epoch}: Val acc@1 = {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config,
            }, 'best_model_advanced.pt')
            print(f"  ✓ New best: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    return 'best_model_advanced.pt'


def evaluate_baseline(device, test_loader, hierarchy_map):
    """Evaluate Model V1 on test set"""
    print("\n" + "="*80)
    print("EVALUATING MODEL V1 (BASELINE)")
    print("="*80)
    
    checkpoint = torch.load('best_model.pt')
    model = HierarchicalS2Model(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Parameters: {model.count_parameters():,}")
    
    test_metrics = {'correct@1': 0, 'correct@3': 0, 'correct@5': 0, 
                    'correct@10': 0, 'rr': 0, 'ndcg': 0, 'f1': 0, 'total': 0}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            l11_seq = batch['l11_seq'].to(device)
            l13_seq = batch['l13_seq'].to(device)
            l14_seq = batch['l14_seq'].to(device)
            X_seq = batch['X_seq'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            X_y = batch['X_y'].to(device)
            
            outputs = model(l11_seq, l13_seq, l14_seq, X_seq, padding_mask)
            result, _, _ = calculate_correct_total_prediction(outputs['logits_X'], X_y)
            
            test_metrics['correct@1'] += result[0]
            test_metrics['correct@3'] += result[1]
            test_metrics['correct@5'] += result[2]
            test_metrics['correct@10'] += result[3]
            test_metrics['rr'] += result[4]
            test_metrics['ndcg'] += result[5]
            test_metrics['total'] += result[6]
    
    perf = get_performance_dict(test_metrics)
    
    print(f"\nTest Results:")
    print(f"  acc@1:  {perf['acc@1']:.2f}%")
    print(f"  acc@5:  {perf['acc@5']:.2f}%")
    print(f"  acc@10: {perf['acc@10']:.2f}%")
    print(f"  MRR:    {perf['mrr']:.2f}%")
    print(f"  NDCG:   {perf['ndcg']:.2f}%")
    
    return perf


def evaluate_advanced(device, test_loader, hierarchy_map):
    """Evaluate Advanced Model on test set (both with and without filtering)"""
    print("\n" + "="*80)
    print("EVALUATING ADVANCED MODEL")
    print("="*80)
    
    checkpoint = torch.load('best_model_advanced.pt')
    model = AdvancedHierarchicalS2Model(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Parameters: {model.count_parameters():,}")
    
    # WITHOUT filtering
    print("\n1. WITHOUT hierarchical filtering:")
    test_metrics_no = {'correct@1': 0, 'correct@3': 0, 'correct@5': 0, 
                       'correct@10': 0, 'rr': 0, 'ndcg': 0, 'f1': 0, 'total': 0}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing (no filter)'):
            l11_seq = batch['l11_seq'].to(device)
            l13_seq = batch['l13_seq'].to(device)
            l14_seq = batch['l14_seq'].to(device)
            X_seq = batch['X_seq'].to(device)
            user_seq = batch['user_seq'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            X_y = batch['X_y'].to(device)
            
            outputs = model(l11_seq, l13_seq, l14_seq, X_seq, user_seq, 
                          padding_mask, hierarchy_map=None, use_filtering=False)
            result, _, _ = calculate_correct_total_prediction(outputs['logits_X'], X_y)
            
            test_metrics_no['correct@1'] += result[0]
            test_metrics_no['correct@3'] += result[1]
            test_metrics_no['correct@5'] += result[2]
            test_metrics_no['correct@10'] += result[3]
            test_metrics_no['rr'] += result[4]
            test_metrics_no['ndcg'] += result[5]
            test_metrics_no['total'] += result[6]
    
    perf_no = get_performance_dict(test_metrics_no)
    
    print(f"  acc@1:  {perf_no['acc@1']:.2f}%")
    print(f"  acc@5:  {perf_no['acc@5']:.2f}%")
    print(f"  acc@10: {perf_no['acc@10']:.2f}%")
    print(f"  MRR:    {perf_no['mrr']:.2f}%")
    print(f"  NDCG:   {perf_no['ndcg']:.2f}%")
    
    # WITH filtering
    print("\n2. WITH hierarchical filtering:")
    test_metrics_yes = {'correct@1': 0, 'correct@3': 0, 'correct@5': 0, 
                        'correct@10': 0, 'rr': 0, 'ndcg': 0, 'f1': 0, 'total': 0}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing (with filter)'):
            l11_seq = batch['l11_seq'].to(device)
            l13_seq = batch['l13_seq'].to(device)
            l14_seq = batch['l14_seq'].to(device)
            X_seq = batch['X_seq'].to(device)
            user_seq = batch['user_seq'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            X_y = batch['X_y'].to(device)
            
            outputs = model(l11_seq, l13_seq, l14_seq, X_seq, user_seq, 
                          padding_mask, hierarchy_map=hierarchy_map, use_filtering=True)
            result, _, _ = calculate_correct_total_prediction(outputs['logits_X'], X_y)
            
            test_metrics_yes['correct@1'] += result[0]
            test_metrics_yes['correct@3'] += result[1]
            test_metrics_yes['correct@5'] += result[2]
            test_metrics_yes['correct@10'] += result[3]
            test_metrics_yes['rr'] += result[4]
            test_metrics_yes['ndcg'] += result[5]
            test_metrics_yes['total'] += result[6]
    
    perf_yes = get_performance_dict(test_metrics_yes)
    
    print(f"  acc@1:  {perf_yes['acc@1']:.2f}%")
    print(f"  acc@5:  {perf_yes['acc@5']:.2f}%")
    print(f"  acc@10: {perf_yes['acc@10']:.2f}%")
    print(f"  MRR:    {perf_yes['mrr']:.2f}%")
    print(f"  NDCG:   {perf_yes['ndcg']:.2f}%")
    
    return perf_no, perf_yes


def main():
    parser = argparse.ArgumentParser(description='Reproduce all model results')
    parser.add_argument('--train', action='store_true', 
                       help='Train models from scratch')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate existing models')
    args = parser.parse_args()
    
    # Setup
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This requires a GPU.")
        sys.exit(1)
    
    print("="*80)
    print("AUTOMATED RESULTS REPRODUCTION")
    print("="*80)
    print(f"Device: {device}")
    print(f"Random seed: 42")
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=64, num_workers=4)
    
    with open('s2_hierarchy_mapping.pkl', 'rb') as f:
        hierarchy_map = pickle.load(f)
    
    # Train or load models
    if args.train:
        print("\nTraining models from scratch...")
        train_baseline(device, train_loader, val_loader, hierarchy_map)
        train_advanced(device, train_loader, val_loader, hierarchy_map)
    elif not args.eval_only:
        print("\nUsing saved models (use --train to train from scratch)")
    
    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION PHASE")
    print("="*80)
    
    perf_v1 = evaluate_baseline(device, test_loader, hierarchy_map)
    perf_adv_no, perf_adv_yes = evaluate_advanced(device, test_loader, hierarchy_map)
    
    # Verification
    print("\n" + "="*80)
    print("VERIFICATION REPORT")
    print("="*80)
    
    expected = {
        'v1': 31.70,
        'adv_no': 44.43,
        'adv_yes': 36.75
    }
    
    tolerance = 1.0  # Allow 1% deviation
    
    v1_match = abs(perf_v1['acc@1'] - expected['v1']) <= tolerance
    adv_no_match = abs(perf_adv_no['acc@1'] - expected['adv_no']) <= tolerance
    adv_yes_match = abs(perf_adv_yes['acc@1'] - expected['adv_yes']) <= tolerance
    
    print(f"\nModel V1 (Baseline):")
    print(f"  Expected: {expected['v1']:.2f}%")
    print(f"  Actual:   {perf_v1['acc@1']:.2f}%")
    print(f"  Status:   {'✓ VERIFIED' if v1_match else '✗ MISMATCH'}")
    
    print(f"\nModel Advanced (Without Filtering):")
    print(f"  Expected: {expected['adv_no']:.2f}%")
    print(f"  Actual:   {perf_adv_no['acc@1']:.2f}%")
    print(f"  Status:   {'✓ VERIFIED' if adv_no_match else '✗ MISMATCH'}")
    
    print(f"\nModel Advanced (With Filtering):")
    print(f"  Expected: {expected['adv_yes']:.2f}%")
    print(f"  Actual:   {perf_adv_yes['acc@1']:.2f}%")
    print(f"  Status:   {'✓ VERIFIED' if adv_yes_match else '✗ MISMATCH'}")
    
    print("\n" + "="*80)
    if v1_match and adv_no_match and adv_yes_match:
        print("ALL RESULTS VERIFIED SUCCESSFULLY ✓")
    else:
        print("VERIFICATION FAILED - Some results don't match expected values")
    print("="*80)
    
    # Save report
    report = {
        'v1': {'expected': expected['v1'], 'actual': float(perf_v1['acc@1']), 
               'verified': bool(v1_match), 'full_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in perf_v1.items()}},
        'advanced_no_filter': {'expected': expected['adv_no'], 'actual': float(perf_adv_no['acc@1']),
                               'verified': bool(adv_no_match), 'full_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in perf_adv_no.items()}},
        'advanced_with_filter': {'expected': expected['adv_yes'], 'actual': float(perf_adv_yes['acc@1']),
                                 'verified': bool(adv_yes_match), 'full_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in perf_adv_yes.items()}},
    }
    
    with open('verification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\nReport saved to verification_report.json")


if __name__ == '__main__':
    main()
