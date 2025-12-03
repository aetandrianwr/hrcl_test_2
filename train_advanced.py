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

from hierarchical_s2_model_advanced import AdvancedHierarchicalS2Model
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


def train_epoch(model, train_loader, optimizer, device, loss_weights, epoch, scheduler=None, hierarchy_map=None):
    model.train()
    
    total_loss = 0
    total_loss_X = 0
    
    criterion = nn.CrossEntropyLoss()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch in pbar:
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
        
        # Forward pass WITHOUT filtering during training (faster)
        outputs = model(l11_seq, l13_seq, l14_seq, X_seq, user_seq, padding_mask, 
                       hierarchy_map=hierarchy_map, use_filtering=False)
        
        # Compute losses
        loss_l11 = criterion(outputs['logits_l11'], l11_y)
        loss_l13 = criterion(outputs['logits_l13'], l13_y)
        loss_l14 = criterion(outputs['logits_l14'], l14_y)
        loss_X = criterion(outputs['logits_X'], X_y)
        
        # Weighted loss - heavily prioritize X
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
    print("ADVANCED MODEL WITH FULL PATTERN EXPLOITATION")
    print("="*80)
    print(f"Parameters: {num_params:,} / 700,000")
    print(f"Budget remaining: {700000 - num_params:,}")
    print("\nFeatures:")
    print("  ✓ User embeddings (personalization)")
    print("  ✓ Frequency & recency features (75% targets in history)")
    print("  ✓ Multi-head copy attention")
    print("  ✓ Copy-generate gate")
    print("  ✓ Hierarchical candidate filtering")
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
        'l11': 0.05,
        'l13': 0.1,
        'l14': 0.15,
        'X': 0.7,
    }
    
    # Optimizer
    num_epochs = 150
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.005)
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.002,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.25,
        anneal_strategy='cos',
        div_factor=20.0,
        final_div_factor=1000.0
    )
    
    best_val_acc = 0
    best_epoch = 0
    patience = 30
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_loss_X = train_epoch(
            model, train_loader, optimizer, device, loss_weights, epoch, scheduler, hierarchy_map
        )
        
        # Validate - USE FILTERING!
        model.eval()
        val_metrics = {'l11': {'correct@1': 0, 'correct@3': 0, 'correct@5': 0, 'correct@10': 0, 
                                'rr': 0, 'ndcg': 0, 'f1': 0, 'total': 0},
                       'l13': {'correct@1': 0, 'correct@3': 0, 'correct@5': 0, 'correct@10': 0, 
                                'rr': 0, 'ndcg': 0, 'f1': 0, 'total': 0},
                       'l14': {'correct@1': 0, 'correct@3': 0, 'correct@5': 0, 'correct@10': 0, 
                                'rr': 0, 'ndcg': 0, 'f1': 0, 'total': 0},
                       'X': {'correct@1': 0, 'correct@3': 0, 'correct@5': 0, 'correct@10': 0, 
                              'rr': 0, 'ndcg': 0, 'f1': 0, 'total': 0}}
        
        with torch.no_grad():
            for batch in val_loader:
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
                
                # Use hierarchical filtering for better accuracy!
                outputs = model(l11_seq, l13_seq, l14_seq, X_seq, user_seq, padding_mask,
                               hierarchy_map=hierarchy_map, use_filtering=True)
                
                # Evaluate (using existing metrics)
                from metrics import calculate_correct_total_prediction
                
                result, _, _ = calculate_correct_total_prediction(outputs['logits_X'], X_y)
                val_metrics['X']['correct@1'] += result[0]
                val_metrics['X']['correct@3'] += result[1]
                val_metrics['X']['correct@5'] += result[2]
                val_metrics['X']['correct@10'] += result[3]
                val_metrics['X']['rr'] += result[4]
                val_metrics['X']['ndcg'] += result[5]
                val_metrics['X']['total'] += result[6]
        
        from metrics import get_performance_dict
        val_perf_X = get_performance_dict(val_metrics['X'])
        val_acc_X = val_perf_X['acc@1']
        
        # Print
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} (X: {train_loss_X:.4f})")
        print(f"Val acc@1 (X): {val_acc_X:.2f}% | acc@5: {val_perf_X['acc@5']:.2f}% | MRR: {val_perf_X['mrr']:.2f}%")
        
        # Save best
        if val_acc_X > best_val_acc:
            best_val_acc = val_acc_X
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc_X,
                'config': config,
            }, 'best_model_advanced.pt')
            print(f"✓ NEW BEST: {val_acc_X:.2f}%")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
        
        if val_acc_X >= 50.0:
            print(f"\n🎉🎉🎉 TARGET ACHIEVED! {val_acc_X:.2f}% >= 50% 🎉🎉🎉")
            break
    
    # Test evaluation
    print(f"\n{'='*80}")
    print(f"Loading best model from epoch {best_epoch}")
    checkpoint = torch.load('best_model_advanced.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Evaluating on test set with hierarchical filtering...")
    test_perf = evaluate_model(model, test_loader, device, hierarchy_map, use_hierarchy=False)
    
    # ALSO evaluate WITH filtering
    print("Evaluating WITH hierarchical filtering...")
    model.eval()
    test_metrics_filtered = {'X': {'correct@1': 0, 'correct@3': 0, 'correct@5': 0, 'correct@10': 0, 
                                     'rr': 0, 'ndcg': 0, 'f1': 0, 'total': 0}}
    
    with torch.no_grad():
        for batch in test_loader:
            l11_seq = batch['l11_seq'].to(device)
            l13_seq = batch['l13_seq'].to(device)
            l14_seq = batch['l14_seq'].to(device)
            X_seq = batch['X_seq'].to(device)
            user_seq = batch['user_seq'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            X_y = batch['X_y'].to(device)
            
            outputs = model(l11_seq, l13_seq, l14_seq, X_seq, user_seq, padding_mask,
                           hierarchy_map=hierarchy_map, use_filtering=True)
            
            from metrics import calculate_correct_total_prediction
            result, _, _ = calculate_correct_total_prediction(outputs['logits_X'], X_y)
            test_metrics_filtered['X']['correct@1'] += result[0]
            test_metrics_filtered['X']['correct@3'] += result[1]
            test_metrics_filtered['X']['correct@5'] += result[2]
            test_metrics_filtered['X']['correct@10'] += result[3]
            test_metrics_filtered['X']['rr'] += result[4]
            test_metrics_filtered['X']['ndcg'] += result[5]
            test_metrics_filtered['X']['total'] += result[6]
    
    from metrics import get_performance_dict
    test_perf_filtered = get_performance_dict(test_metrics_filtered['X'])
    
    print(f"\n{'='*80}")
    print("FINAL TEST RESULTS - ADVANCED MODEL")
    print(f"{'='*80}")
    print(f"\nExact Location (X) - WITHOUT filtering:")
    print(f"  acc@1:  {test_perf['X']['acc@1']:.2f}%")
    print(f"  acc@5:  {test_perf['X']['acc@5']:.2f}%")
    print(f"  acc@10: {test_perf['X']['acc@10']:.2f}%")
    print(f"  MRR:    {test_perf['X']['mrr']:.2f}%")
    
    print(f"\nExact Location (X) - WITH hierarchical filtering:")
    print(f"  acc@1:  {test_perf_filtered['acc@1']:.2f}%")
    print(f"  acc@5:  {test_perf_filtered['acc@5']:.2f}%")
    print(f"  acc@10: {test_perf_filtered['acc@10']:.2f}%")
    print(f"  MRR:    {test_perf_filtered['mrr']:.2f}%")
    
    print(f"\n{'='*80}")
    print(f"Parameters: {num_params:,} / 700,000")
    print(f"Best Val acc@1 (X): {best_val_acc:.2f}%")
    print(f"Test acc@1 (X) without filtering: {test_perf['X']['acc@1']:.2f}%")
    print(f"Test acc@1 (X) WITH filtering: {test_perf_filtered['acc@1']:.2f}%")
    print(f"{'='*80}")
    
    if test_perf_filtered['acc@1'] >= 50.0:
        print(f"\n✅✅✅ SUCCESS! TARGET ACHIEVED! ✅✅✅")
    else:
        print(f"\nGap to 50%: {50.0 - test_perf_filtered['acc@1']:.2f}%")
    
    results = {
        'test_without_filtering': test_perf,
        'test_with_filtering': {'X': test_perf_filtered}
    }
    
    with open('test_results_advanced.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return test_perf_filtered['acc@1']


if __name__ == '__main__':
    final_acc = main()
