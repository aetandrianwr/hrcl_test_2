"""
Quick verification script - evaluates exact location (X) prediction only
"""
import torch
import numpy as np
import pickle
import json
from dataset import create_dataloaders
from hierarchical_s2_model_advanced import AdvancedHierarchicalS2Model
from metrics import calculate_correct_total_prediction, get_performance_dict
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_X_prediction(model, test_loader, hierarchy_map, use_filtering=False):
    """Evaluate exact location X prediction"""
    model.eval()
    
    all_metrics = {
        'correct@1': 0, 'correct@3': 0, 'correct@5': 0, 'correct@10': 0,
        'rr': 0, 'ndcg': 0, 'f1': 0, 'total': 0
    }
    
    with torch.no_grad():
        for batch in test_loader:
            l11_seq = batch['l11_seq'].to(device)
            l13_seq = batch['l13_seq'].to(device)
            l14_seq = batch['l14_seq'].to(device)
            X_seq = batch['X_seq'].to(device)
            user_seq = batch['user_seq'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            X_y = batch['X_y'].to(device)
            
            # Forward pass
            outputs = model(l11_seq, l13_seq, l14_seq, X_seq, user_seq,
                           padding_mask, hierarchy_map=hierarchy_map,
                           use_filtering=use_filtering)
            
            logits_X = outputs['logits_X']
            
            # Calculate metrics
            batch_metrics, _, _ = calculate_correct_total_prediction(logits_X, X_y)
            
            all_metrics['correct@1'] += batch_metrics[0]
            all_metrics['correct@3'] += batch_metrics[1]
            all_metrics['correct@5'] += batch_metrics[2]
            all_metrics['correct@10'] += batch_metrics[3]
            all_metrics['rr'] += batch_metrics[4]
            all_metrics['ndcg'] += batch_metrics[5]
            all_metrics['total'] += batch_metrics[6]
    
    return get_performance_dict(all_metrics)

def main():
    print("\n" + "="*80)
    print("  HIERARCHICAL S2 LOCATION PREDICTION - RESULTS VERIFICATION")
    print("="*80)
    print(f"\nDevice: {device}")
    
    # Load datasets
    print("Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=128, num_workers=0)
    print(f"✓ Test set: {len(test_loader.dataset)} samples\n")
    
    # Load hierarchy mapping
    with open('s2_hierarchy_mapping.pkl', 'rb') as f:
        hierarchy_map = pickle.load(f)
    
    results = {}
    
    # ========================================================================
    # ADVANCED MODEL
    # ========================================================================
    try:
        print("="*80)
        print("  ADVANCED MODEL (GRU-based with User Personalization)")
        print("="*80)
        
        # Load checkpoint
        checkpoint = torch.load('best_model_advanced.pt', map_location=device)
        config = checkpoint['config']
        
        print(f"\nConfiguration:")
        print(f"  d_model: {config['d_model']}")
        print(f"  nhead: {config['nhead']}")
        print(f"  dropout: {config['dropout']}")
        print(f"  num_users: {config['num_users']}")
        
        # Create model
        model = AdvancedHierarchicalS2Model(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        params = model.count_parameters()
        print(f"  Parameters: {params:,} / 700,000")
        print(f"  Budget used: {params/700000*100:.1f}%")
        
        # Evaluate WITHOUT filtering
        print(f"\nEvaluating WITHOUT hierarchical filtering...")
        perf_no_filter = evaluate_X_prediction(model, test_loader, hierarchy_map, use_filtering=False)
        
        print(f"\n{'─'*80}")
        print(f"  TEST RESULTS (Exact Location X, NO filtering)")
        print(f"{'─'*80}")
        print(f"  acc@1:  {perf_no_filter['acc@1']:6.2f}%")
        print(f"  acc@5:  {perf_no_filter['acc@5']:6.2f}%")
        print(f"  acc@10: {perf_no_filter['acc@10']:6.2f}%")
        print(f"  MRR:    {perf_no_filter['mrr']:6.2f}%")
        print(f"  NDCG:   {perf_no_filter['ndcg']:6.2f}%")
        print(f"{'─'*80}\n")
        
        results['advanced_no_filter'] = perf_no_filter
        
        # Evaluate WITH filtering
        print(f"Evaluating WITH hierarchical filtering...")
        perf_filter = evaluate_X_prediction(model, test_loader, hierarchy_map, use_filtering=True)
        
        print(f"\n{'─'*80}")
        print(f"  TEST RESULTS (Exact Location X, WITH filtering)")
        print(f"{'─'*80}")
        print(f"  acc@1:  {perf_filter['acc@1']:6.2f}%")
        print(f"  acc@5:  {perf_filter['acc@5']:6.2f}%")
        print(f"  acc@10: {perf_filter['acc@10']:6.2f}%")
        print(f"  MRR:    {perf_filter['mrr']:6.2f}%")
        print(f"  NDCG:   {perf_filter['ndcg']:6.2f}%")
        print(f"{'─'*80}\n")
        
        results['advanced_with_filter'] = perf_filter
        
    except FileNotFoundError:
        print("✗ best_model_advanced.pt not found!\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    if results:
        print("="*80)
        print("  FINAL SUMMARY")
        print("="*80)
        print(f"\n{'Configuration':<35} | {'acc@1':<8} | {'acc@5':<8} | {'MRR':<8}")
        print("─"*80)
        
        if 'advanced_no_filter' in results:
            r = results['advanced_no_filter']
            print(f"{'Advanced (WITHOUT filtering)':<35} | {r['acc@1']:>7.2f}% | {r['acc@5']:>7.2f}% | {r['mrr']:>7.2f}%")
        
        if 'advanced_with_filter' in results:
            r = results['advanced_with_filter']
            print(f"{'Advanced (WITH filtering)':<35} | {r['acc@1']:>7.2f}% | {r['acc@5']:>7.2f}% | {r['mrr']:>7.2f}%")
        
        print("\n" + "="*80)
        print("  EXPECTED vs ACTUAL")
        print("="*80)
        print(f"  Expected (no filter):  ~46.94% acc@1")
        if 'advanced_no_filter' in results:
            actual = results['advanced_no_filter']['acc@1']
            diff = actual - 46.94
            print(f"  Actual (no filter):    {actual:6.2f}% acc@1 (diff: {diff:+.2f}%)")
        
        print(f"\n  Expected (with filter): ~39.92% acc@1")
        if 'advanced_with_filter' in results:
            actual = results['advanced_with_filter']['acc@1']
            diff = actual - 39.92
            print(f"  Actual (with filter):  {actual:6.2f}% acc@1 (diff: {diff:+.2f}%)")
        print("="*80 + "\n")
        
        # Save results
        with open('verification_report.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("✓ Results saved to verification_report.json")
        print("✓ Verification complete!\n")
    else:
        print("✗ No results to display\n")

if __name__ == '__main__':
    main()
