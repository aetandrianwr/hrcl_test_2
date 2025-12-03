# Technical Documentation: Hierarchical S2 Next-Location Prediction Models

## Table of Contents
1. [Overview](#overview)
2. [Model V1: Baseline](#model-v1-baseline)
3. [Model Advanced: Pattern-Based](#model-advanced-pattern-based)
4. [Reproduction Guide](#reproduction-guide)
5. [Results Verification](#results-verification)

---

## Overview

This documentation describes three next-location prediction models using S2 hierarchical spatial indexing on the GeoLife dataset. All models predict the next location across 4 hierarchical levels: S2 level 11, 13, 14, and exact location X.

**Constraint**: Maximum 700,000 trainable parameters  
**Random Seed**: 42 (for reproducibility)  
**Hardware**: CUDA-enabled GPU required  
**Environment**: mlenv conda environment

---

## Model V1: Baseline

### Architecture

**File**: `hierarchical_s2_model.py`

#### Core Design
```
Input: Sequences of S2 cells at 4 levels (L11, L13, L14, X)
Output: Predictions for next cell at each level

Components:
1. Embeddings (separate per level)
   - L11: 315 vocab → 32 dim
   - L13: 675 vocab → 32 dim  
   - L14: 930 vocab → 48 dim
   - X: 1190 vocab → 64 dim

2. Shared Transformer Encoder
   - d_model: 80
   - nhead: 4
   - num_layers: 2
   - dim_feedforward: 160
   - dropout: 0.2

3. Hierarchical Processing
   - Level 11: Encode L11 embeddings → predict next L11
   - Level 13: Concat(L11_hidden, L13_emb) → encode → predict L13
   - Level 14: Concat(L13_hidden, L14_emb) → encode → predict L14
   - Level X: Concat(L14_hidden, X_emb) → encode → predict X

4. Classification Heads
   - Shared pre-classifier: d_model → 64
   - Separate linear layers for each level
```

#### Parameters: 596,970 / 700,000

#### Loss Function
```python
Weighted sum of cross-entropy losses:
- L11: weight 0.1
- L13: weight 0.2
- L14: weight 0.2
- X:   weight 0.5  (primary objective)
```

#### Training Configuration
```yaml
Optimizer: AdamW
  lr: 0.001
  weight_decay: 0.01

Scheduler: OneCycleLR
  max_lr: 0.001
  pct_start: 0.3
  epochs: 100
  div_factor: 25.0

Batch size: 64
Early stopping: patience 20 epochs
```

#### Key Features
- ✅ Hierarchical embedding at all 4 levels
- ✅ Shared transformer for parameter efficiency
- ✅ Sequential hierarchical processing (L11→L13→L14→X)
- ✅ Positional encoding for temporal order
- ❌ No user embeddings
- ❌ No copy mechanism
- ❌ No attention over history

#### Performance
```
Test Results:
  acc@1:  31.70%
  acc@5:  55.08%
  acc@10: 58.48%
  MRR:    42.50%
  NDCG:   45.86%
```

---

## Model Advanced: Pattern-Based

### Architecture

**File**: `hierarchical_s2_model_advanced.py`

This model exploits discovered data patterns:
1. **75.7% of targets appear in sequence history** → Copy mechanism
2. **Strong user-location affinity** → User embeddings
3. **Recency & frequency matter** → Contextual features
4. **Hierarchical structure** → Optional candidate filtering

#### Core Design

```
Input: Sequences of S2 cells (L11, L13, L14, X) + User ID

Components:

1. USER EMBEDDING
   - 50 users → d_model (96) dimensions
   - Captures user-specific location preferences

2. HIERARCHICAL LOCATION EMBEDDINGS (compact)
   - L11: 315 vocab → 24 dim (d_model/4)
   - L13: 675 vocab → 24 dim (d_model/4)
   - L14: 930 vocab → 32 dim (d_model/3)
   - X: 1190 vocab → 48 dim (d_model/2)
   - Combined → d_model via projection

3. USER CONTEXTUAL FEATURES MODULE
   Extracts from history:
   
   a) Frequency Distribution
      - Count each location's occurrences
      - Normalize to probability distribution
      - Project to d_model
   
   b) Recency Distribution  
      - Exponential decay weighting: weight = decay^(seq_len - pos - 1)
      - Learnable decay parameter (init: 0.9)
      - Recent positions weighted higher
      - Project to d_model
   
   c) Combined Features
      - Concatenate frequency + recency embeddings
      - Linear projection to d_model

4. SEQUENCE ENCODER
   - GRU (not Transformer for efficiency)
   - Input: d_model
   - Hidden: d_model/2 (48)
   - Layers: 1
   - Bidirectional: False
   - Output projected back to d_model

5. HIERARCHICAL COPY ATTENTION
   - Multi-head attention (4 heads)
   - Query: final GRU state
   - Keys/Values: full GRU sequence output
   - Returns: 
     * Context vector (weighted history)
     * Attention weights per timestep

6. COPY DISTRIBUTION BUILDER
   For each batch:
   - Map attention weights to vocabulary
   - For each timestep t with weight w:
       copy_dist[location_at_t] += w
   - Add frequency boost: copy_dist += 0.3 * freq_dist
   - Add recency boost: copy_dist += 0.2 * recency_dist
   - Normalize to probability

7. GENERATION DISTRIBUTION
   - Combine: final_state + user_emb + user_context + attended_history
   - Feed through classifier
   - Standard softmax

8. COPY-GENERATE GATE
   - Input: [final_state, user_context, attended_history]
   - MLP: 3*d_model → d_model → 1
   - Sigmoid activation
   - Output: p_copy ∈ [0,1]

9. FINAL PREDICTION
   final_prob = p_copy * copy_dist + (1 - p_copy) * gen_dist

10. HIERARCHICAL CANDIDATE FILTERING (optional)
    If enabled:
    - Predict L11 from current state
    - Map L11 → valid L13 cells
    - Map L13 → valid L14 cells  
    - Map L14 → valid X locations
    - Mask out invalid X candidates
    - Only ~62 candidates on average (vs 1190)
```

#### Parameters: 620,808 / 700,000

#### Loss Function
```python
Weighted sum (same as V1 but higher X weight):
- L11: weight 0.05
- L13: weight 0.1
- L14: weight 0.15
- X:   weight 0.7  (primary objective)
```

#### Training Configuration
```yaml
Optimizer: AdamW
  lr: 0.002
  weight_decay: 0.005

Scheduler: OneCycleLR
  max_lr: 0.002
  pct_start: 0.25
  epochs: 150
  div_factor: 20.0
  final_div_factor: 1000.0

Batch size: 64
Early stopping: patience 30 epochs
Dropout: 0.25
```

#### Key Features
- ✅ User embeddings (personalization)
- ✅ Copy mechanism with attention
- ✅ Frequency features (learnable from history)
- ✅ Recency features (exponential decay)
- ✅ Multi-head attention over encoded history
- ✅ Copy-generate gating
- ✅ Optional hierarchical filtering
- ✅ GRU instead of Transformer (parameter efficient)

#### Performance

**Without Hierarchical Filtering** (recommended):
```
Test Results:
  acc@1:  44.43%  (+12.73 vs baseline)
  acc@5:  82.01%  (+26.93 vs baseline)
  acc@10: 84.61%  (+26.13 vs baseline)
  MRR:    61.33%  (+18.83 vs baseline)
  NDCG:   67.03%  (+21.17 vs baseline)
```

**With Hierarchical Filtering**:
```
Test Results:
  acc@1:  36.75%  (+5.05 vs baseline)
  acc@5:  44.97%
  acc@10: 46.03%
  MRR:    40.91%
  NDCG:   42.04%
```

**Note**: Filtering hurts performance because imperfect L11 predictions (52% acc) eliminate valid candidates.

---

## Reproduction Guide

### Prerequisites

```bash
# Activate conda environment
conda activate mlenv

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Verify data
ls data/geolife/geolife_transformer_7_*.pk
```

### Directory Structure
```
/data/hrcl_test_2/
├── data/
│   └── geolife/
│       ├── geolife_transformer_7_train.pk
│       ├── geolife_transformer_7_validation.pk
│       └── geolife_transformer_7_test.pk
├── hierarchical_s2_model.py              # V1 model
├── hierarchical_s2_model_advanced.py     # Advanced model
├── dataset.py                             # Data loading
├── metrics.py                             # Evaluation metrics
├── train.py                               # V1 training script
├── train_advanced.py                      # Advanced training script
├── reproduce_results.py                   # Automated reproduction
└── s2_hierarchy_mapping.pkl              # Hierarchical relationships
```

### Step-by-Step Reproduction

#### 1. Train Model V1 (Baseline)

```bash
cd /data/hrcl_test_2
conda activate mlenv

# Train baseline model
python train.py

# Expected output:
# - best_model.pt (saved checkpoint)
# - test_results.json
# - Test acc@1 (X): ~31.70%
```

Training time: ~30-45 minutes on GPU

#### 2. Train Advanced Model (Without Filtering)

```bash
cd /data/hrcl_test_2
conda activate mlenv

# Train advanced model
python train_advanced.py

# Expected output:
# - best_model_advanced.pt
# - test_results_advanced.json
# - Test acc@1 (X): ~44.43% (without filtering)
# - Test acc@1 (X): ~36.75% (with filtering)
```

Training time: ~45-60 minutes on GPU

#### 3. Evaluate Saved Models

```bash
# Evaluate any saved model
python << 'EOF'
import torch
from hierarchical_s2_model import HierarchicalS2Model
from hierarchical_s2_model_advanced import AdvancedHierarchicalS2Model
from dataset import create_dataloaders
from metrics import evaluate_model
import pickle

device = torch.device('cuda')

# Load hierarchy
with open('s2_hierarchy_mapping.pkl', 'rb') as f:
    hierarchy_map = pickle.load(f)

# Load test data
_, _, test_loader = create_dataloaders(batch_size=64, num_workers=4)

# Evaluate V1
checkpoint_v1 = torch.load('best_model.pt')
model_v1 = HierarchicalS2Model(checkpoint_v1['config']).to(device)
model_v1.load_state_dict(checkpoint_v1['model_state_dict'])
results_v1 = evaluate_model(model_v1, test_loader, device, hierarchy_map)
print(f"V1 Test acc@1: {results_v1['X']['acc@1']:.2f}%")

# Evaluate Advanced
checkpoint_adv = torch.load('best_model_advanced.pt')
model_adv = AdvancedHierarchicalS2Model(checkpoint_adv['config']).to(device)
model_adv.load_state_dict(checkpoint_adv['model_state_dict'])
results_adv = evaluate_model(model_adv, test_loader, device, hierarchy_map)
print(f"Advanced Test acc@1: {results_adv['X']['acc@1']:.2f}%")
EOF
```

---

## Results Verification

### Automated Verification Script

A comprehensive script `reproduce_results.py` is provided to verify all results:

```bash
cd /data/hrcl_test_2
conda activate mlenv
python reproduce_results.py
```

This script:
1. ✅ Verifies random seed (42)
2. ✅ Checks parameter counts
3. ✅ Loads saved models
4. ✅ Evaluates on test set
5. ✅ Compares with expected results
6. ✅ Generates verification report

### Expected Output

```
================================================================================
MODEL VERIFICATION REPORT
================================================================================

Model V1 (Baseline):
  Parameters: 596,970 / 700,000 ✓
  Test acc@1: 31.70% ✓
  Status: VERIFIED

Model Advanced (Without Filtering):
  Parameters: 620,808 / 700,000 ✓
  Test acc@1: 44.43% ✓
  Improvement: +12.73 points ✓
  Status: VERIFIED

Model Advanced (With Filtering):
  Test acc@1: 36.75% ✓
  Improvement: +5.05 points ✓
  Status: VERIFIED

================================================================================
ALL RESULTS VERIFIED SUCCESSFULLY ✓
================================================================================
```

### Key Differences Summary

| Feature | V1 Baseline | Advanced |
|---------|-------------|----------|
| Architecture | Shared Transformer | GRU + Copy Attention |
| User Embeddings | ❌ | ✅ |
| Copy Mechanism | ❌ | ✅ Multi-head attention |
| Frequency Features | ❌ | ✅ Learnable |
| Recency Features | ❌ | ✅ Exponential decay |
| Parameters | 596,970 | 620,808 |
| **Test acc@1** | **31.70%** | **44.43%** |
| **Improvement** | baseline | **+40.1%** |

### Why Advanced Model Performs Better

1. **Copy Mechanism** (+15-20 points estimated)
   - Exploits 75.7% revisitation pattern
   - Attention learns to weight recent positions higher
   - Directly addresses most common case

2. **User Personalization** (+5-8 points estimated)
   - Captures user-specific habits
   - Example: User 40 → Location 811 (339 times)

3. **Contextual Features** (+3-5 points estimated)
   - Frequency: learns popular locations per user
   - Recency: recent = more relevant

4. **Better Architecture** (+2-3 points estimated)
   - GRU more efficient than Transformer for this task
   - More parameters allocated to reasoning vs embeddings

---

## Troubleshooting

### Issue: Different results
**Solution**: Ensure random seed 42 is set everywhere:
```python
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
```

### Issue: Out of memory
**Solution**: Reduce batch size in data loader:
```python
train_loader, val_loader, test_loader = create_dataloaders(
    batch_size=32,  # Reduced from 64
    num_workers=4
)
```

### Issue: No GPU
**Solution**: Models require CUDA. Use a GPU instance or cloud service.

---

## Citation

If you use these models, please cite:

```bibtex
@techreport{hierarchical_s2_2024,
  title={Hierarchical S2 Next-Location Prediction with Pattern-Based Copy Mechanisms},
  author={[Your Name]},
  year={2024},
  note={GeoLife Dataset, 700k parameter budget}
}
```

---

## Files Reference

| File | Purpose | Lines |
|------|---------|-------|
| `hierarchical_s2_model.py` | V1 baseline model | 250 |
| `hierarchical_s2_model_advanced.py` | Advanced pattern-based model | 346 |
| `dataset.py` | Data loading and batching | 100 |
| `metrics.py` | Evaluation metrics | 150 |
| `train.py` | V1 training script | 200 |
| `train_advanced.py` | Advanced training script | 320 |
| `reproduce_results.py` | Automated verification | 150 |

---

## Contact & Support

For questions or issues:
1. Check this documentation
2. Review training logs (`training_log.txt`, `training_advanced_log.txt`)
3. Verify data integrity
4. Ensure CUDA and mlenv environment are properly configured

---

*Last updated: 2024-12-03*
