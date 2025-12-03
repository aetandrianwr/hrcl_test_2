# Model Comparison Summary

## All Models Developed

### 1. Model V1 (Baseline)
**File:** `hierarchical_s2_model.py`  
**Architecture:** Shared Transformer encoder  
**Parameters:** 596,970 / 700,000  

**Features:**
- Hierarchical embedding at all 4 levels  
- Shared transformer for parameter efficiency  
- Sequential hierarchical processing (L11→L13→L14→X)  
- Positional encoding  

**Test Results:**
- acc@1: **31.70%**
- acc@5: 55.08%
- acc@10: 58.48%
- MRR: 42.50%
- NDCG: 45.86%

---

### 2. Advanced Model (GRU-based)
**File:** `hierarchical_s2_model_advanced.py`  
**Architecture:** GRU encoder + Copy mechanism  
**Parameters:** 620,808 / 700,000  

**Features:**
- ✅ User embeddings for personalization  
- ✅ Copy mechanism with multi-head attention  
- ✅ Frequency features (learnable from history)  
- ✅ Recency features (exponential decay)  
- ✅ Copy-generate gating  
- ✅ GRU encoder (parameter efficient)  
- ✅ Optional hierarchical filtering  

**Test Results (WITHOUT filtering):**
- acc@1: **44.43%** (+12.73 vs V1)
- acc@5: 82.01%
- acc@10: 84.61%
- MRR: 61.33%
- NDCG: 67.03%

**Test Results (WITH filtering):**
- acc@1: **36.75%** (+5.05 vs V1)
- acc@5: 44.97%
- acc@10: 46.03%
- MRR: 40.91%
- NDCG: 42.04%

**Note:** Filtering reduces performance because imperfect L11 predictions eliminate valid candidates.

---

### 3. Advanced Model V2 (Transformer-based)
**File:** `hierarchical_s2_model_advanced_v2.py`  
**Architecture:** Transformer encoder + Copy mechanism  
**Parameters:** 641,772 / 700,000  
**Status:** 🔄 Currently training

**Features:**
Same as Advanced Model but:
- ❌ GRU encoder
- ✅ **Transformer encoder** (2 layers, 4 heads)  
- ✅ All other features identical (copy mechanism, user embeddings, etc.)

**Key Difference:**
- Uses Transformer instead of GRU for sequence encoding
- Slightly more parameters (641k vs 621k)
- d_model: 80 (vs 96 in GRU version, adjusted to fit budget)

**Training Progress (as of epoch 5):**
- Val acc@1: 18.81% (still training...)
- Expected to reach similar or slightly better results than GRU version

---

## Model Architecture Comparison

| Feature | V1 Baseline | Advanced (GRU) | Advanced V2 (Transformer) |
|---------|------------|----------------|---------------------------|
| **Sequence Encoder** | Shared Transformer | GRU | Transformer |
| **User Embeddings** | ❌ | ✅ | ✅ |
| **Copy Mechanism** | ❌ | ✅ | ✅ |
| **Frequency Features** | ❌ | ✅ | ✅ |
| **Recency Features** | ❌ | ✅ | ✅ |
| **d_model** | 80 | 96 | 80 |
| **Parameters** | 596,970 | 620,808 | 641,772 |
| **Test acc@1** | 31.70% | **44.43%** | TBD |
| **Improvement** | baseline | **+40.1%** | TBD |

---

## Why Advanced Models Perform Better

### Data Pattern Exploitation

1. **Copy Mechanism** (+15-20 points estimated)
   - 75.7% of targets appear in sequence history
   - Attention learns to weight recent positions higher
   - Directly addresses most common case

2. **User Personalization** (+5-8 points estimated)
   - Captures user-specific habits
   - Example: User 40 → Location 811 (339 times)

3. **Contextual Features** (+3-5 points estimated)
   - Frequency: learns popular locations per user
   - Recency: recent = more relevant

4. **Better Architecture** (+2-3 points estimated)
   - More parameters allocated to reasoning vs embeddings

### GRU vs Transformer for Sequence Encoding

**GRU Advantages:**
- ✅ More parameter efficient
- ✅ Better for long sequences with position-independent patterns
- ✅ Faster training (1.2-1.5 it/s)
- ✅ Achieved 44.43% test acc@1

**Transformer Advantages:**
- ✅ Better at modeling position-specific dependencies
- ✅ Parallel processing of sequences
- ✅ More powerful attention mechanism
- ❌ Slower training (~1.1-1.3 it/s)
- ❌ Slightly more parameters
- ⏳ Results TBD

---

## Training Scripts

| Model | Training Script | Status |
|-------|----------------|--------|
| V1 Baseline | `train.py` | ✓ Complete |
| Advanced (GRU) | `train_advanced.py` | ✓ Complete |
| Advanced V2 (Transformer) | `train_advanced_v2.py` | 🔄 Running |

---

## Quick Verification

```bash
# Verify V1 and Advanced (GRU)
conda activate mlenv
cd /data/hrcl_test_2
python reproduce_results.py --eval-only

# Train Advanced V2 (Transformer)
python train_advanced_v2.py
```

---

## Expected Final Results

Based on pattern mining and architecture analysis:

| Model | Expected acc@1 | Rationale |
|-------|----------------|-----------|
| V1 | 31.70% | ✓ Verified |
| Advanced (GRU) | 44.43% | ✓ Verified |
| Advanced V2 (Transformer) | 42-46% | Similar to GRU, depends on training |

**Hypothesis:** GRU might slightly outperform Transformer for this task because:
- Location revisitation is more about recency than absolute position
- GRU's sequential bias matches the temporal nature better
- Copy mechanism already handles attention, reducing Transformer's advantage

---

*Last updated: 2024-12-03*  
*Advanced V2 training in progress...*
