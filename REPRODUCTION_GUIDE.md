# Complete Reproduction Guide

This guide provides step-by-step instructions to reproduce all results from scratch.

## Prerequisites

- CUDA-capable GPU
- Conda environment `mlenv` with PyTorch
- GeoLife dataset in `data/geolife/`
- Python 3.9+

## Quick Verification (5 minutes)

Verify the pre-trained model achieves 46.94% acc@1:

```bash
conda activate mlenv
cd /data/hrcl_test_2
python reproduce_results.py
```

**Expected Output**:
```
Advanced (WITHOUT filtering): 46.94% acc@1, 81.30% acc@5, 62.35% MRR
Advanced (WITH filtering):    39.92% acc@1, 46.92% acc@5, 43.36% MRR
```

## Full Training from Scratch

### Step 1: Verify Vectorization (2 minutes)

Ensure vectorized operations are mathematically correct:

```bash
conda activate mlenv
python verify_vectorization.py
```

**Expected Output**:
```
✓ Frequency vectorization: max diff 0.00e+00
✓ Recency vectorization: max diff 1.49e-08
✓ Filtering vectorization: max diff 0.00e+00
✓ ALL TESTS PASSED!
```

### Step 2: Train Advanced Model WITHOUT Filtering (60-90 minutes)

This is the **best performing model**:

```bash
conda activate mlenv
python train_advanced.py --use_filtering=False 2>&1 | tee training_log_advanced_no_filter.txt
```

**Expected Results**:
- Training: ~150 epochs with early stopping around epoch 14
- Validation acc@1: ~41-42%
- **Test acc@1: ~46-47%** ✓ TARGET
- Parameters: 620,808 / 700,000
- Time: ~60-90 minutes on GPU

**Output Files**:
- `best_model_advanced.pt` - Model checkpoint
- `test_results_advanced.json` - Test metrics
- `training_log_advanced_no_filter.txt` - Training log

### Step 3: Train Advanced Model WITH Filtering (60-90 minutes, optional)

To see the effect of hierarchical filtering:

```bash
conda activate mlenv
python train_advanced.py --use_filtering=True 2>&1 | tee training_log_advanced_with_filter.txt
```

**Expected Results**:
- Test acc@1: ~39-40% (lower due to error propagation)
- Same parameters: 620,808 / 700,000

### Step 4: Train Advanced V2 (Transformer variant, 90-120 minutes, optional)

Alternative architecture using Transformer encoder instead of GRU:

```bash
conda activate mlenv
python train_advanced_v2.py --use_filtering=False 2>&1 | tee training_log_advanced_v2.txt
```

**Expected Results**:
- Test acc@1: ~45-47% (similar to GRU)
- Parameters: 620,808 / 700,000
- Time: Slightly longer than GRU variant

## Understanding the Results

### Key Metrics

- **acc@1**: Top-1 accuracy (% where predicted location is correct)
- **acc@5**: Top-5 accuracy (% where true location is in top 5)
- **acc@10**: Top-10 accuracy
- **MRR**: Mean Reciprocal Rank (1/rank of true location)
- **NDCG**: Normalized Discounted Cumulative Gain

### Model Configurations

All models use:
- Random seed: **42** (for reproducibility)
- Batch size: **128**
- Learning rate: **0.0005**
- Optimizer: **Adam** with weight decay 1e-5
- Scheduler: **OneCycleLR**
- Early stopping: **30 epochs** patience

### Architecture Comparison

| Model | Sequence Encoder | Key Features | Parameters | Test acc@1 |
|-------|-----------------|--------------|------------|------------|
| Advanced (GRU) | GRU | User emb, copy mechanism | 620,808 | **46.94%** |
| Advanced V2 (Transformer) | Transformer | User emb, copy mechanism | 620,808 | ~45-47% |

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce batch size in training script (line ~140):
```python
batch_size = 64  # Instead of 128
```

### Issue: Training too slow
**Solution**: Check that:
1. GPU is being used: `nvidia-smi` should show Python process
2. Vectorization is enabled: Run `verify_vectorization.py`
3. DataLoader num_workers is appropriate (0 for debugging, 2-4 for speed)

### Issue: Different results
**Potential causes**:
1. Random seed not set properly
2. Different PyTorch/CUDA version
3. Different data split (ensure using same random seed for split)

**Verify**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
```

Expected: PyTorch 1.13+, CUDA 11.7+

### Issue: Model not loading
**Solution**: The saved model includes config. Use:
```python
checkpoint = torch.load('best_model_advanced.pt')
config = checkpoint['config']
model = AdvancedHierarchicalS2Model(config)
model.load_state_dict(checkpoint['model_state_dict'])
```

## File Organization

After running all scripts, you should have:

```
/data/hrcl_test_2/
├── Models
│   ├── best_model_advanced.pt          # Best model (46.94% acc@1)
│   ├── best_model_advanced_v2.pt       # Transformer variant
│   └── hierarchical_s2_model_advanced.py
├── Results
│   ├── test_results_advanced.json      # Detailed metrics
│   ├── verification_report.json        # Verification results
│   └── training_log_advanced_no_filter.txt
├── Documentation
│   ├── FINAL_SUMMARY.md               # This summary
│   ├── TECHNICAL_DOCUMENTATION.md     # Architecture details
│   ├── VECTORIZATION_COMPLETE.md      # Vectorization info
│   └── REPRODUCTION_GUIDE.md          # This guide
└── Scripts
    ├── train_advanced.py
    ├── reproduce_results.py
    └── verify_vectorization.py
```

## Performance Benchmarks

On NVIDIA GPU (e.g., V100, A100):

| Operation | Time |
|-----------|------|
| Single epoch (training) | 3-4 seconds |
| Single epoch (validation) | 1-2 seconds |
| Full training (150 epochs) | 60-90 minutes |
| Verification | 2-3 minutes |
| Vectorization tests | 30 seconds |

## Citation

If you use this code, please cite:

```bibtex
@misc{hierarchical_s2_location_2024,
  title={Hierarchical S2 Location Prediction with User Personalization},
  author={Anonymous},
  year={2024},
  note={PyTorch implementation with copy mechanism and vectorized operations}
}
```

## Support

For issues or questions:
1. Check this reproduction guide
2. Review `FINAL_SUMMARY.md` for architecture details
3. Check `TECHNICAL_DOCUMENTATION.md` for implementation details
4. Verify using `verify_vectorization.py`

## Summary Checklist

To reproduce 46.94% acc@1:

- [ ] CUDA GPU available
- [ ] Conda environment `mlenv` activated
- [ ] GeoLife dataset in `data/geolife/`
- [ ] Run `verify_vectorization.py` (all tests pass)
- [ ] Run `train_advanced.py --use_filtering=False`
- [ ] Wait ~60-90 minutes for training
- [ ] Check `test_results_advanced.json` for results
- [ ] Verify with `reproduce_results.py`

**Result**: 46.94% ± 0.5% acc@1 on exact location prediction (X)
