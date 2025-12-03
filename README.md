# Hierarchical S2 Next-Location Prediction Models - Reproduction Guide

## Quick Start

### Prerequisites
```bash
# Ensure you're in the mlenv conda environment
conda activate mlenv

# Verify CUDA is available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Verify Existing Results (Recommended)

The fastest way to verify all results:

```bash
cd /data/hrcl_test_2
python reproduce_results.py --eval-only
```

**Expected Output:**
```
Model V1 (Baseline):          31.70% test acc@1 ✓
Advanced (Without Filtering): 44.43% test acc@1 ✓
Advanced (With Filtering):    36.75% test acc@1 ✓

ALL RESULTS VERIFIED SUCCESSFULLY ✓
```

### Train From Scratch (Optional)

To retrain all models:

```bash
cd /data/hrcl_test_2
python reproduce_results.py --train
```

⏱️ Training time: ~2-3 hours on GPU

---

## Individual Model Training

### Train Model V1 (Baseline)

```bash
conda activate mlenv
cd /data/hrcl_test_2
python train.py
```

**Expected Result:** ~31.70% test acc@1

### Train Advanced Model

```bash
conda activate mlenv
cd /data/hrcl_test_2
python train_advanced.py
```

**Expected Results:**
- Without filtering: ~44.43% test acc@1
- With filtering: ~36.75% test acc@1

---

## Files Overview

| File | Description |
|------|-------------|
| `TECHNICAL_DOCUMENTATION.md` | Complete technical documentation |
| `reproduce_results.py` | Automated training and verification script |
| `hierarchical_s2_model.py` | Model V1 (Baseline) architecture |
| `hierarchical_s2_model_advanced.py` | Advanced model with copy mechanism |
| `dataset.py` | Data loading utilities |
| `metrics.py` | Evaluation metrics (required specification) |
| `train.py` | Training script for V1 |
| `train_advanced.py` | Training script for Advanced model |

---

## Results Summary

### Model Comparison

| Model | Architecture | acc@1 | acc@5 | MRR | Parameters |
|-------|-------------|-------|-------|-----|------------|
| **V1 Baseline** | Shared Transformer | 31.70% | 55.08% | 42.50% | 596,970 |
| **Advanced (no filter)** | GRU + Copy Attention | **44.43%** | **82.01%** | **61.33%** | 620,808 |
| **Advanced (with filter)** | + Hierarchical Masking | 36.75% | 44.97% | 40.91% | 620,808 |

### Key Improvements

The Advanced model achieves **+40.1% relative improvement** over baseline by exploiting:

1. **Copy Mechanism** - 75.7% of targets appear in history
2. **User Embeddings** - Strong user-location affinity
3. **Frequency & Recency Features** - Learnable context from history
4. **Multi-head Attention** - Better sequence understanding

---

## Verification

All results have been verified and match expected values (±1% tolerance):

✓ **V1:** 31.70% ± 1.0%  
✓ **Advanced (no filter):** 44.43% ± 1.0%  
✓ **Advanced (with filter):** 36.75% ± 1.0%

Report saved in: `verification_report.json`

---

## Troubleshooting

### Issue: CUDA not available
**Solution:** This project requires a GPU. Use a machine with CUDA support.

### Issue: Different random results
**Solution:** Random seed (42) is set automatically. Ensure you're using the same PyTorch version.

### Issue: Out of memory
**Solution:** Reduce batch size in dataloaders:
```python
# In reproduce_results.py or train*.py
train_loader, val_loader, test_loader = create_dataloaders(
    batch_size=32,  # Reduced from 64
    num_workers=4
)
```

---

## Documentation

For detailed technical documentation, see:
- **Technical Details:** `TECHNICAL_DOCUMENTATION.md`
- **Pattern Mining Analysis:** `PATTERN_MINING_RESULTS.md`
- **Summary:** `PROJECT_SUMMARY.md`

---

## Citation

```bibtex
@techreport{hierarchical_s2_patterns_2024,
  title={Hierarchical S2 Next-Location Prediction with Pattern-Based Copy Mechanisms},
  author={[Your Name]},
  year={2024},
  note={GeoLife Dataset, 700k parameter budget, 44.43\% test accuracy}
}
```

---

## Contact

For questions or issues:
1. Check `TECHNICAL_DOCUMENTATION.md`
2. Review training logs in `training_*.txt`
3. Verify `verification_report.json`

---

*Last verified: 2024-12-03*
