# Hierarchical S2 Location Prediction

PyTorch implementation of next-location prediction using S2 hierarchical spatial indexing with user personalization and copy mechanism.

## 🚀 Quick Start

```bash
# Verify pre-trained model
conda activate mlenv
python reproduce_results.py
```

**Expected**: 46.94% acc@1 on exact location prediction

## 📊 Results Summary

| Model | Test acc@1 | Test acc@5 | MRR | Parameters |
|-------|-----------|-----------|-----|------------|
| **Advanced (no filter)** | **46.94%** | **81.30%** | **62.35%** | 620,808 / 700k |
| Advanced (with filter) | 39.92% | 46.92% | 43.36% | 620,808 / 700k |

## 🏗️ Architecture Highlights

- **User Personalization**: User embeddings + frequency/recency features
- **Copy Mechanism**: Multi-head attention over history with learnable gate
- **Hierarchical S2**: 4 levels (11→13→14→X) with soft embeddings
- **Vectorized**: GPU-accelerated with `scatter_add_` (~40% faster)
- **Parameter Efficient**: GRU encoder, 620k / 700k budget

## 📁 Key Files

- `train_advanced.py` - Train best model
- `reproduce_results.py` - Verify results (5 min)
- `hierarchical_s2_model_advanced.py` - Model architecture
- `FINAL_SUMMARY.md` - Comprehensive summary
- `REPRODUCTION_GUIDE.md` - Step-by-step reproduction

## 🎯 Training from Scratch

```bash
conda activate mlenv
python train_advanced.py --use_filtering=False
```

Time: ~60-90 minutes on GPU

## 📖 Documentation

1. **FINAL_SUMMARY.md** - Project overview, results, lessons learned
2. **REPRODUCTION_GUIDE.md** - Detailed reproduction instructions
3. **TECHNICAL_DOCUMENTATION.md** - Architecture details
4. **VECTORIZATION_COMPLETE.md** - Vectorization implementation

## ✅ Requirements

- CUDA-capable GPU
- PyTorch 1.13+
- Conda environment `mlenv`
- GeoLife dataset in `data/geolife/`

## 🎓 Key Findings

✓ **User personalization is critical** (+15% improvement)  
✓ **Copy mechanism outperforms pure generation** (+10-12%)  
✓ **Soft hierarchy > hard filtering** (avoids error propagation)  
✓ **Vectorization matters** (40% faster training)  

## 📈 Performance

- **Best acc@1**: 46.94% (gap to 50%: 3.06%)
- **Training time**: 3-4s/epoch (vectorized)
- **Parameters**: 620,808 / 700,000 (88.7% budget)
- **Random seed**: 42 (reproducible)

## 🔬 Dataset

- **Source**: GeoLife GPS trajectories
- **Users**: 50
- **Sequences**: 24,524 (70/15/15 split)
- **Locations**: 1,190 unique GPS points
- **S2 Levels**: 315 (L11), 675 (L13), 930 (L14)

## 🏆 Citation

```bibtex
@misc{hierarchical_s2_location_2024,
  title={Hierarchical S2 Location Prediction with User Personalization},
  year={2024}
}
```

## 📞 Support

For issues:
1. Check `REPRODUCTION_GUIDE.md`
2. Run `verify_vectorization.py`
3. Review `FINAL_SUMMARY.md`

---

**Result**: 46.94% acc@1 on next-location prediction ✓
