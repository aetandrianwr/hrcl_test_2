#!/bin/bash
# 
# FINAL RESULTS REPRODUCTION SCRIPT
# ==================================
# This script reproduces all the key results from the hierarchical S2 location prediction project
#
# Usage: ./reproduce_results.sh
#

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  HIERARCHICAL S2 LOCATION PREDICTION - RESULTS REPRODUCTION"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Activate conda environment
echo "→ Activating mlenv conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mlenv
echo "✓ Environment activated"
echo ""

# Verify dataset
echo "→ Verifying dataset..."
if [ ! -d "data/geolife" ]; then
    echo "✗ ERROR: data/geolife directory not found!"
    echo "  Please ensure the GeoLife dataset is in data/geolife/"
    exit 1
fi
echo "✓ Dataset found"
echo ""

# Verify hierarchy mapping
echo "→ Verifying S2 hierarchy mapping..."
if [ ! -f "s2_hierarchy_mapping.pkl" ]; then
    echo "✗ ERROR: s2_hierarchy_mapping.pkl not found!"
    echo "  This file should be generated during dataset preprocessing"
    exit 1
fi
echo "✓ Hierarchy mapping found"
echo ""

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  OPTION 1: QUICK VERIFICATION (Using Pre-trained Models)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "This option loads pre-trained models and evaluates them on the test set."
echo "Expected results:"
echo "  • Baseline V1: ~31.70% acc@1"
echo "  • Advanced (no filter): ~46.94% acc@1"
echo "  • Advanced (with filter): ~39.92% acc@1"
echo ""

read -p "Run quick verification? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Running verification..."
    python reproduce_results.py
    echo ""
    echo "✓ Quick verification complete!"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  OPTION 2: FULL TRAINING (Reproduces from scratch)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "This option trains all models from scratch. This will take time:"
echo "  • Baseline V1: ~30-40 minutes"
echo "  • Advanced: ~60-90 minutes"
echo "  • Advanced V2 (Transformer): ~90-120 minutes"
echo ""

read -p "Run full training? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    
    # Baseline V1
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo "  Training Baseline V1 Model"
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo ""
    echo "Expected: ~31.70% acc@1 on test set"
    echo "Parameters: <700k"
    echo ""
    python train.py 2>&1 | tee training_v1_reproduction.txt
    echo ""
    echo "✓ Baseline V1 training complete!"
    echo ""
    
    # Advanced (no filter)
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo "  Training Advanced Model (WITHOUT hierarchical filtering)"
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo ""
    echo "Expected: ~46.94% acc@1 on test set"
    echo "Parameters: 620,808 / 700,000"
    echo "Features:"
    echo "  ✓ User embeddings"
    echo "  ✓ Frequency & recency features"
    echo "  ✓ Multi-head copy attention"
    echo "  ✓ Copy-generate gate"
    echo ""
    python train_advanced.py --use_filtering=False 2>&1 | tee training_advanced_nofilter_reproduction.txt
    echo ""
    echo "✓ Advanced (no filter) training complete!"
    echo ""
    
    # Advanced (with filter)
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo "  Training Advanced Model (WITH hierarchical filtering)"
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo ""
    echo "Expected: ~39.92% acc@1 on test set"
    echo "Note: Filtering reduces accuracy due to propagation of coarse-level errors"
    echo ""
    python train_advanced.py --use_filtering=True 2>&1 | tee training_advanced_filter_reproduction.txt
    echo ""
    echo "✓ Advanced (with filter) training complete!"
    echo ""
    
    # Advanced V2 (Transformer)
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo "  Training Advanced V2 Model (Transformer encoder)"
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo ""
    echo "Expected: Similar to Advanced GRU variant"
    echo "Uses Transformer encoder instead of GRU"
    echo ""
    read -p "Train Advanced V2? (Takes longer) (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python train_advanced_v2.py --use_filtering=False 2>&1 | tee training_advanced_v2_reproduction.txt
        echo ""
        echo "✓ Advanced V2 training complete!"
    fi
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  VECTORIZATION VERIFICATION"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Verify that vectorized operations produce identical results..."
echo ""

read -p "Run vectorization tests? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python verify_vectorization.py
    echo ""
    echo "✓ Vectorization tests complete!"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  REPRODUCTION COMPLETE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Summary of Results:"
echo ""
echo "  Model                          | acc@1  | acc@5  | MRR    | Parameters"
echo "  ──────────────────────────────|─────---|─────---|────────|───────────"
echo "  Baseline V1                   | 31.70% | 75.36% | 50.57% | <700k"
echo "  Advanced (no filter)          | 46.94% | 81.30% | 62.35% | 620,808"
echo "  Advanced (with filter)        | 39.92% | 46.92% | 43.36% | 620,808"
echo ""
echo "Key Findings:"
echo "  • User personalization + copy mechanism improves acc@1 by +15.24%"
echo "  • Hierarchical filtering hurts accuracy (-7.02%) due to error propagation"
echo "  • Best model: Advanced without filtering (46.94% acc@1)"
echo "  • Gap to 50% target: 3.06%"
echo ""
echo "For detailed analysis, see:"
echo "  • TECHNICAL_DOCUMENTATION.md"
echo "  • VECTORIZATION_COMPLETE.md"
echo "  • FINAL_RESULTS_SUMMARY.md"
echo ""
