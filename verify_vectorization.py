#!/usr/bin/env python3
"""
Verify that vectorized operations produce identical results to the original
"""
import torch
import torch.nn.functional as F
import numpy as np


def test_frequency_vectorization():
    """Test that vectorized frequency computation matches the original"""
    print("Testing frequency vectorization...")
    
    B, T, vocab_size = 4, 10, 100
    X_seq = torch.randint(0, vocab_size, (B, T))
    padding_mask = torch.zeros(B, T, dtype=torch.bool)
    padding_mask[0, 8:] = True  # First sample has padding
    padding_mask[1, 9:] = True  # Second sample has padding
    
    # Original (loop-based)
    freq_dist_orig = torch.zeros(B, vocab_size)
    for b in range(B):
        for t in range(T):
            if not padding_mask[b, t]:
                loc = X_seq[b, t].item()
                freq_dist_orig[b, loc] += 1.0
    total_counts = freq_dist_orig.sum(dim=1, keepdim=True) + 1e-8
    freq_dist_orig = freq_dist_orig / total_counts
    
    # Vectorized
    one_hot = F.one_hot(X_seq, num_classes=vocab_size).float()
    mask = (~padding_mask).unsqueeze(-1).float()
    one_hot = one_hot * mask
    freq_dist_vec = one_hot.sum(dim=1)
    total_counts_vec = freq_dist_vec.sum(dim=1, keepdim=True) + 1e-8
    freq_dist_vec = freq_dist_vec / total_counts_vec
    
    # Compare
    diff = (freq_dist_orig - freq_dist_vec).abs().max().item()
    print(f"  Max difference: {diff:.2e}")
    assert diff < 1e-6, f"Frequency vectorization failed! Diff: {diff}"
    print("  ✓ Frequency vectorization correct!")


def test_recency_vectorization():
    """Test that vectorized recency computation matches the original"""
    print("\nTesting recency vectorization...")
    
    B, T, vocab_size = 4, 10, 100
    X_seq = torch.randint(0, vocab_size, (B, T))
    padding_mask = torch.zeros(B, T, dtype=torch.bool)
    padding_mask[0, 8:] = True
    padding_mask[1, 9:] = True
    
    decay = 0.9
    
    # Original (loop-based)
    recency_dist_orig = torch.zeros(B, vocab_size)
    for b in range(B):
        seq_len = int((~padding_mask[b]).sum().item())
        for t in range(seq_len):
            loc = X_seq[b, t].item()
            weight = decay ** (seq_len - t - 1)
            recency_dist_orig[b, loc] += weight
    total_recency = recency_dist_orig.sum(dim=1, keepdim=True) + 1e-8
    recency_dist_orig = recency_dist_orig / total_recency
    
    # Vectorized
    one_hot = F.one_hot(X_seq, num_classes=vocab_size).float()
    seq_lens = (~padding_mask).sum(dim=1).float()
    positions = torch.arange(T).float()
    decay_power = seq_lens.unsqueeze(1) - positions.unsqueeze(0) - 1
    recency_weights = torch.tensor(decay) ** decay_power
    recency_weights = recency_weights * (~padding_mask).float()
    weighted_one_hot = one_hot * recency_weights.unsqueeze(-1)
    recency_dist_vec = weighted_one_hot.sum(dim=1)
    total_recency_vec = recency_dist_vec.sum(dim=1, keepdim=True) + 1e-8
    recency_dist_vec = recency_dist_vec / total_recency_vec
    
    # Compare
    diff = (recency_dist_orig - recency_dist_vec).abs().max().item()
    print(f"  Max difference: {diff:.2e}")
    assert diff < 1e-6, f"Recency vectorization failed! Diff: {diff}"
    print("  ✓ Recency vectorization correct!")


def test_filtering_vectorization():
    """Test that vectorized filtering produces correct masks"""
    print("\nTesting filtering vectorization...")
    
    B, vocab_X = 4, 100
    logits = torch.randn(B, vocab_X)
    
    # Simulate hierarchy
    valid_sets = [
        [0, 5, 10, 15, 20],
        [3, 7, 11, 19, 25],
        [1, 2, 8, 14, 22],
        [4, 6, 9, 12, 18]
    ]
    
    # Original (loop-based with clone)
    logits_orig = logits.clone()
    for b in range(B):
        mask = torch.ones(vocab_X) * (-1e9)
        for x in valid_sets[b]:
            mask[x] = 0.0
        logits_orig[b] = logits_orig[b] + mask
    
    # Vectorized (using torch.where)
    mask = torch.zeros(B, vocab_X, dtype=torch.bool)
    for b in range(B):
        valid_X_tensor = torch.tensor(valid_sets[b], dtype=torch.long)
        mask[b, valid_X_tensor] = True
    logits_vec = torch.where(mask, logits, torch.tensor(-1e9))
    
    # Compare
    diff = (logits_orig - logits_vec).abs().max().item()
    print(f"  Max difference: {diff:.2e}")
    assert diff < 1e-6, f"Filtering vectorization failed! Diff: {diff}"
    print("  ✓ Filtering vectorization correct!")


if __name__ == '__main__':
    print("="*60)
    print("VECTORIZATION CORRECTNESS TESTS")
    print("="*60)
    
    test_frequency_vectorization()
    test_recency_vectorization()
    test_filtering_vectorization()
    
    print("\n" + "="*60)
    print("✓ ALL VECTORIZATION TESTS PASSED!")
    print("="*60)
