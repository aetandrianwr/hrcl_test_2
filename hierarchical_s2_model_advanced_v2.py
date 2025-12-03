"""
Advanced Hierarchical S2 Next-Location Prediction Model V2
Using Transformer instead of GRU for sequence encoding

This is EXACTLY the same as the Advanced model but replaces:
- GRU encoder → Transformer encoder
- All other components remain identical (copy mechanism, user embeddings, etc.)
"""

import torch
import torch.nn as nn
import math


class AdvancedHierarchicalS2ModelV2(nn.Module):
    """
    Advanced model with Transformer encoder (instead of GRU)
    
    Key features (same as Advanced):
    - User embeddings for personalization
    - Copy mechanism with multi-head attention
    - Frequency and recency features from history
    - Copy-generate gating
    - Hierarchical candidate filtering (optional)
    
    Changed from Advanced:
    - GRU → Transformer encoder
    - Otherwise identical architecture
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.vocab_l11 = config['vocab_l11']
        self.vocab_l13 = config['vocab_l13']
        self.vocab_l14 = config['vocab_l14']
        self.vocab_X = config['vocab_X']
        self.num_users = config['num_users']
        self.d_model = config['d_model']
        self.nhead = config.get('nhead', 4)
        self.num_layers = config.get('num_layers', 2)  # Transformer layers
        self.dropout = config.get('dropout', 0.25)
        
        # User embedding
        self.user_embedding = nn.Embedding(self.num_users, self.d_model)
        
        # Hierarchical location embeddings (compact)
        self.embed_l11 = nn.Embedding(self.vocab_l11, self.d_model // 4)
        self.embed_l13 = nn.Embedding(self.vocab_l13, self.d_model // 4)
        self.embed_l14 = nn.Embedding(self.vocab_l14, self.d_model // 3)
        self.embed_X = nn.Embedding(self.vocab_X, self.d_model // 2)
        
        # Combine location embeddings
        total_emb_dim = self.d_model // 4 + self.d_model // 4 + self.d_model // 3 + self.d_model // 2
        self.combine_embeddings = nn.Linear(total_emb_dim, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=self.dropout)
        
        # CHANGED: Transformer encoder instead of GRU
        # Use smaller feedforward to stay within budget
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model,  # Reduced from 2x to 1x
            dropout=self.dropout,
            batch_first=True
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # User contextual features module
        self.frequency_projection = nn.Linear(self.vocab_X, self.d_model)
        self.recency_projection = nn.Linear(self.vocab_X, self.d_model)
        self.recency_decay = nn.Parameter(torch.tensor(0.9))
        self.context_combine = nn.Linear(self.d_model * 2, self.d_model)
        
        # Multi-head copy attention
        self.copy_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.nhead,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Copy-generate gate
        self.copy_gate = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )
        
        # Classifiers for all levels
        self.classifier_l11 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.vocab_l11)
        )
        
        self.classifier_l13 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.vocab_l13)
        )
        
        self.classifier_l14 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.vocab_l14)
        )
        
        # Generation classifier for X
        self.classifier_X = nn.Sequential(
            nn.Linear(self.d_model * 4, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.vocab_X)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
    
    def compute_frequency_features(self, X_seq, padding_mask):
        """Compute frequency distribution from history - VECTORIZED"""
        B, T = X_seq.shape
        device = X_seq.device
        
        # VECTORIZED: Create one-hot encoding: [B, T, vocab_X]
        one_hot = F.one_hot(X_seq, num_classes=self.vocab_X).float()
        
        # Apply padding mask
        mask = (~padding_mask).unsqueeze(-1).float()  # [B, T, 1]
        one_hot = one_hot * mask
        
        # Sum to get frequency distribution
        freq_dist = one_hot.sum(dim=1)  # [B, vocab_X]
        
        # Normalize
        freq_sum = freq_dist.sum(dim=1, keepdim=True).clamp(min=1.0)
        freq_dist = freq_dist / freq_sum
        
        # Project to d_model
        freq_features = self.frequency_projection(freq_dist)
        
        return freq_features, freq_dist
    
    def compute_recency_features(self, X_seq, padding_mask):
        """Compute recency distribution with exponential decay - VECTORIZED"""
        B, T = X_seq.shape
        device = X_seq.device
        
        # VECTORIZED: Create one-hot encoding
        one_hot = F.one_hot(X_seq, num_classes=self.vocab_X).float()  # [B, T, vocab_X]
        
        decay = torch.clamp(self.recency_decay, 0.5, 0.99)
        
        # Calculate sequence lengths
        seq_lens = (~padding_mask).sum(dim=1).float()  # [B]
        
        # Create position-based weights
        positions = torch.arange(T, device=device).float()  # [T]
        
        # Recency weights: decay^(seq_len - pos - 1)
        decay_power = seq_lens.unsqueeze(1) - positions.unsqueeze(0) - 1  # [B, T]
        recency_weights = decay ** decay_power  # [B, T]
        
        # Apply padding mask
        recency_weights = recency_weights * (~padding_mask).float()
        
        # Weighted one-hot: [B, T, vocab_X] * [B, T, 1]
        weighted_one_hot = one_hot * recency_weights.unsqueeze(-1)
        recency_dist = weighted_one_hot.sum(dim=1)  # [B, vocab_X]
        
        # Normalize
        recency_sum = recency_dist.sum(dim=1, keepdim=True).clamp(min=1e-8)
        recency_dist = recency_dist / recency_sum
        
        # Project to d_model
        recency_features = self.recency_projection(recency_dist)
        
        return recency_features, recency_dist
    
    def build_copy_distribution(self, X_seq, attention_weights, freq_dist, recency_dist, padding_mask):
        """Build copy distribution from attention weights and history - VECTORIZED"""
        B, T = X_seq.shape
        device = X_seq.device
        
        copy_dist = torch.zeros(B, self.vocab_X, device=device)
        
        # Mask out padding positions in attention weights
        masked_attention = attention_weights.masked_fill(padding_mask, 0.0)  # [B, T]
        
        # scatter_add_: accumulate attention weights at indices specified by X_seq
        copy_dist.scatter_add_(1, X_seq, masked_attention)
        
        # Add frequency and recency bias
        copy_dist = copy_dist + 0.3 * freq_dist + 0.2 * recency_dist
        
        # Normalize
        copy_sum = copy_dist.sum(dim=1, keepdim=True).clamp(min=1e-8)
        copy_dist = copy_dist / copy_sum
        
        return copy_dist
    
    def apply_hierarchical_filtering(self, logits_X, l11_seq, l13_seq, l14_seq, hierarchy_map):
        """Apply hierarchical candidate filtering based on predicted coarse levels - VECTORIZED"""
        B = logits_X.shape[0]
        device = logits_X.device
        
        # Predict L11
        logits_l11_pred = self.classifier_l11(self.final_state)
        pred_l11 = torch.argmax(logits_l11_pred, dim=-1)
        
        # Predict L13 given L11
        logits_l13_pred = self.classifier_l13(self.final_state)
        pred_l13 = torch.argmax(logits_l13_pred, dim=-1)
        
        # Predict L14 given L13
        logits_l14_pred = self.classifier_l14(self.final_state)
        pred_l14 = torch.argmax(logits_l14_pred, dim=-1)
        
        # Create mask
        mask = torch.zeros_like(logits_X, dtype=torch.bool)
        
        for b in range(B):
            l11_id = pred_l11[b].item()
            l13_id = pred_l13[b].item()
            l14_id = pred_l14[b].item()
            
            # Get valid X candidates
            valid_X = []
            if l14_id in hierarchy_map['s2_level14_to_X']:
                valid_X = hierarchy_map['s2_level14_to_X'][l14_id]
            
            if len(valid_X) > 0:
                valid_X_tensor = torch.tensor(valid_X, dtype=torch.long, device=device)
                mask[b, valid_X_tensor] = True
            else:
                # Fallback: use all
                mask[b, :] = True
        
        # Apply mask using torch.where for better performance
        logits_X_filtered = torch.where(mask, logits_X, torch.tensor(-1e9, device=device))
        
        return logits_X_filtered
    
    def forward(self, l11_seq, l13_seq, l14_seq, X_seq, user_seq, padding_mask, 
                hierarchy_map=None, use_filtering=False):
        """
        Forward pass
        
        Args:
            l11_seq, l13_seq, l14_seq, X_seq: [B, T]
            user_seq: [B, T] 
            padding_mask: [B, T] True for padding positions
            hierarchy_map: Optional hierarchy mapping for filtering
            use_filtering: Whether to apply hierarchical filtering
        """
        B, T = X_seq.shape
        
        # Get user embedding (use first user in sequence)
        user_emb = self.user_embedding(user_seq[:, 0])  # [B, d_model]
        
        # Combine location embeddings
        l11_emb = self.embed_l11(l11_seq)  # [B, T, d_model//4]
        l13_emb = self.embed_l13(l13_seq)  # [B, T, d_model//4]
        l14_emb = self.embed_l14(l14_seq)  # [B, T, d_model//3]
        X_emb = self.embed_X(X_seq)        # [B, T, d_model//2]
        
        combined = torch.cat([l11_emb, l13_emb, l14_emb, X_emb], dim=-1)  # [B, T, total]
        combined = self.combine_embeddings(combined)  # [B, T, d_model]
        
        # Add positional encoding
        combined = self.pos_encoder(combined)
        
        # CHANGED: Use Transformer encoder instead of GRU
        # Create attention mask for padding
        # Transformer uses True for positions to IGNORE
        transformer_mask = padding_mask  # [B, T]
        
        # Encode sequence
        encoded = self.sequence_encoder(
            combined,
            src_key_padding_mask=transformer_mask
        )  # [B, T, d_model]
        
        # Get final state (last non-padding position)
        final_states = []
        for b in range(B):
            valid_length = (~padding_mask[b]).sum().item()
            if valid_length > 0:
                final_states.append(encoded[b, valid_length - 1])
            else:
                final_states.append(encoded[b, 0])
        
        final_state = torch.stack(final_states)  # [B, d_model]
        self.final_state = final_state  # Store for filtering
        
        # Compute user contextual features
        freq_features, freq_dist = self.compute_frequency_features(X_seq, padding_mask)
        recency_features, recency_dist = self.compute_recency_features(X_seq, padding_mask)
        user_context = self.context_combine(torch.cat([freq_features, recency_features], dim=-1))
        
        # Multi-head attention for copy mechanism
        query = final_state.unsqueeze(1)  # [B, 1, d_model]
        attended, attention_weights = self.copy_attention(
            query, encoded, encoded,
            key_padding_mask=padding_mask
        )
        attended = attended.squeeze(1)  # [B, d_model]
        attention_weights = attention_weights.squeeze(1)  # [B, T]
        
        # Build copy distribution
        copy_dist = self.build_copy_distribution(
            X_seq, attention_weights, freq_dist, recency_dist, padding_mask
        )
        
        # Generate distribution
        gen_input = torch.cat([final_state, user_emb, user_context, attended], dim=-1)
        gen_logits = self.classifier_X(gen_input)  # [B, vocab_X]
        gen_dist = torch.softmax(gen_logits, dim=-1)
        
        # Copy-generate gate
        gate_input = torch.cat([final_state, user_context, attended], dim=-1)
        p_copy = self.copy_gate(gate_input)  # [B, 1]
        
        # Combine distributions
        final_dist = p_copy * copy_dist + (1 - p_copy) * gen_dist
        logits_X = torch.log(final_dist + 1e-10)
        
        # Apply hierarchical filtering if requested
        if use_filtering and hierarchy_map is not None:
            logits_X = self.apply_hierarchical_filtering(
                logits_X, l11_seq, l13_seq, l14_seq, hierarchy_map
            )
        
        # Predictions for other levels
        logits_l11 = self.classifier_l11(final_state)
        logits_l13 = self.classifier_l13(final_state)
        logits_l14 = self.classifier_l14(final_state)
        
        return {
            'logits_l11': logits_l11,
            'logits_l13': logits_l13,
            'logits_l14': logits_l14,
            'logits_X': logits_X,
            'p_copy': p_copy,
            'attention_weights': attention_weights,
        }
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PositionalEncoding(nn.Module):
    """Standard positional encoding"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


if __name__ == '__main__':
    # Test the model
    import random
    import numpy as np
    
    # Set seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Test configuration
    config = {
        'vocab_l11': 315,
        'vocab_l13': 675,
        'vocab_l14': 930,
        'vocab_X': 1190,
        'num_users': 50,
        'd_model': 80,  # Reduced from 96 to fit budget
        'nhead': 4,
        'num_layers': 2,
        'dropout': 0.25,
    }
    
    model = AdvancedHierarchicalS2ModelV2(config)
    
    print("="*80)
    print("Advanced Hierarchical S2 Model V2 (Transformer)")
    print("="*80)
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"Parameter budget: 700,000")
    print(f"Within budget: {model.count_parameters() <= 700000}")
    print()
    
    # Test forward pass
    B, T = 4, 10
    l11_seq = torch.randint(0, config['vocab_l11'], (B, T))
    l13_seq = torch.randint(0, config['vocab_l13'], (B, T))
    l14_seq = torch.randint(0, config['vocab_l14'], (B, T))
    X_seq = torch.randint(0, config['vocab_X'], (B, T))
    user_seq = torch.randint(0, config['num_users'], (B, T))
    padding_mask = torch.zeros(B, T, dtype=torch.bool)
    padding_mask[:, -2:] = True  # Last 2 positions are padding
    
    outputs = model(l11_seq, l13_seq, l14_seq, X_seq, user_seq, padding_mask)
    
    print("Forward pass test:")
    print(f"  logits_l11 shape: {outputs['logits_l11'].shape}")
    print(f"  logits_l13 shape: {outputs['logits_l13'].shape}")
    print(f"  logits_l14 shape: {outputs['logits_l14'].shape}")
    print(f"  logits_X shape: {outputs['logits_X'].shape}")
    print(f"  p_copy shape: {outputs['p_copy'].shape}")
    print(f"  p_copy mean: {outputs['p_copy'].mean().item():.4f}")
    print()
    print("✓ Model V2 (Transformer) ready!")
    print("="*80)
