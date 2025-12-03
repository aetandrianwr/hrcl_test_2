import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class UserContextualFeatures(nn.Module):
    """
    Extract user-specific contextual features from history:
    - Frequency: how often each location appears
    - Recency: weighted by position (recent = higher weight)
    - Radius of gyration: spatial dispersion
    - Entropy: predictability of user behavior
    """
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Learnable feature extractors
        self.freq_proj = nn.Linear(vocab_size, d_model)
        self.recency_proj = nn.Linear(vocab_size, d_model)
        self.combine = nn.Linear(d_model * 2, d_model)
        
        # Recency decay - learnable
        self.recency_decay = nn.Parameter(torch.tensor([0.9]))
        
    def forward(self, X_seq, padding_mask=None):
        # X_seq: [B, T]
        B, T = X_seq.shape
        device = X_seq.device
        
        # VECTORIZED IMPLEMENTATION
        # 1. FREQUENCY: one-hot count of each location
        # Create one-hot encoding: [B, T, vocab_size]
        one_hot = F.one_hot(X_seq, num_classes=self.vocab_size).float()  # [B, T, V]
        
        # Apply padding mask if provided
        if padding_mask is not None:
            # padding_mask is True for padding, False for valid
            mask = (~padding_mask).unsqueeze(-1).float()  # [B, T, 1]
            one_hot = one_hot * mask
        
        # Sum across time dimension to get frequency distribution
        freq_dist = one_hot.sum(dim=1)  # [B, V]
        
        # Normalize
        total_counts = freq_dist.sum(dim=1, keepdim=True) + 1e-8
        freq_dist = freq_dist / total_counts
        
        # 2. RECENCY: exponentially weighted by position (recent = higher)
        # Create position-based weights: more recent = higher weight
        # positions: [T] -> [0, 1, 2, ..., T-1]
        positions = torch.arange(T, device=device).float()  # [T]
        
        # Calculate sequence lengths
        if padding_mask is not None:
            seq_lens = (~padding_mask).sum(dim=1).float()  # [B]
        else:
            seq_lens = torch.full((B,), T, dtype=torch.float, device=device)  # [B]
        
        # Recency weights: decay^(seq_len - pos - 1)
        # Expand for broadcasting: positions [1, T], seq_lens [B, 1]
        decay_power = seq_lens.unsqueeze(1) - positions.unsqueeze(0) - 1  # [B, T]
        recency_weights = self.recency_decay ** decay_power  # [B, T]
        
        # Apply padding mask
        if padding_mask is not None:
            recency_weights = recency_weights * (~padding_mask).float()
        
        # Weighted one-hot: [B, T, V] * [B, T, 1]
        weighted_one_hot = one_hot * recency_weights.unsqueeze(-1)  # [B, T, V]
        recency_dist = weighted_one_hot.sum(dim=1)  # [B, V]
        
        # Normalize
        total_recency = recency_dist.sum(dim=1, keepdim=True) + 1e-8
        recency_dist = recency_dist / total_recency
        
        # Project and combine
        freq_emb = F.relu(self.freq_proj(freq_dist))
        recency_emb = F.relu(self.recency_proj(recency_dist))
        
        combined = self.combine(torch.cat([freq_emb, recency_emb], dim=-1))
        
        return combined, freq_dist, recency_dist


class HierarchicalCopyAttention(nn.Module):
    """
    Multi-head attention over history with copy mechanism
    """
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query, keys, values, mask=None):
        # query: [B, d_model]
        # keys, values: [B, T, d_model]
        B = query.size(0)
        T = keys.size(1)
        
        # Multi-head projections
        Q = self.q_proj(query).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, head_dim]
        K = self.k_proj(keys).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)    # [B, H, T, head_dim]
        V = self.v_proj(values).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, head_dim]
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, 1, T]
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)  # [B, H, 1, T]
        
        # Weighted sum
        context = torch.matmul(attn_weights, V)  # [B, H, 1, head_dim]
        context = context.transpose(1, 2).contiguous().view(B, 1, self.d_model).squeeze(1)  # [B, d_model]
        
        output = self.out_proj(context)
        
        # Also return attention weights for copy mechanism
        copy_weights = attn_weights.mean(dim=1).squeeze(1)  # [B, T]
        
        return output, copy_weights


class AdvancedHierarchicalS2Model(nn.Module):
    """
    Advanced model with:
    - User embeddings
    - User-specific contextual features (frequency, recency)
    - Hierarchical attention and copy mechanism
    - Candidate filtering using hierarchy
    """
    def __init__(self, config):
        super().__init__()
        
        self.vocab_l11 = config['vocab_l11']
        self.vocab_l13 = config['vocab_l13']
        self.vocab_l14 = config['vocab_l14']
        self.vocab_X = config['vocab_X']
        self.num_users = config.get('num_users', 50)
        
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dropout = config['dropout']
        
        # USER EMBEDDING (personalization)
        self.user_emb = nn.Embedding(self.num_users, self.d_model // 2)
        
        # Location embeddings
        self.emb_l11 = nn.Embedding(self.vocab_l11 + 1, self.d_model // 4, padding_idx=0)
        self.emb_l13 = nn.Embedding(self.vocab_l13 + 1, self.d_model // 4, padding_idx=0)
        self.emb_l14 = nn.Embedding(self.vocab_l14 + 1, self.d_model // 3, padding_idx=0)
        self.emb_X = nn.Embedding(self.vocab_X + 1, self.d_model // 2, padding_idx=0)
        
        # Combine hierarchical embeddings
        combined_dim = self.d_model // 4 + self.d_model // 4 + self.d_model // 3 + self.d_model // 2
        self.loc_combine = nn.Linear(combined_dim, self.d_model)
        
        # USER CONTEXTUAL FEATURES (frequency, recency)
        self.user_context = UserContextualFeatures(self.d_model, self.vocab_X)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # GRU for sequence encoding (parameter efficient)
        self.gru = nn.GRU(
            self.d_model,
            self.d_model // 2,  # Smaller hidden size
            num_layers=1,  # Single layer
            batch_first=True,
            dropout=0.0,
            bidirectional=False  # Unidirectional to save parameters
        )
        self.gru_proj = nn.Linear(self.d_model // 2, self.d_model)
        
        # HIERARCHICAL COPY ATTENTION
        self.copy_attn = HierarchicalCopyAttention(self.d_model, num_heads=self.nhead)
        
        # Copy gate: decide between copy and generate
        self.copy_gate = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )
        
        # Hierarchical classifiers (smaller due to copy mechanism)
        hidden_dim = self.d_model // 2
        self.fc = nn.Linear(self.d_model, hidden_dim)
        
        self.classifier_l11 = nn.Linear(hidden_dim, self.vocab_l11)
        self.classifier_l13 = nn.Linear(hidden_dim, self.vocab_l13)
        self.classifier_l14 = nn.Linear(hidden_dim, self.vocab_l14)
        self.classifier_X = nn.Linear(hidden_dim, self.vocab_X)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, l11_seq, l13_seq, l14_seq, X_seq, user_seq, padding_mask=None, 
                hierarchy_map=None, use_filtering=False):
        B, T = X_seq.shape
        device = X_seq.device
        
        # === USER EMBEDDING ===
        user_emb = self.user_emb(user_seq[:, 0])  # [B, d_model//2] - user is constant
        
        # === USER CONTEXTUAL FEATURES ===
        user_context, freq_dist, recency_dist = self.user_context(X_seq, padding_mask)  # [B, d_model]
        
        # === LOCATION SEQUENCE ENCODING ===
        # Embed all hierarchical levels
        l11_emb = self.emb_l11(l11_seq)
        l13_emb = self.emb_l13(l13_seq)
        l14_emb = self.emb_l14(l14_seq)
        X_emb = self.emb_X(X_seq)
        
        # Combine hierarchical information
        loc_cat = torch.cat([l11_emb, l13_emb, l14_emb, X_emb], dim=-1)  # [B, T, combined_dim]
        loc_encoded = self.loc_combine(loc_cat)  # [B, T, d_model]
        
        # Add positional encoding
        loc_encoded = self.pos_encoder(loc_encoded)
        
        # === GRU ENCODING ===
        if padding_mask is not None:
            lengths = (~padding_mask).sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                loc_encoded, lengths, batch_first=True, enforce_sorted=False
            )
            gru_out, _ = self.gru(packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True, total_length=T)
        else:
            gru_out, _ = self.gru(loc_encoded)
        
        gru_out = self.gru_proj(gru_out)  # [B, T, d_model]
        gru_out = self.layer_norm(gru_out)
        
        # === GET FINAL STATE ===
        if padding_mask is not None:
            seq_lengths = (~padding_mask).sum(dim=1)
            last_indices = seq_lengths - 1
            batch_indices = torch.arange(B, device=device)
            final_state = gru_out[batch_indices, last_indices]
        else:
            final_state = gru_out[:, -1, :]  # [B, d_model]
        
        # === COPY MECHANISM WITH ATTENTION ===
        # Attend to history
        attended_history, copy_weights = self.copy_attn(final_state, gru_out, gru_out, mask=padding_mask)
        
        # Build copy distribution from attention weights - VECTORIZED
        # Use scatter_add to accumulate attention weights for each location
        copy_dist = torch.zeros(B, self.vocab_X, device=device)
        
        # Mask out padding positions in copy_weights
        if padding_mask is not None:
            masked_copy_weights = copy_weights.masked_fill(padding_mask, 0.0)  # [B, T]
        else:
            masked_copy_weights = copy_weights
        
        # scatter_add_: accumulate weights at indices specified by X_seq
        # X_seq: [B, T], masked_copy_weights: [B, T]
        copy_dist.scatter_add_(1, X_seq, masked_copy_weights)
        
        # Combine: user context + frequency + recency
        copy_dist = copy_dist + 0.3 * freq_dist + 0.2 * recency_dist
        copy_dist = copy_dist / (copy_dist.sum(dim=1, keepdim=True) + 1e-8)
        
        # === GENERATION DISTRIBUTION ===
        # Combine: final state + user embedding + user context + attended history
        user_emb_expanded = user_emb.unsqueeze(1).expand(-1, 1, -1).squeeze(1)  # [B, d_model//2]
        
        # Pad user_emb to d_model
        user_emb_full = torch.zeros(B, self.d_model, device=device)
        user_emb_full[:, :self.d_model//2] = user_emb_expanded
        
        combined_state = final_state + user_context + attended_history + user_emb_full
        combined_state = self.dropout_layer(combined_state)
        
        hidden = F.relu(self.fc(combined_state))
        
        # Hierarchical predictions
        logits_l11 = self.classifier_l11(hidden)
        logits_l13 = self.classifier_l13(hidden)
        logits_l14 = self.classifier_l14(hidden)
        gen_logits_X = self.classifier_X(hidden)
        
        # === COPY GATE ===
        gate_input = torch.cat([final_state, user_context, attended_history], dim=-1)
        copy_prob = self.copy_gate(gate_input)  # [B, 1]
        
        # === FINAL PREDICTION (INTERPOLATE COPY AND GENERATE) ===
        gen_prob = F.softmax(gen_logits_X, dim=-1)
        final_prob_X = copy_prob * copy_dist + (1 - copy_prob) * gen_prob
        logits_X = torch.log(final_prob_X + 1e-10)
        
        # === HIERARCHICAL CANDIDATE FILTERING (if enabled) ===
        if use_filtering and hierarchy_map is not None:
            # VECTORIZED FILTERING
            # Use predicted L11 to filter X candidates
            pred_l11 = torch.argmax(logits_l11, dim=-1)  # [B]
            
            # Pre-build filtering masks (this can be cached but for correctness we build it here)
            # Create a mask tensor [B, vocab_X] where True = valid candidate
            filter_mask = torch.zeros(B, self.vocab_X, dtype=torch.bool, device=device)
            
            # For each sample in batch, mark valid X locations
            for b in range(B):
                l11_pred = pred_l11[b].item()
                
                # Get valid X locations under this L11
                valid_X = []
                if l11_pred in hierarchy_map['s2_level11_to_13']:
                    for l13 in hierarchy_map['s2_level11_to_13'][l11_pred]:
                        if l13 in hierarchy_map['s2_level13_to_14']:
                            for l14 in hierarchy_map['s2_level13_to_14'][l13]:
                                if l14 in hierarchy_map['s2_level14_to_X']:
                                    valid_X.extend(hierarchy_map['s2_level14_to_X'][l14])
                
                # Mark valid candidates
                if len(valid_X) > 0:
                    valid_X_tensor = torch.tensor(valid_X, dtype=torch.long, device=device)
                    filter_mask[b, valid_X_tensor] = True
                else:
                    # If no valid candidates found, allow all (fallback)
                    filter_mask[b, :] = True
            
            # Apply mask: set invalid candidates to -inf
            logits_X = torch.where(filter_mask, logits_X, torch.tensor(-1e9, device=device))
        
        return {
            'logits_l11': logits_l11,
            'logits_l13': logits_l13,
            'logits_l14': logits_l14,
            'logits_X': logits_X,
            'copy_prob': copy_prob,
            'copy_dist': copy_dist,
            'freq_dist': freq_dist,
            'recency_dist': recency_dist,
            'hidden_l11': hidden,
            'hidden_l13': hidden,
            'hidden_l14': hidden,
            'hidden_X': combined_state,
        }
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
