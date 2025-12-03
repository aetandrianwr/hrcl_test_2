import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class CopyAttentionMechanism(nn.Module):
    """Attention mechanism to copy from history"""
    def __init__(self, d_model):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)
        
    def forward(self, query, keys, mask=None):
        # query: [B, d_model] - current state
        # keys: [B, T, d_model] - history
        # mask: [B, T] - padding mask
        
        Q = self.query_proj(query).unsqueeze(1)  # [B, 1, d_model]
        K = self.key_proj(keys)  # [B, T, d_model]
        
        # Attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [B, 1, T]
        scores = scores.squeeze(1)  # [B, T]
        
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)  # [B, T]
        return attn_weights


class HierarchicalS2WithCopy(nn.Module):
    """
    Hierarchical S2 model with:
    - Copy mechanism from history (exploits 75% pattern)
    - User embeddings (strong user-location affinity)
    - Attention to recent positions (Last-1 = 28% accuracy)
    """
    def __init__(self, config):
        super().__init__()
        
        # Vocabulary sizes
        self.vocab_l11 = config['vocab_l11']
        self.vocab_l13 = config['vocab_l13']
        self.vocab_l14 = config['vocab_l14']
        self.vocab_X = config['vocab_X']
        self.num_users = config.get('num_users', 50)
        
        # Model dimensions
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        # USER EMBEDDING - critical for user-location patterns!
        self.user_emb = nn.Embedding(self.num_users, self.d_model)  # Project to full d_model
        
        # Location embeddings - smaller to save params
        emb_dim = self.d_model // 2  # 56 for d_model=112
        self.emb_l11 = nn.Embedding(self.vocab_l11 + 1, emb_dim // 2, padding_idx=0)  # 28
        self.emb_l13 = nn.Embedding(self.vocab_l13 + 1, emb_dim // 2, padding_idx=0)  # 28
        self.emb_l14 = nn.Embedding(self.vocab_l14 + 1, emb_dim, padding_idx=0)  # 56
        self.emb_X = nn.Embedding(self.vocab_X + 1, emb_dim, padding_idx=0)  # 56
        
        # Combine embeddings: 28 + 28 + 56 + 56 = 168, then project to d_model
        self.location_proj = nn.Linear(emb_dim * 3, self.d_model)  # 168 -> 112
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # Shared transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 2,
            dropout=self.dropout,
            batch_first=True,
            activation='gelu'
        )
        self.shared_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # COPY MECHANISM - attention over history
        self.copy_attn_X = CopyAttentionMechanism(self.d_model)
        
        # Copy gate: decide between generating new or copying from history
        self.copy_gate = nn.Linear(self.d_model * 2, 1)  # sigmoid gate
        
        # Hierarchical processing
        self.proj_l13 = nn.Linear(self.d_model * 2, self.d_model)
        self.proj_l14 = nn.Linear(self.d_model * 2, self.d_model)
        self.proj_X = nn.Linear(self.d_model * 2, self.d_model)
        
        # Classification heads - smaller due to copy mechanism
        hidden_dim = self.d_model // 2
        self.pre_class = nn.Linear(self.d_model, hidden_dim)
        
        self.classifier_l11 = nn.Linear(hidden_dim, self.vocab_l11)
        self.classifier_l13 = nn.Linear(hidden_dim, self.vocab_l13)
        self.classifier_l14 = nn.Linear(hidden_dim, self.vocab_l14)
        self.classifier_X = nn.Linear(hidden_dim, self.vocab_X)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, l11_seq, l13_seq, l14_seq, X_seq, user_seq, padding_mask=None, return_copy_scores=False):
        B, T = l11_seq.shape
        
        # Embed all levels + USER
        l11_emb = self.emb_l11(l11_seq)  # [B, T, emb_dim//2]
        l13_emb = self.emb_l13(l13_seq)
        l14_emb = self.emb_l14(l14_seq)  # [B, T, emb_dim]
        X_emb = self.emb_X(X_seq)
        
        # User embedding (constant per sequence but add to each timestep)
        user_emb = self.user_emb(user_seq[:, 0]).unsqueeze(1).expand(-1, T, -1)  # [B, T, d_model//2]
        
        # Combine all location levels
        loc_combined = torch.cat([l11_emb, l13_emb, l14_emb, X_emb], dim=-1)  # [B, T, emb_dim*2 + emb_dim//2]
        
        # Project to d_model
        combined = self.location_proj(loc_combined)  # [B, T, d_model]
        combined = self.pos_encoder(combined)
        
        # Add user information
        combined = combined + user_emb
        
        # Encode with transformer
        encoded = self.shared_transformer(combined, src_key_padding_mask=padding_mask)  # [B, T, d_model]
        
        # Get final state
        if padding_mask is not None:
            seq_lengths = (~padding_mask).sum(dim=1)
            last_indices = seq_lengths - 1
            batch_indices = torch.arange(B, device=encoded.device)
            final_state = encoded[batch_indices, last_indices]
        else:
            final_state = encoded[:, -1, :]  # [B, d_model]
        
        # COPY MECHANISM for X
        # Attention over encoded history (not raw embeddings!)
        copy_scores = self.copy_attn_X(final_state, encoded, mask=padding_mask)  # [B, T]
        
        # For each position in history, get the location ID
        # We need to convert attention scores to vocabulary distribution
        copy_dist = torch.zeros(B, self.vocab_X, device=X_seq.device)
        
        # Scatter-add: for each timestep, add attention weight to that location's score
        for b in range(B):
            for t in range(T):
                if padding_mask is None or not padding_mask[b, t]:
                    loc_id = X_seq[b, t].item()
                    copy_dist[b, loc_id] += copy_scores[b, t]
        
        # Generate distribution (standard classification)
        hidden = self.dropout_layer(final_state)
        hidden = F.gelu(self.pre_class(hidden))
        gen_logits = self.classifier_X(hidden)  # [B, vocab_X]
        
        # Copy gate: interpolate between generate and copy
        gate_input = torch.cat([final_state, encoded[batch_indices, last_indices] if padding_mask is not None else encoded[:, -1, :]], dim=-1)
        copy_gate = torch.sigmoid(self.copy_gate(gate_input))  # [B, 1]
        
        # Final logits: mix generation and copying
        # Use log-space for numerical stability
        gen_prob = F.log_softmax(gen_logits, dim=-1)
        copy_prob = torch.log(copy_dist + 1e-10)  # Add small epsilon
        
        # Weighted combination
        final_logits_X = torch.log(copy_gate * torch.exp(copy_prob) + (1 - copy_gate) * torch.exp(gen_prob) + 1e-10)
        
        # For hierarchical levels, use standard approach
        # Process through hierarchy
        # Level 11
        l11_hidden = self.shared_transformer(self.pos_encoder(self.emb_l11(l11_seq)), src_key_padding_mask=padding_mask)
        l11_last = l11_hidden[batch_indices, last_indices] if padding_mask is not None else l11_hidden[:, -1, :]
        
        # Level 13
        l13_input = torch.cat([l11_hidden, self.emb_l13(l13_seq)], dim=-1)
        l13_proj = self.proj_l13(l13_input)
        l13_hidden = self.shared_transformer(self.pos_encoder(l13_proj), src_key_padding_mask=padding_mask)
        l13_last = l13_hidden[batch_indices, last_indices] if padding_mask is not None else l13_hidden[:, -1, :]
        
        # Level 14
        l14_input = torch.cat([l13_hidden, self.emb_l14(l14_seq)], dim=-1)
        l14_proj = self.proj_l14(l14_input)
        l14_hidden = self.shared_transformer(self.pos_encoder(l14_proj), src_key_padding_mask=padding_mask)
        l14_last = l14_hidden[batch_indices, last_indices] if padding_mask is not None else l14_hidden[:, -1, :]
        
        # Classifiers for hierarchical levels
        l11_h = F.gelu(self.pre_class(self.dropout_layer(l11_last)))
        l13_h = F.gelu(self.pre_class(self.dropout_layer(l13_last)))
        l14_h = F.gelu(self.pre_class(self.dropout_layer(l14_last)))
        
        logits_l11 = self.classifier_l11(l11_h)
        logits_l13 = self.classifier_l13(l13_h)
        logits_l14 = self.classifier_l14(l14_h)
        
        result = {
            'logits_l11': logits_l11,
            'logits_l13': logits_l13,
            'logits_l14': logits_l14,
            'logits_X': final_logits_X,
            'hidden_l11': l11_last,
            'hidden_l13': l13_last,
            'hidden_l14': l14_last,
            'hidden_X': final_state,
        }
        
        if return_copy_scores:
            result['copy_scores'] = copy_scores
            result['copy_gate'] = copy_gate
        
        return result
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
