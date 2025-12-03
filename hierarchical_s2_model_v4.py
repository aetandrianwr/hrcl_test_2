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


class SimpleAttentionLayer(nn.Module):
    """Lightweight attention without FFN"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm(x + self.dropout(attn_out))
        return x


class HierarchicalS2ModelV4(nn.Module):
    """
    Ultra-compact: minimal embeddings + lightweight attention + shared weights
    """
    def __init__(self, config):
        super().__init__()
        
        # Vocabulary sizes
        self.vocab_l11 = config['vocab_l11']
        self.vocab_l13 = config['vocab_l13']
        self.vocab_l14 = config['vocab_l14']
        self.vocab_X = config['vocab_X']
        
        # Model dimensions
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dropout = config['dropout']
        
        # Tiny embedding dimensions
        emb_l11 = max(16, self.d_model // 6)
        emb_l13 = max(20, self.d_model // 5)
        emb_l14 = max(24, self.d_model // 4)
        emb_X = max(32, self.d_model // 3)
        
        self.emb_l11 = nn.Embedding(self.vocab_l11 + 1, emb_l11, padding_idx=0)
        self.emb_l13 = nn.Embedding(self.vocab_l13 + 1, emb_l13, padding_idx=0)
        self.emb_l14 = nn.Embedding(self.vocab_l14 + 1, emb_l14, padding_idx=0)
        self.emb_X = nn.Embedding(self.vocab_X + 1, emb_X, padding_idx=0)
        
        # Single shared projection layer
        max_cat_dim = max(emb_l11, emb_l13 + self.d_model, emb_l14 + self.d_model, emb_X + self.d_model)
        self.shared_proj = nn.Linear(max_cat_dim, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # Single shared attention layer
        self.shared_attn = SimpleAttentionLayer(self.d_model, self.nhead, dropout=self.dropout)
        
        # Classification heads
        self.classifier_l11 = nn.Linear(self.d_model, self.vocab_l11)
        self.classifier_l13 = nn.Linear(self.d_model, self.vocab_l13)
        self.classifier_l14 = nn.Linear(self.d_model, self.vocab_l14)
        self.classifier_X = nn.Linear(self.d_model, self.vocab_X)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _project_and_process(self, x, attn_mask):
        """Shared processing function"""
        # Pad to max dimension for shared projection
        B, T, D = x.shape
        if D < self.shared_proj.in_features:
            padding = torch.zeros(B, T, self.shared_proj.in_features - D, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=-1)
        
        x = self.shared_proj(x)
        x = self.pos_encoder(x)
        x = self.shared_attn(x, key_padding_mask=attn_mask)
        return x
    
    def forward(self, l11_seq, l13_seq, l14_seq, X_seq, padding_mask=None):
        B, T = l11_seq.shape
        
        # Level 11
        l11_emb = self.emb_l11(l11_seq)
        l11_hidden = self._project_and_process(l11_emb, padding_mask)
        
        # Level 13 - concatenate with previous level
        l13_emb = self.emb_l13(l13_seq)
        l13_cat = torch.cat([l13_emb, l11_hidden], dim=-1)
        l13_hidden = self._project_and_process(l13_cat, padding_mask)
        
        # Level 14
        l14_emb = self.emb_l14(l14_seq)
        l14_cat = torch.cat([l14_emb, l13_hidden], dim=-1)
        l14_hidden = self._project_and_process(l14_cat, padding_mask)
        
        # Level X
        X_emb = self.emb_X(X_seq)
        X_cat = torch.cat([X_emb, l14_hidden], dim=-1)
        X_hidden = self._project_and_process(X_cat, padding_mask)
        
        # Get last timestep
        if padding_mask is not None:
            seq_lengths = (~padding_mask).sum(dim=1)
            last_indices = seq_lengths - 1
            batch_indices = torch.arange(B, device=l11_hidden.device)
            l11_last = l11_hidden[batch_indices, last_indices]
            l13_last = l13_hidden[batch_indices, last_indices]
            l14_last = l14_hidden[batch_indices, last_indices]
            X_last = X_hidden[batch_indices, last_indices]
        else:
            l11_last = l11_hidden[:, -1, :]
            l13_last = l13_hidden[:, -1, :]
            l14_last = l14_hidden[:, -1, :]
            X_last = X_hidden[:, -1, :]
        
        # Classification
        logits_l11 = self.classifier_l11(l11_last)
        logits_l13 = self.classifier_l13(l13_last)
        logits_l14 = self.classifier_l14(l14_last)
        logits_X = self.classifier_X(X_last)
        
        return {
            'logits_l11': logits_l11,
            'logits_l13': logits_l13,
            'logits_l14': logits_l14,
            'logits_X': logits_X,
            'hidden_l11': l11_last,
            'hidden_l13': l13_last,
            'hidden_l14': l14_last,
            'hidden_X': X_last,
        }
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
