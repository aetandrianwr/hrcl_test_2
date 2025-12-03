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
        # x: [B, T, d_model]
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class HierarchicalS2ModelV2(nn.Module):
    """
    More parameter-efficient model using GRU + Attention instead of full Transformers
    """
    def __init__(self, config):
        super().__init__()
        
        # Vocabulary sizes (including padding at index 0)
        self.vocab_l11 = config['vocab_l11']
        self.vocab_l13 = config['vocab_l13']
        self.vocab_l14 = config['vocab_l14']
        self.vocab_X = config['vocab_X']
        
        # Model dimensions
        self.d_model = config['d_model']
        self.hidden_dim = config.get('hidden_dim', self.d_model)
        self.dropout = config['dropout']
        
        # Embedding layers - smaller for coarser levels
        emb_dim_l11 = self.d_model // 2
        emb_dim_l13 = int(self.d_model * 0.6)
        emb_dim_l14 = int(self.d_model * 0.75)
        
        self.emb_l11 = nn.Embedding(self.vocab_l11 + 1, emb_dim_l11, padding_idx=0)
        self.emb_l13 = nn.Embedding(self.vocab_l13 + 1, emb_dim_l13, padding_idx=0)
        self.emb_l14 = nn.Embedding(self.vocab_l14 + 1, emb_dim_l14, padding_idx=0)
        self.emb_X = nn.Embedding(self.vocab_X + 1, self.d_model, padding_idx=0)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=100)
        
        # GRU encoders for each level - 2 layers, bidirectional
        self.gru_l11 = nn.GRU(
            emb_dim_l11, self.hidden_dim // 2, num_layers=2,
            batch_first=True, dropout=self.dropout, bidirectional=True
        )
        
        self.gru_l13 = nn.GRU(
            emb_dim_l13 + self.hidden_dim, self.hidden_dim // 2, num_layers=2,
            batch_first=True, dropout=self.dropout, bidirectional=True
        )
        
        self.gru_l14 = nn.GRU(
            emb_dim_l14 + self.hidden_dim, self.hidden_dim // 2, num_layers=2,
            batch_first=True, dropout=self.dropout, bidirectional=True
        )
        
        self.gru_X = nn.GRU(
            self.d_model + self.hidden_dim, self.hidden_dim // 2, num_layers=2,
            batch_first=True, dropout=self.dropout, bidirectional=True
        )
        
        # Attention mechanisms for each level
        self.attn_l11 = nn.MultiheadAttention(self.hidden_dim, num_heads=4, dropout=self.dropout, batch_first=True)
        self.attn_l13 = nn.MultiheadAttention(self.hidden_dim, num_heads=4, dropout=self.dropout, batch_first=True)
        self.attn_l14 = nn.MultiheadAttention(self.hidden_dim, num_heads=4, dropout=self.dropout, batch_first=True)
        self.attn_X = nn.MultiheadAttention(self.hidden_dim, num_heads=4, dropout=self.dropout, batch_first=True)
        
        # Layer normalization
        self.ln_l11 = nn.LayerNorm(self.hidden_dim)
        self.ln_l13 = nn.LayerNorm(self.hidden_dim)
        self.ln_l14 = nn.LayerNorm(self.hidden_dim)
        self.ln_X = nn.LayerNorm(self.hidden_dim)
        
        # Classification heads with dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.classifier_l11 = nn.Linear(self.hidden_dim, self.vocab_l11)
        self.classifier_l13 = nn.Linear(self.hidden_dim, self.vocab_l13)
        self.classifier_l14 = nn.Linear(self.hidden_dim, self.vocab_l14)
        self.classifier_X = nn.Linear(self.hidden_dim, self.vocab_X)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, l11_seq, l13_seq, l14_seq, X_seq, padding_mask=None):
        # All inputs: [B, T]
        # padding_mask: [B, T], True = padding
        
        B, T = l11_seq.shape
        
        # Level 11: embed and encode
        l11_emb = self.emb_l11(l11_seq)  # [B, T, emb_dim_l11]
        
        # Pack sequence for GRU if we have padding
        if padding_mask is not None:
            lengths = (~padding_mask).sum(dim=1).cpu()
            l11_packed = nn.utils.rnn.pack_padded_sequence(
                l11_emb, lengths, batch_first=True, enforce_sorted=False
            )
            l11_gru_out, _ = self.gru_l11(l11_packed)
            l11_gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                l11_gru_out, batch_first=True, total_length=T
            )
        else:
            l11_gru_out, _ = self.gru_l11(l11_emb)  # [B, T, hidden_dim]
        
        # Self-attention
        attn_mask_full = None
        if padding_mask is not None:
            attn_mask_full = padding_mask
        
        l11_attn_out, _ = self.attn_l11(l11_gru_out, l11_gru_out, l11_gru_out, key_padding_mask=attn_mask_full)
        l11_hidden = self.ln_l11(l11_gru_out + l11_attn_out)  # [B, T, hidden_dim]
        
        # Level 13: concatenate with l11_hidden
        l13_emb = self.emb_l13(l13_seq)  # [B, T, emb_dim_l13]
        l13_input = torch.cat([l13_emb, l11_hidden], dim=-1)  # [B, T, emb_dim_l13 + hidden_dim]
        
        if padding_mask is not None:
            l13_packed = nn.utils.rnn.pack_padded_sequence(
                l13_input, lengths, batch_first=True, enforce_sorted=False
            )
            l13_gru_out, _ = self.gru_l13(l13_packed)
            l13_gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                l13_gru_out, batch_first=True, total_length=T
            )
        else:
            l13_gru_out, _ = self.gru_l13(l13_input)
        
        l13_attn_out, _ = self.attn_l13(l13_gru_out, l13_gru_out, l13_gru_out, key_padding_mask=attn_mask_full)
        l13_hidden = self.ln_l13(l13_gru_out + l13_attn_out)  # [B, T, hidden_dim]
        
        # Level 14: concatenate with l13_hidden
        l14_emb = self.emb_l14(l14_seq)  # [B, T, emb_dim_l14]
        l14_input = torch.cat([l14_emb, l13_hidden], dim=-1)
        
        if padding_mask is not None:
            l14_packed = nn.utils.rnn.pack_padded_sequence(
                l14_input, lengths, batch_first=True, enforce_sorted=False
            )
            l14_gru_out, _ = self.gru_l14(l14_packed)
            l14_gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                l14_gru_out, batch_first=True, total_length=T
            )
        else:
            l14_gru_out, _ = self.gru_l14(l14_input)
        
        l14_attn_out, _ = self.attn_l14(l14_gru_out, l14_gru_out, l14_gru_out, key_padding_mask=attn_mask_full)
        l14_hidden = self.ln_l14(l14_gru_out + l14_attn_out)  # [B, T, hidden_dim]
        
        # Level X: concatenate with l14_hidden
        X_emb = self.emb_X(X_seq)  # [B, T, d_model]
        X_input = torch.cat([X_emb, l14_hidden], dim=-1)
        
        if padding_mask is not None:
            X_packed = nn.utils.rnn.pack_padded_sequence(
                X_input, lengths, batch_first=True, enforce_sorted=False
            )
            X_gru_out, _ = self.gru_X(X_packed)
            X_gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                X_gru_out, batch_first=True, total_length=T
            )
        else:
            X_gru_out, _ = self.gru_X(X_input)
        
        X_attn_out, _ = self.attn_X(X_gru_out, X_gru_out, X_gru_out, key_padding_mask=attn_mask_full)
        X_hidden = self.ln_X(X_gru_out + X_attn_out)  # [B, T, hidden_dim]
        
        # Use last non-padded timestep for each sequence
        if padding_mask is not None:
            seq_lengths = (~padding_mask).sum(dim=1)  # [B]
            last_indices = seq_lengths - 1  # [B]
            
            batch_indices = torch.arange(B, device=l11_hidden.device)
            l11_last = l11_hidden[batch_indices, last_indices]  # [B, hidden_dim]
            l13_last = l13_hidden[batch_indices, last_indices]
            l14_last = l14_hidden[batch_indices, last_indices]
            X_last = X_hidden[batch_indices, last_indices]
        else:
            l11_last = l11_hidden[:, -1, :]
            l13_last = l13_hidden[:, -1, :]
            l14_last = l14_hidden[:, -1, :]
            X_last = X_hidden[:, -1, :]
        
        # Apply dropout before classification
        l11_last = self.dropout_layer(l11_last)
        l13_last = self.dropout_layer(l13_last)
        l14_last = self.dropout_layer(l14_last)
        X_last = self.dropout_layer(X_last)
        
        # Classification
        logits_l11 = self.classifier_l11(l11_last)  # [B, vocab_l11]
        logits_l13 = self.classifier_l13(l13_last)  # [B, vocab_l13]
        logits_l14 = self.classifier_l14(l14_last)  # [B, vocab_l14]
        logits_X = self.classifier_X(X_last)  # [B, vocab_X]
        
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
