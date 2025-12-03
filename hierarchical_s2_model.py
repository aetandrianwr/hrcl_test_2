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


class HierarchicalS2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Vocabulary sizes (including padding at index 0)
        self.vocab_l11 = config['vocab_l11']
        self.vocab_l13 = config['vocab_l13']
        self.vocab_l14 = config['vocab_l14']
        self.vocab_X = config['vocab_X']
        
        # Model dimensions
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        # Embedding layers with smaller dimensions for lower levels
        emb_dim_l11 = self.d_model // 2
        emb_dim_l13 = self.d_model // 2
        emb_dim_l14 = int(self.d_model * 0.75)
        
        self.emb_l11 = nn.Embedding(self.vocab_l11 + 1, emb_dim_l11, padding_idx=0)
        self.emb_l13 = nn.Embedding(self.vocab_l13 + 1, emb_dim_l13, padding_idx=0)
        self.emb_l14 = nn.Embedding(self.vocab_l14 + 1, emb_dim_l14, padding_idx=0)
        self.emb_X = nn.Embedding(self.vocab_X + 1, self.d_model, padding_idx=0)
        
        # Project embeddings to d_model
        self.emb_proj_l11 = nn.Linear(emb_dim_l11, self.d_model)
        self.emb_proj_l13 = nn.Linear(emb_dim_l13, self.d_model)
        self.emb_proj_l14 = nn.Linear(emb_dim_l14, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=100)
        
        # SHARED transformer encoder for all levels
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 2,
            dropout=self.dropout,
            batch_first=True,
            activation='gelu'
        )
        self.shared_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Projection layers for concatenated inputs (simpler)
        self.proj_l13 = nn.Linear(self.d_model * 2, self.d_model)
        self.proj_l14 = nn.Linear(self.d_model * 2, self.d_model)
        self.proj_X = nn.Linear(self.d_model * 2, self.d_model)
        
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
    
    def create_padding_mask(self, seq, pad_idx=0):
        # seq: [B, T]
        # Returns mask [B, T] where True = padding
        return seq == pad_idx
    
    def forward(self, l11_seq, l13_seq, l14_seq, X_seq, padding_mask=None):
        # All inputs: [B, T]
        # padding_mask: [B, T], True = padding
        
        B, T = l11_seq.shape
        
        # Create attention mask for transformer (True = ignore)
        if padding_mask is not None:
            attn_mask = padding_mask
        else:
            attn_mask = None
        
        # Level 11: embed, project, and encode
        l11_emb = self.emb_l11(l11_seq)  # [B, T, emb_dim_l11]
        l11_emb = self.emb_proj_l11(l11_emb)  # [B, T, d_model]
        l11_emb = self.pos_encoder(l11_emb)
        
        l11_hidden = self.shared_transformer(l11_emb, src_key_padding_mask=attn_mask)  # [B, T, d_model]
        
        # Level 13: concatenate l11_hidden with l13 embeddings
        l13_emb = self.emb_l13(l13_seq)  # [B, T, emb_dim_l13]
        l13_emb = self.emb_proj_l13(l13_emb)  # [B, T, d_model]
        l13_combined = torch.cat([l11_hidden, l13_emb], dim=-1)  # [B, T, 2*d_model]
        l13_combined = self.proj_l13(l13_combined)  # [B, T, d_model]
        l13_combined = self.pos_encoder(l13_combined)
        
        l13_hidden = self.shared_transformer(l13_combined, src_key_padding_mask=attn_mask)  # [B, T, d_model]
        
        # Level 14: concatenate l13_hidden with l14 embeddings
        l14_emb = self.emb_l14(l14_seq)  # [B, T, emb_dim_l14]
        l14_emb = self.emb_proj_l14(l14_emb)  # [B, T, d_model]
        l14_combined = torch.cat([l13_hidden, l14_emb], dim=-1)  # [B, T, 2*d_model]
        l14_combined = self.proj_l14(l14_combined)  # [B, T, d_model]
        l14_combined = self.pos_encoder(l14_combined)
        
        l14_hidden = self.shared_transformer(l14_combined, src_key_padding_mask=attn_mask)  # [B, T, d_model]
        
        # Level X: concatenate l14_hidden with X embeddings
        X_emb = self.emb_X(X_seq)  # [B, T, d_model]
        X_combined = torch.cat([l14_hidden, X_emb], dim=-1)  # [B, T, 2*d_model]
        X_combined = self.proj_X(X_combined)  # [B, T, d_model]
        X_combined = self.pos_encoder(X_combined)
        
        X_hidden = self.shared_transformer(X_combined, src_key_padding_mask=attn_mask)  # [B, T, d_model]
        
        # Use last non-padded timestep for each sequence
        if padding_mask is not None:
            # Get the last valid (non-padded) index for each sequence
            seq_lengths = (~padding_mask).sum(dim=1)  # [B]
            last_indices = seq_lengths - 1  # [B]
            
            # Gather last valid hidden states
            batch_indices = torch.arange(B, device=l11_hidden.device)
            l11_last = l11_hidden[batch_indices, last_indices]  # [B, d_model]
            l13_last = l13_hidden[batch_indices, last_indices]
            l14_last = l14_hidden[batch_indices, last_indices]
            X_last = X_hidden[batch_indices, last_indices]
        else:
            # Use last timestep
            l11_last = l11_hidden[:, -1, :]  # [B, d_model]
            l13_last = l13_hidden[:, -1, :]
            l14_last = l14_hidden[:, -1, :]
            X_last = X_hidden[:, -1, :]
        
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
