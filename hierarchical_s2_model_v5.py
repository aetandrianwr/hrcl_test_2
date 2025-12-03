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


class HierarchicalS2ModelV5(nn.Module):
    """
    Optimized model with:
    - Larger core dimensions
    - Factorized classifiers to reduce parameters
    - Better balance between model capacity and parameter budget
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
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        # Factorization dimension for classifiers
        self.factor_dim = config.get('factor_dim', self.d_model // 2)
        
        # Embeddings - balanced sizes
        emb_l11 = min(64, self.d_model)
        emb_l13 = min(80, self.d_model)
        emb_l14 = min(96, self.d_model)
        emb_X = self.d_model
        
        self.emb_l11 = nn.Embedding(self.vocab_l11 + 1, emb_l11, padding_idx=0)
        self.emb_l13 = nn.Embedding(self.vocab_l13 + 1, emb_l13, padding_idx=0)
        self.emb_l14 = nn.Embedding(self.vocab_l14 + 1, emb_l14, padding_idx=0)
        self.emb_X = nn.Embedding(self.vocab_X + 1, emb_X, padding_idx=0)
        
        # Project embeddings to d_model
        self.emb_proj_l11 = nn.Linear(emb_l11, self.d_model) if emb_l11 != self.d_model else nn.Identity()
        self.emb_proj_l13 = nn.Linear(emb_l13, self.d_model) if emb_l13 != self.d_model else nn.Identity()
        self.emb_proj_l14 = nn.Linear(emb_l14, self.d_model) if emb_l14 != self.d_model else nn.Identity()
        
        # Concat projection layers
        self.proj_l13 = nn.Linear(self.d_model * 2, self.d_model)
        self.proj_l14 = nn.Linear(self.d_model * 2, self.d_model)
        self.proj_X = nn.Linear(self.d_model * 2, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # Shared transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 2,
            dropout=self.dropout,
            batch_first=True,
            activation='gelu'
        )
        self.shared_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Factorized classification heads to save parameters
        # Instead of direct linear: hidden -> vocab_size
        # Use: hidden -> factor_dim -> vocab_size
        self.pre_classifier = nn.Linear(self.d_model, self.factor_dim)
        self.classifier_l11 = nn.Linear(self.factor_dim, self.vocab_l11)
        self.classifier_l13 = nn.Linear(self.factor_dim, self.vocab_l13)
        self.classifier_l14 = nn.Linear(self.factor_dim, self.vocab_l14)
        self.classifier_X = nn.Linear(self.factor_dim, self.vocab_X)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, l11_seq, l13_seq, l14_seq, X_seq, padding_mask=None):
        B, T = l11_seq.shape
        
        # Level 11
        l11_emb = self.emb_l11(l11_seq)
        l11_emb = self.emb_proj_l11(l11_emb)
        l11_emb = self.pos_encoder(l11_emb)
        l11_hidden = self.shared_transformer(l11_emb, src_key_padding_mask=padding_mask)
        
        # Level 13
        l13_emb = self.emb_l13(l13_seq)
        l13_emb = self.emb_proj_l13(l13_emb)
        l13_combined = torch.cat([l11_hidden, l13_emb], dim=-1)
        l13_combined = self.proj_l13(l13_combined)
        l13_combined = self.pos_encoder(l13_combined)
        l13_hidden = self.shared_transformer(l13_combined, src_key_padding_mask=padding_mask)
        
        # Level 14
        l14_emb = self.emb_l14(l14_seq)
        l14_emb = self.emb_proj_l14(l14_emb)
        l14_combined = torch.cat([l13_hidden, l14_emb], dim=-1)
        l14_combined = self.proj_l14(l14_combined)
        l14_combined = self.pos_encoder(l14_combined)
        l14_hidden = self.shared_transformer(l14_combined, src_key_padding_mask=padding_mask)
        
        # Level X
        X_emb = self.emb_X(X_seq)
        X_combined = torch.cat([l14_hidden, X_emb], dim=-1)
        X_combined = self.proj_X(X_combined)
        X_combined = self.pos_encoder(X_combined)
        X_hidden = self.shared_transformer(X_combined, src_key_padding_mask=padding_mask)
        
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
        
        # Apply dropout
        l11_last = self.dropout_layer(l11_last)
        l13_last = self.dropout_layer(l13_last)
        l14_last = self.dropout_layer(l14_last)
        X_last = self.dropout_layer(X_last)
        
        # Factorized classification
        l11_factor = F.gelu(self.pre_classifier(l11_last))
        l13_factor = F.gelu(self.pre_classifier(l13_last))
        l14_factor = F.gelu(self.pre_classifier(l14_last))
        X_factor = F.gelu(self.pre_classifier(X_last))
        
        logits_l11 = self.classifier_l11(l11_factor)
        logits_l13 = self.classifier_l13(l13_factor)
        logits_l14 = self.classifier_l14(l14_factor)
        logits_X = self.classifier_X(X_factor)
        
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
