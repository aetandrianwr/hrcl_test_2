import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np


class GeoLifeS2Dataset(Dataset):
    def __init__(self, data_path, max_len=60):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Input sequences (historical trajectory)
        l11_seq = sample['s2_level11_X']
        l13_seq = sample['s2_level13_X']
        l14_seq = sample['s2_level14_X']
        X_seq = sample['X']
        user_seq = sample['user_X']  # USER FEATURE!
        
        # Targets (next location at each level)
        l11_y = sample['s2_level11_Y']
        l13_y = sample['s2_level13_Y']
        l14_y = sample['s2_level14_Y']
        X_y = sample['Y']
        
        # Get sequence length
        seq_len = len(X_seq)
        
        return {
            'l11_seq': torch.tensor(l11_seq, dtype=torch.long),
            'l13_seq': torch.tensor(l13_seq, dtype=torch.long),
            'l14_seq': torch.tensor(l14_seq, dtype=torch.long),
            'X_seq': torch.tensor(X_seq, dtype=torch.long),
            'user_seq': torch.tensor(user_seq, dtype=torch.long),
            'l11_y': torch.tensor(l11_y, dtype=torch.long),
            'l13_y': torch.tensor(l13_y, dtype=torch.long),
            'l14_y': torch.tensor(l14_y, dtype=torch.long),
            'X_y': torch.tensor(X_y, dtype=torch.long),
            'seq_len': seq_len,
        }


def collate_fn(batch):
    # Find max sequence length in batch
    max_len = max([item['seq_len'] for item in batch])
    
    B = len(batch)
    
    # Initialize padded tensors (padding with 0)
    l11_seq_padded = torch.zeros(B, max_len, dtype=torch.long)
    l13_seq_padded = torch.zeros(B, max_len, dtype=torch.long)
    l14_seq_padded = torch.zeros(B, max_len, dtype=torch.long)
    X_seq_padded = torch.zeros(B, max_len, dtype=torch.long)
    user_seq_padded = torch.zeros(B, max_len, dtype=torch.long)
    padding_mask = torch.ones(B, max_len, dtype=torch.bool)
    
    # Targets
    l11_y = torch.zeros(B, dtype=torch.long)
    l13_y = torch.zeros(B, dtype=torch.long)
    l14_y = torch.zeros(B, dtype=torch.long)
    X_y = torch.zeros(B, dtype=torch.long)
    
    for i, item in enumerate(batch):
        seq_len = item['seq_len']
        l11_seq_padded[i, :seq_len] = item['l11_seq']
        l13_seq_padded[i, :seq_len] = item['l13_seq']
        l14_seq_padded[i, :seq_len] = item['l14_seq']
        X_seq_padded[i, :seq_len] = item['X_seq']
        user_seq_padded[i, :seq_len] = item['user_seq']
        padding_mask[i, :seq_len] = False
        
        l11_y[i] = item['l11_y']
        l13_y[i] = item['l13_y']
        l14_y[i] = item['l14_y']
        X_y[i] = item['X_y']
    
    return {
        'l11_seq': l11_seq_padded,
        'l13_seq': l13_seq_padded,
        'l14_seq': l14_seq_padded,
        'X_seq': X_seq_padded,
        'user_seq': user_seq_padded,
        'padding_mask': padding_mask,
        'l11_y': l11_y,
        'l13_y': l13_y,
        'l14_y': l14_y,
        'X_y': X_y,
    }


def create_dataloaders(batch_size=64, num_workers=4):
    train_dataset = GeoLifeS2Dataset('data/geolife/geolife_transformer_7_train.pk')
    val_dataset = GeoLifeS2Dataset('data/geolife/geolife_transformer_7_validation.pk')
    test_dataset = GeoLifeS2Dataset('data/geolife/geolife_transformer_7_test.pk')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
