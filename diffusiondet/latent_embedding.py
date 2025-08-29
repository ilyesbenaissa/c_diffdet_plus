import torch
from torch import nn

class LatentEmbedding(nn.Module):
    def __init__(self, d_model, n_mlp=3):
        super().__init__()
        self.d_model = d_model
        self.n_mlp = n_mlp
        
        layers = [
            nn.Linear(d_model, d_model),
            nn.SiLU()
        ]
        for _ in range(n_mlp - 1):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(nn.SiLU())
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, z):
        return self.mlp(z)

class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attn_output, _ = self.cross_attn(query, key, value)
        query = query + self.dropout(attn_output)
        query = self.norm(query)
        return query
