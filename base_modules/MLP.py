import numpy as np
import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_dim,
                 out_features,
                 dropout=0.5,
                 device="cuda:0"):
        super(MLPBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=out_features),
            nn.Dropout(dropout)
        )
        self.to(device)

    def forward(self, x):
        return self.block(x)


class GatedAttention(nn.Module):
    def __init__(self, hidden_dim, device="cuda:0"):
        super(GatedAttention, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.Sigmoid()
        )
        self.weights = None
        self.to(device)

    def forward(self, x):
        weights = self.encoder(x)
        self.weights = weights
        return torch.mul(weights, x)

class DualMixerLayer(nn.Module):
    def __init__(self, window_size, hidden_dim, dropout=0.5):
        super(DualMixerLayer, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(in_features=window_size, out_features=window_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=window_size * 2, out_features=window_size),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(normalized_shape=window_size, elementwise_affine=True)
        self.ln2 = nn.LayerNorm(normalized_shape=hidden_dim, elementwise_affine=True)
        self.ln3 = nn.LayerNorm(normalized_shape=window_size, elementwise_affine=True)
        self.ln4 = nn.LayerNorm(normalized_shape=hidden_dim, elementwise_affine=True)
        # self.ln1 = nn.BatchNorm1d(num_features=hidden_dim, affine=True)
        # self.ln2 = nn.BatchNorm1d(num_features=window_size, affine=True)
        # self.ln3 = nn.BatchNorm1d(num_features=hidden_dim, affine=True)
        # self.ln4 = nn.BatchNorm1d(num_features=window_size, affine=True)
        self.gat_weights_1 = None
        self.gat_weights_2 = None
        self.gat1 = GatedAttention(hidden_dim=window_size)
        self.gat2 = GatedAttention(hidden_dim=hidden_dim)

    def forward(self, x1, x2):
        # x1.shape = (b, w, f), x2.shape = (b, w, f)
        x1 = x1.transpose(-1, -2)  # x1.shape = (b, f, w)
        x1 = self.ln1(self.block1(x1) + x1)  # x1.shape = (b, f, w)
        x2 = self.ln2(self.block2(x2) + x2)  # x2.shape = (b, w, f)
        x1 = self.ln3(x1 + self.gat2(x2).transpose(-1, -2))
        x2 = self.ln4(x2 + self.gat1(x1).transpose(-1, -2))  # x2.shape = (b, f, w)
        return x1.transpose(-1, -2), x2

class DualMixer(nn.Module):
    def __init__(self, window_size, in_features, hidden_dim, num_layers, dropout=0.5):
        super(DualMixer, self).__init__()
        self.layers = nn.ModuleList()
        self.input_embedding = nn.Linear(in_features=in_features, out_features=hidden_dim)
        for i in range(num_layers):
            self.layers.append(DualMixerLayer(window_size, hidden_dim, dropout=dropout))
        self.out_gat1 = GatedAttention(hidden_dim=window_size)
        self.out_gat2 = GatedAttention(hidden_dim=hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(in_features=hidden_dim*window_size, out_features=1)
        )

    def forward(self, x):
        # x.shape = (B, W, F)
        x = self.input_embedding(x)  # (B, W, H)
        f1 = x
        f2 = x
        for layer in self.layers:
            f1, f2 = layer(f1, f2)
        f1 = self.out_gat1(f1.transpose(-1, -2))
        f2 = self.out_gat2(f2)
        f = torch.flatten(f1.transpose(-1, -2) + f2, start_dim=-2, end_dim=-1)
        return self.output(f)
