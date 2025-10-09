from mamba_ssm import Mamba
from tools import Patching

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self,
                 patch_size,
                 patch_step,
                 hidden_dim,
                 mamba_state,
                 mamba_d_conv,
                 device):
        super().__init__()
        if patch_size > 1:
            self.patcher = Patching(patch_size=patch_size, patch_step=patch_step)
        else:
            self.patcher = None
            patch_size = 1
        self.encoder = Mamba(d_model=1, d_state=mamba_state, d_conv=mamba_d_conv, expand=2, device=device)
        self.fwd = nn.Linear(in_features=patch_size, out_features=hidden_dim)
        self.to(device)

    def forward(self, x):
        """
        x.shape = (B, L) or (B, L, 1)
        return : (B, P, D), where P is the number of patches, P = L if patch_size = 0.
        """
        if len(x.shape) == 3:
            x = x.reshape(-1, x.shape[-2])  # (b, L, C) -> (B, L)
        if self.patcher is not None:
            x = self.patcher(x)  # (B, P, S)
        else:
            x = x.unsqueeze(-1)  # (B, P, 1)
        B, P, _ = x.shape
        x = x.reshape(B*P, -1)  # (B*P, S)
        x = x.unsqueeze(-1)  # (B*P, S, 1)
        x = self.encoder(x).squeeze(dim=-1)  # (B*P, S)
        x = self.fwd(x)  # (B*P, D)
        x = x.reshape(B, P, -1)  # (B, P, D)
        return x


class MambaLayer(nn.Module):
    def __init__(self,
                 hidden_dim,
                 mamba_state,
                 mamba_d_conv,
                 device):
        super().__init__()
        self.encoder = nn.Sequential(
            # nn.LayerNorm(hidden_dim),
            Mamba(d_model=hidden_dim,
                  d_conv=mamba_d_conv,
                  d_state=mamba_state,
                  expand=2,
                  device=device)
        )
        self.fwd = nn.Sequential(
            # nn.LayerNorm(hidden_dim),
            nn.Linear(in_features=hidden_dim, out_features=2 * hidden_dim),
            nn.GELU(),
            nn.Linear(in_features=2 * hidden_dim, out_features=hidden_dim),
        )
        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        """
        x.shape = (B, P, D)
        """
        short_cut = x
        x = self.encoder(x) + short_cut  # (B, P, D)
        _, P, _ = x.shape
        if P>2:
            x = self.bn(self.avg_pool(x.transpose(-1, -2))).transpose(-1, -2)  # (B, P//2, D)
        else:
            x = self.bn(x.transpose(-1, -2)).transpose(-1, -2)
        short_cut = x
        x = self.fwd(x) + short_cut
        return x


class MambaExtractor(nn.Module):
    def __init__(self,
                 hidden_dim,
                 mamba_state,
                 mamba_d_conv,
                 layer_num,
                 device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mamba_d_conv = mamba_d_conv
        self.mamba_state = mamba_state
        self.device = device
        self.layers = nn.Sequential(*[MambaLayer(hidden_dim, mamba_state, mamba_d_conv, device) for _ in range(layer_num)])
        self.to(device)

    def forward(self, x):
        """
        x.shape = (B, L)
        """
        return self.layers(x).mean(dim=-2)  # (B, D)
