from typing import Union

from CNN import InputEmbedding, ResNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from os.path import join

from base_modules.MLP import DualMixer
from configs.configs import DAMCNNConfig, DualMixerConfig, IMDSSNConfig, PeftMuTSNoPretrainConfig
from models.mlp_mixers import DualMixerModel
from train import set_seed
from base import AutoTestTrainableModule
from configs.configs import PretrainedCNNConfig, PeftMuTSConfig


class MixingLayer(nn.Module):
    def __init__(self, extrac_dim: int, in_dim: int, inner_dim: int, num_tokens: int, dynamic=False, device="cuda"):
        """
        :param extrac_dim: the dimension of fused feature
        :param in_dim: the dimension of input feature
        :param inner_dim: the dimension of the Random Projection
        :param num_tokens: the number of fusion tokens
        :param dynamic: whether to use dynamic Random Projector, if True, the Random Projector will be different
                        for every Batch samples, if False, the Random Projector will be constant and initialized
                        while training start.
        """
        super().__init__()
        self.num_tokens = num_tokens
        self.in_dim = in_dim
        self.extrac_dim = extrac_dim
        self.dynamic = dynamic
        self.random_projector = None
        self.tokens = None
        self.inner_dim = inner_dim
        self.upper_mlp_q = nn.Linear(self.inner_dim, self.extrac_dim, bias=False)
        self.upper_mlp_k = nn.Linear(self.inner_dim, self.extrac_dim, bias=False)
        self.upper_mlp_v = nn.Linear(self.inner_dim, self.extrac_dim, bias=False)
        self.token_mlp = nn.Sequential(
            nn.Linear(self.extrac_dim, self.extrac_dim),
            nn.SiLU(),
            nn.Linear(self.extrac_dim, self.extrac_dim),
        )
        self.v_mlp_q = nn.Linear(self.extrac_dim, self.extrac_dim, bias=False)
        self.v_mlp_k = nn.Linear(self.extrac_dim, self.extrac_dim, bias=False)
        self.v_mlp_v = nn.Linear(self.extrac_dim, self.extrac_dim, bias=False)
        self.out_mlp = nn.Sequential(
            nn.Linear(self.extrac_dim, self.extrac_dim),
            nn.SiLU(),
            nn.Linear(self.extrac_dim, self.extrac_dim),
        )
        self.ln_1 = nn.LayerNorm(self.extrac_dim)
        self.ln_2 = nn.LayerNorm(self.extrac_dim)
        self.ln_3 = nn.LayerNorm(self.extrac_dim)
        self.ln_4 = nn.LayerNorm(self.extrac_dim)
        self.device = device
        if not self.dynamic:
            self._generate_random_projector()
        self._init_tokens()

    def forward(self, f, u=None):
        """
        f.shape = (B, N, T, D), D == in_dim
        u.shape = (B, num, De), De == extrac_dim
        """
        B, N, T, D = f.shape
        tokens = self.tokens  # (1, num, D)
        tokens = tokens.unsqueeze(dim=-2).repeat((B, 1, T, 1))  # (B, num, T, D)
        c_f = torch.concat([tokens, f], dim=1)  # (B, N, T, D), N=N+num
        if self.dynamic:
            self._generate_random_projector(B)
            qkv = torch.einsum("bntd,sbdi->sbnti", c_f, self.random_projector)
        else:
            qkv = torch.einsum("bntd,sdi->sbnti", c_f, self.random_projector)
        q, k, v = self.upper_mlp_q(qkv[0]), self.upper_mlp_k(qkv[1]), self.upper_mlp_v(qkv[2])  # (B, N, T, Du)
        v, _ = self._attention(q, k, v)  # (B, N, T, Du)
        f_tokens = v[:, :self.num_tokens].mean(dim=-2)  # (B, num, Du)
        v = v[:, self.num_tokens:]  # (B, N, T, Du)  N = N-num
        f_tokens = self.ln_1(self.token_mlp(f_tokens) - f_tokens)  # (B, num, Du)
        if u is not None:
            # f_tokens = self.ln_2(f_tokens + u).unsqueeze(dim=-2).repeat((1, 1, T, 1))  # (B, num, T, Du)
            f_tokens = torch.concat([u, f_tokens], dim=-2).unsqueeze(dim=-2).repeat((1, 1, T, 1))  # (B, num, T, Du)
            v_q = torch.concat([f_tokens, v], dim=1)  # (B, N, T, Du), N = N+num
            q, k, v = self.v_mlp_q(v_q), self.v_mlp_k(v), self.v_mlp_v(v)
            v_, _ = self._attention(q, k, v)
            f_tokens = self.ln_3(v_[:, :self.num_tokens] + f_tokens[:, :self.num_tokens])
            f_tokens = (self.out_mlp(f_tokens) + f_tokens).mean(dim=-2)
        return f_tokens

    def _generate_random_projector(self, batch=None):
        if self.dynamic:
            assert batch is not None
            self.random_projector = torch.randn((3, batch, self.in_dim, self.inner_dim)).to(self.device)
        else:
            if self.random_projector is None:
                self.random_projector = torch.randn((3, self.in_dim, self.inner_dim)).to(self.device)

    def _init_tokens(self):
        self.tokens = torch.randn((self.in_dim, self.num_tokens))
        self.tokens = torch.linalg.qr(self.tokens)[0]
        self.tokens = self.tokens.transpose(-1, -2).unsqueeze(dim=0)  # (1, k, D)
        self.tokens = nn.Parameter(self.tokens, requires_grad=True)

    def _attention(self, q, k, v):
        """
        q.shape = (B, N, T, D)
        """
        B, Nq, T, D = q.shape
        B, Nk, T, D = k.shape
        q_, k_, v_ = q.reshape(B, Nq, -1), k.reshape(B, Nk, -1), v.reshape(B, Nk, -1)
        # q_, k_ = q_/torch.norm(q_, dim=-1, keepdim=True), k_/torch.norm(k_, dim=-1, keepdim=True)
        att = torch.einsum("bnd,bmd->bnm", q_, k_) / self.extrac_dim ** 0.5
        att = torch.softmax(-att, dim=-1)  # (B, Nq, Nk)
        v_ = torch.einsum("bnm,bmd->bmd", att, v_)
        v_ = v_.reshape(B, Nk, T, D)
        return v_, att


class AdapterModule(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.cnn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim),
        )
        self.bn = nn.BatchNorm1d(in_dim)

    def forward(self, x):
        return self.cnn(x.transpose(-1, -2)).transpose(-1, -2) + x


class PretrainedCNN(AutoTestTrainableModule):
    def __init__(self,
                 config: PretrainedCNNConfig):
        super().__init__(config)
        self.saved_stat = config.saved_stat
        self.in_features = config.in_features
        self.input_embedding = InputEmbedding(in_channels=1, out_channels=config.embed_dim)
        self.encoder = ResNet(in_channels=config.embed_dim, size="big")
        self.reshape = True
        if config.saved_stat is not None:
            self._load_model()
        if config.fine_tune == "all":
            self.encoder.requires_grad_(True)
            self.input_embedding.requires_grad_(True)
        elif config.fine_tune == "false":
            self.encoder.requires_grad_(False)
            self.input_embedding.requires_grad_(False)
        elif config.fine_tune == "bias":
            self.encoder.requires_grad_(False)
            self.input_embedding.requires_grad_(False)
            for k, v in self.encoder.state_dict().items():
                if "bias" in k:
                    v.requires_grad_(True)
                    print(k)
            for k, v in self.input_embedding.state_dict().items():
                if "bias" in k:
                    v.requires_grad_(True)
        elif config.fine_tune == "input":
            self.input_embedding = InputEmbedding(in_channels=config.in_features, out_channels=config.embed_dim)
            self.encoder.requires_grad_(False)
            self.input_embedding.requires_grad_(True)
            self.reshape = False
        else:
            raise ValueError(f"Unknown fine-tuning mode:{config.fine_tune}.")
        self.output = nn.Linear(1024, 1) if config.output_layer_mode == "norm" \
            else OutputLayer(1024)
        self.config = config
        self.to(config.device)

    def forward(self, x):
        """
        x.shape = (B, T, N)
        """
        out = self.output(self.feature_extractor(x))
        return out

    def feature_extractor(self, x):
        B, T, N = x.shape
        if self.reshape:
            x = x.reshape(B * N, T, 1)  # (B*N, T, 1)
            x = self.input_embedding(x)  # (B*N, T, D)
        else:
            x = self.input_embedding(x)  # (B, T, D)
        x, (f1, f2, f3, f4) = self.encoder.forward_per_layers(x)  # (B*N, s, D)/(B, s, D)
        if self.reshape:
            x = x.reshape(B, N, x.shape[-2], -1)  # (B, N, T, D)
            return x.mean(dim=-2).mean(dim=-2)
        else:
            return x.mean(dim=-2)  # (B, D) -> (B, 1)

    def _load_model(self):
        _model = torch.load(join(self.saved_stat, "model_ck.pt"))
        self.input_embedding.load_state_dict(_model["input_embedding"])
        self.encoder.load_state_dict(_model["encoder"])


class SideNet(nn.Module):
    def __init__(self, in_dim, inner_dim, in_features, expert=1):
        super().__init__()
        self.inner_dim = inner_dim // expert
        self.A = nn.Parameter(torch.randn(in_dim, self.inner_dim))
        self.bias = nn.Parameter(torch.zeros(in_features + 1, 1, self.inner_dim))
        if expert > 1:
            self.router = nn.Sequential(
                nn.Linear(in_dim, expert),
                nn.Softmax(dim=-1)
            )
            self.B = nn.Parameter(torch.zeros(self.inner_dim, expert, in_dim))
        else:
            self.router = None
            self.B = nn.Parameter(torch.zeros(self.inner_dim, in_dim))

    def forward(self, x):
        a = self.get_router(x)
        low_rank = self.low_rank(x)
        output = self.high_rank(low_rank, a)
        return output

    def low_rank(self, x):
        # x.shape = (B, T, D)
        return torch.matmul(x, self.A)

    def high_rank(self, x, a = None):
        # x.shape = (B, T, D), a.shape = (B, T, E)
        if a is not None:
            assert a.shape[-1] == self.B.shape[-2], (f"Please check the number of experts, "
                                                     f"got the experts score with number {a.shape[-1]} "
                                                     f"but the experts parameters with number {self.B.shape[-2]}.")
            experts_out = torch.einsum("...D, DEH -> ...EH", x, self.B)  # (B, T, E, H)
            experts_out = torch.einsum("...E, ...EH -> ...H", a, experts_out)  # (B, T, H)
        else:
            experts_out = torch.matmul(x, self.B)
        return experts_out

    def get_router(self, x):
        if self.router is not None:
            a = self.router(x)
        else:
            a = None
        return a


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, beta=1):
        return x * F.sigmoid(beta * x)


class SSFLayer(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.gamma = nn.Parameter(torch.randn(in_features))
        self.beta = nn.Parameter(torch.zeros(in_features))

    def forward(self, x):
        """
        param: x.shape = (B, N, T, D)
        return: (B, N, T, D)
        """
        x = torch.mul(x, self.gamma)
        x = torch.add(x, self.beta)
        return x


class ShiftLayer(nn.Module):
    def __init__(self, features, dim_num=3, mode='zero', factor=1e-5):
        super().__init__()
        dim = [1] * dim_num
        dim[-1] = features
        self.gamma = nn.Parameter(torch.zeros(dim) if mode == 'zero' else torch.ones(dim) * factor, requires_grad=True)
        self.phi = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        return x * self.gamma + self.phi


class OutputLayer(nn.Module):
    def __init__(self, in_features, mode="zero"):
        super().__init__()
        assert mode in ["zero", "norm"]
        self.A = nn.Parameter(torch.zeros(in_features, 1))
        if mode == "norm":
            self.A = nn.Parameter(torch.empty(in_features, 1))
            torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return x @ self.A

class PeftMuTS(AutoTestTrainableModule):
    def __init__(self, config: Union[PeftMuTSConfig, PeftMuTSNoPretrainConfig]):
        super(PeftMuTS, self).__init__(config)
        self.in_features = config.in_features
        self.saved_stat = config.saved_stat
        self.input_embedding = InputEmbedding(in_channels=1, out_channels=config.embed_dim)
        self.encoder = ResNet(in_channels=config.embed_dim, size="big")
        self.e_layers = self.encoder.get_layers()
        self.side_dim = config.side_dim
        self.side_layers = nn.ModuleList([SideNet(dim[0], dim[1], config.in_features, expert=dim[2])
                                          for dim in self.side_dim])
        res_dim = [config.embed_dim, 128, 256, 512, 1024]
        self.DWConv_layers = nn.ModuleList([nn.Sequential(
            nn.Conv1d(
                in_channels=res_dim[i],
                out_channels=res_dim[i+1],
                kernel_size=3,
                stride=2,
                padding=1,
                groups=res_dim[i],
                bias=False
            )
        ) for i in range(len(res_dim)-1)])
        # self.output = nn.Sequential(nn.Linear(1024, 1))
        self.output = nn.Linear(1024, 1) if config.output_layer_mode == "norm" \
            else OutputLayer(1024)
        self.silu = SiLU()
        if config.saved_stat is not None:
            self._load_model()
        if config.fine_tune is False:
            self.encoder.requires_grad_(False)
            self.input_embedding.requires_grad_(False)

        # ! Just for experiment
        self.fusion = config.fusion
        if self.fusion:
            self.global_token = nn.Parameter(torch.zeros(1, config.window_size, 1), requires_grad=True)
        # self.temporal_tokens = nn.Parameter(torch.zeros(1, config.window_size, config.in_features), requires_grad=True)
        self.shift = config.shift
        if self.shift:
            self.beta = nn.Parameter(torch.ones(1) * config.shift_init_factor, requires_grad=True)
            self.phi = nn.Parameter(torch.zeros(1), requires_grad=True)
        else:
            self.beta = None
            self.phi = None
        self.to(config.device)

    def _load_model(self):
        _model = torch.load(join(self.saved_stat, "model_ck.pt"))
        self.input_embedding.load_state_dict(_model["input_embedding"])
        self.encoder.load_state_dict(_model["encoder"])

    def forward(self, x):
        f = self.feature_extractor(x)
        out = self.output(f)
        return out

    def feature_extractor(self, x):
        # x.shape = (B, T, N)
        B, T, N = x.shape
        if self.fusion:
            x = torch.concat([self.global_token.repeat(B, 1, 1), x], dim=-1)
            N += 1
        x = x.reshape(B * N, T, 1)  # (B*N, T, 1)

        # f0 = self.shift_layer(self.input_embedding(x))  # (B*N, T, D)
        f0 = self.input_embedding(x)  # (B*N, T, D)
        f0 = self.beta * f0 + self.phi if self.shift else f0
        f = f0.transpose(-1, -2)  # (B*N, D, T)

        s_f = 0
        i = 0  # encoder layers count
        j = 0  # DWConv count
        for layer in self.side_layers:
            f = f.transpose(-1, -2)  # (B*N, T, D)
            if self.fusion:
                router_a = layer.get_router(f)  # (B*N, T, E)
                l_f = layer.low_rank(f)  # (B*N, T, Dr)
                l_f = l_f.reshape(B, N, l_f.shape[-2], -1)  # (B, N, T, Dr)
                l_f[:, 0:1] = self.silu(l_f + layer.bias, beta=1).mean(dim=-3, keepdim=True)  # (B, N, T, Dr)
                l_f = l_f.reshape(B*N, l_f.shape[-2], -1)  # (B*N, T, Dr)
                h_f = layer.high_rank(l_f, router_a)  # (B*N, Ti, Di)
            else:
                h_f = layer(f)
            s_f = s_f + h_f  # (B*N, Ti, Di)

            if i % 2 == 0:
                s_f = self.DWConv_layers[j](s_f.transpose(-1, -2)).transpose(-1, -2)  # (B*N, Di+1, Ti+1)
                j += 1
            # f = self.e_layers[i](f.transpose(-1, -2))  # (B*N, Di, Ti)
            f = self.e_layers[i](f.transpose(-1, -2))  # (B*N, Di, Ti)
            f = f + self.silu(s_f.transpose(-1, -2), beta=0.01)
            i += 1
        f_out = f.mean(dim=-1, keepdim=False)  # (B*N, Di)
        if self.fusion:
            f_out = f_out.reshape(B, N, -1)[:, 0]
        else:
            f_out = f_out.reshape(B, N, -1)[:, 0]
        return f_out

    def compute_loss(self,
                     x: torch.Tensor,
                     y: torch.Tensor,
                     criterion) -> torch.Tensor:
        out = self(x)
        return criterion(out, y)

