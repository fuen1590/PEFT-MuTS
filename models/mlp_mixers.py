from base_modules.MLP import DualMixer
from train import set_seed
from base import AutoTestTrainableModule
from configs.configs import DualMixerConfig
from datasets.cmapss import get_partial_data, CmapssPartial, DEFAULT_ROOT, DEFAULT_SENSORS, Subset

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

import torch
import os
import numpy as np


class DualMixerModel(AutoTestTrainableModule):
    def __init__(self, config: DualMixerConfig):
        super().__init__(config)
        self.config = config
        self.mixer = DualMixer(window_size=config.window_size,
                          hidden_dim=config.hidden_dim,
                          in_features=config.in_features,
                          num_layers=config.layer_num,
                          dropout=config.dropout)
        self.to(config.device)

    def forward(self, x):
        # x.shape = (B, H, F)
        return self.mixer(x)
