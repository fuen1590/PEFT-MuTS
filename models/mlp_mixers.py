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


if __name__ == '__main__':
    set_seed(2025)
    config = DualMixerConfig()
    train, test, val, _ = get_partial_data(DEFAULT_ROOT,
                                           subset=Subset.__members__[config.dataset],
                                           window_size=config.window_size,
                                           slide_step=config.sample_step,
                                           sensors=DEFAULT_SENSORS,
                                           rul_threshold=125,
                                           label_norm=True,
                                           val_ratio=0.2,
                                           mode=config.data_cut,
                                           ratio=config.data_ratio,
                                           retrain_max_label=True)
    model = DualMixerModel(config)
    model.prepare_data(train,
                       test,
                       val,
                       config.batch_size,
                       num_workers=1)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    sche = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=1)
    model.train_model(config.epoch,
                      torch.nn.MSELoss(),
                      optimizer=opt,
                      lr_schedular=sche,
                      early_stop=config.early_stopping)
