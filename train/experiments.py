from models.DAMCNN import DAMCNN
from models.IMDSSN import IMDSSN
from models.base import AutoTestTrainableModule
from configs.configs import TrainConfig, PeftMuTSConfig, PretrainedCNNConfig, LinearPretrainedCNNConfig, \
    RandomCNNConfig, DAMCNNConfig, DualMixerConfig, IMDSSNConfig, PeftMuTSNoPretrainConfig
from datasets.dataset_helpers import generate_cmapss_dataset, generate_xjtu_dataset
from typing import Type, TypeVar, List

from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts
from torch.nn import MSELoss, L1Loss
from torch import save

import torch
import torch.nn as nn

from os.path import join

from models.mlp_mixers import DualMixerModel
from models.peft_muts import PretrainedCNN, PeftMuTS
from train import set_seed

MODEL_T = TypeVar("MODEL_T", bound=AutoTestTrainableModule)

AVAILABLE_MODELS = {"FEIResNet": [PretrainedCNNConfig, PretrainedCNN],
                    "LinearResNet": [LinearPretrainedCNNConfig, PretrainedCNN],
                    "RandomInitResNet": [RandomCNNConfig, PretrainedCNN],
                    "PeftMuTS": [PeftMuTSConfig, PeftMuTS],
                    "PeftMuTSNoPretrain": [PeftMuTSNoPretrainConfig, PeftMuTS],
                    "DAMCNN": [DAMCNNConfig, DAMCNN],
                    "IMDSSN": [IMDSSNConfig, IMDSSN],
                    "DualMixer": [DualMixerConfig, DualMixerModel], }

DATASET_CONFIGS_CMAPSS = {"FD002": [[0.3, 0.6, 0.8], [0.3, 0.3, 0.8],
                                    [0.3, 0.08, 0.8], [0.3, 0.03, 0.8]],
                          "FD004": [[0.3, 0.6, 0.8], [0.3, 0.3, 0.8],
                                    [0.3, 0.08, 0.8], [0.3, 0.03, 0.8]], }
DATASET_CONFIGS_BEARING = {"OP_A": [[0.05, 1.0], [0.1, 1.0]],
                           "OP_B": [[0.1, 0.1], [0.1, 0.5], [0.1, 1.0]],
                           "OP_C": [[0.1, 0.1], [0.1, 0.5], [0.1, 1.0]]}


class FocalMSELoss(nn.Module):
    def __init__(self, gamma=1.0, reduction='mean'):
        super(FocalMSELoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        # Compute absolute error
        error = torch.abs(input - target)
        # Compute focal weight
        focal_weight = error ** self.gamma
        # Apply to squared error
        loss = focal_weight * (error ** 2)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # no reduction


def test_results_format_smape(results: List):
    s = ""
    for r in results:
        assert isinstance(r, dict)
        s += "{:.4f};{:.2f};{:.4f}\n".format(r["mae"], r["smape"], r["rmse"])
    return s

def test_results_format_mape(results: List):
    s = ""
    for r in results:
        assert isinstance(r, dict)
        s += "{:.4f};{:.2f};{:.4f}\n".format(r["mae"], r["mape"]*100, r["rmse"])
    return s


def test_results_save(results: List, path: str):
    save(results, path)


def train(config: TrainConfig,
          modelType: Type[MODEL_T],
          times: int = 1,
          seeds: List[int] = None,):
    if "FD" in config.dataset:
        train_set, test_set, val_set = generate_cmapss_dataset(config.dataset,
                                                               config.window_size,
                                                               config.sample_step,
                                                               config.data_cut,
                                                               config.data_ratio)
    elif "OP" in config.dataset:
        train_set, test_set, val_set = generate_xjtu_dataset(config.dataset,
                                                             [1],
                                                             config.window_size,
                                                             config.sample_step,
                                                             config.data_cut,
                                                             config.data_ratio)
    results = []
    model = None
    visual_samples = test_set.data[: 220]
    visual_labels = test_set.labels[: 220, 0]
    for _ in range(times):
        if seeds is not None:
            if isinstance(seeds, int):
                set_seed(seeds)
            elif isinstance(seeds, list):
                if _ >= len(seeds):
                    Warning(f"The number of random seeds given '{len(seeds)}' is less than the training times '{times}',"
                            f"thus the random seeds is not applied!")
                else:
                    set_seed(seeds[_])
        model = modelType(config)
        model.prepare_data(train_set,
                           test_set,
                           val_set,
                           config.batch_size,
                           num_workers=0)
        if config.visual_epoch > 0 or config.feature_distance_epoch > 0:
            model.set_visual_samples(visual_samples, visual_labels)
        opt = AdamW(model.parameters(), lr=config.lr)
        sche = ExponentialLR(opt, gamma=0.99)
        model.train_model(config.epoch,
                          MSELoss(),
                          optimizer=opt,
                          lr_schedular=sche,
                          early_stop=config.early_stopping)
        results.append(model.test_results)
    if model is not None:
        test_results_save(results, join(model.model_path, "results.pt"))
    print("Training Done.")
    return results


if __name__ == '__main__':
    results = {}
    random_seeds = [20251, 20252, 20253, 20254, 20255]
    datasets = ["FD002", "FD004"]
    ratios = [[0.3, 0.03, 0.8], [0.3, 0.03, 0.8]]
    config = AVAILABLE_MODELS["PeftMuTS"][0]()
    model = AVAILABLE_MODELS["PeftMuTS"][1]
    config.dataset = datasets[0]
    config.data_ratio = DATASET_CONFIGS_CMAPSS[config.dataset][0]
    train(config, model, times=5, seeds=random_seeds)
