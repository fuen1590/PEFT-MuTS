import torch

from configs import TrainConfig
from train import TrainableModule
from os.path import join
from typing import Union
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

import numpy as np


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Symmetric Mean Absolute Percentage Error (SMAPE)
    between y_true and y_pred for (B, 1) shaped arrays.

    Args:
        y_true (np.ndarray): Ground truth values, shape (B, 1)
        y_pred (np.ndarray): Predicted values, shape (B, 1)

    Returns:
        float: SMAPE value (percentage, 0 ~ 200)
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    denominator = (np.abs(y_true) + np.abs(y_pred)) + 1e-8  # Avoid division by 0
    diff = np.abs(y_true - y_pred)

    smape_value = 100 * np.mean(diff / denominator)
    return smape_value


class AutoTestTrainableModule(TrainableModule):
    def __init__(self, config: TrainConfig):
        super(AutoTestTrainableModule, self).__init__(config)
        self.config = config
        self.test_results = {"rmse": 0, "mae": 0, "mape": 0, "SMAPE": 0}
        if config.visual_epoch > 0 or config.feature_distance_epoch > 0:
            self.visual_samples = None
            self.visual_labels = None
            self.visual_features = []

    def test_end(self):
        label_path = join(self.model_path, "model_test_labels_part1.npy")
        output_path = join(self.model_path, "model_test_output_part1.npy")
        labels = np.load(label_path)
        outputs = np.load(output_path)
        mse = mean_squared_error(labels, outputs)
        mae = mean_absolute_error(labels, outputs)
        mape = mean_absolute_percentage_error(labels, outputs)
        s_mape = smape(labels, outputs)
        self.logger.info("Test results:\n\tMSE:{:.4f}\n\tMAE:{:.4f}\n\tMAPE:{:.4f}\n\tRMSE:{:.4f}\n\tSMAPE:{:.4f}".
                         format(mse, mae, mape, mse ** 0.5, s_mape))
        self.test_results["rmse"] = mse ** 0.5
        self.test_results["mae"] = mae
        self.test_results["mape"] = mape
        self.test_results["smape"] = s_mape
        self._visual_process()
        torch.save(np.stack(self.train_losses, axis=0), join(self.model_path, "losses.pt"))

    def train_start(self):
        params_num = 0
        for p in self.parameters():
            if p.requires_grad:
                params_num += p.numel()
        self.logger.info(f"Fine-tuning parameters: {params_num}")
        self._visual_features_process(epoch=-1)

    def feature_extractor(self, x: torch.Tensor):
        """
        :param x: input with shape (B, ...)
        :return: y.shape = (B, D), where D = feature dimension
        """
        if self.config.visual_epoch > 0:
            raise NotImplementedError("The AutoTestTrainableModule needs the feature_extractor method while using"
                                      "visual process.")

    def set_visual_samples(self, vx: Union[torch.Tensor, np.ndarray], labels: Union[torch.Tensor, np.ndarray]):
        """
        :param vx: input with shape (B, ...)
        :param labels: input with shape (B)
        """
        self.visual_samples = torch.FloatTensor(torch.from_numpy(vx).to(torch.float32)).to(self.config.device) \
            if isinstance(vx, np.ndarray) else vx.to(self.config.device)
        self.visual_labels = labels

    def epoch_end(self, epoch):
        self._visual_features_process(epoch)

    def test_start(self):
        self._visual_features_process(epoch=-1)

    def _visual_features_process(self, epoch):
        if self.config.visual_epoch > 0:
            if epoch % self.config.visual_epoch == 0 or epoch==-1:
                self._saving_visual_features()
        if self.config.feature_distance_epoch > 0:
            if epoch % self.config.feature_distance_epoch == 0 or epoch==-1:
                self._saving_visual_features()

    def _saving_visual_features(self):
        if self.visual_samples is not None and self.visual_labels is not None:
            self.logger.info(f"Saving visual features...")
            with torch.no_grad():
                self.eval()
                self.visual_features.append(self.feature_extractor(self.visual_samples))
                self.train()
        else:
            self.logger.info("You turn on the visualizing but the visual samples/labels are not given! Please set the "
                             "visual samples by model.set_visual_samples(x: torch.Tensor, labels: torch.Tensor).")

    def _visual_process(self):
        if self.config.visual_epoch > 0:
            N = len(self.visual_features)
            visual_features = torch.concat(self.visual_features, dim=0)  # (B*N, D), N = times of visualizing
            visual_features = feature_visual(visual_features, 10)  # ndarray, (B*N, 2)
            visual_features = visual_features.reshape(-1, N, visual_features.shape[-1])  # ndarray, (B, N, 2)
            if isinstance(self.visual_labels, torch.Tensor):
                visual_labels = self.visual_labels.cpu().numpy()
            else:
                visual_labels = self.visual_labels
            p = plot_epochwise_samples_with_label_edges(visual_features, visual_labels)
            p.savefig(join(self.model_path, "visual.png"))
            p.show()
            visual_data = {"features": torch.stack(self.visual_features, dim=1),"tsne": visual_features, "labels": visual_labels}
            torch.save(visual_data, join(self.model_path, "visual_data.pt"))
        if self.config.feature_distance_epoch > 0:
            N = len(self.visual_features)
            features = torch.stack(self.visual_features, dim=0)  # (N, B, D), N = times of visualizing
            l2_norm = features.norm(p=2, dim=-1, keepdim=True)  # (N, B, 1)
            torch.save({"feature_norms": l2_norm}, join(self.model_path, "l2_norm.pt"))





def feature_visual(features: Union[torch.Tensor, np.ndarray], perplexity = 30):
    # features.shape = (B, D)
    if isinstance(features, torch.Tensor):
        if features.device != torch.device("cpu"):
            features = features.to("cpu")
    features = features.detach()
    tsne = TSNE(perplexity=perplexity, n_components=2)
    visual = tsne.fit_transform(features)
    return visual


def plot_epochwise_samples_with_label_edges(features: np.ndarray, labels: np.ndarray):
    """
    Visualize (B, N, 2) features with different colors for N epochs and edge color depth for labels.

    Parameters:
    -----------
    features : np.ndarray
        Feature array of shape (B, N, 2), where B is the number of samples and
        N is the number of epochs per sample.
    labels : np.ndarray
        Label array of shape (B,), label value determines edge color intensity.

    Returns:
    --------
    None
        Displays a matplotlib plot.
    """
    assert features.ndim == 3 and features.shape[2] == 2, "features should have shape (B, N, 2)"
    assert labels.ndim == 1 and labels.shape[0] == features.shape[0], "labels should have shape (B,)"

    B, N, _ = features.shape

    # colormap 设置
    colors_inner = cm.Accent(np.linspace(0, 10, N))
    # cmap_inner = cm.get_cmap('tab10', N)
    cmap_edge = cm.get_cmap('Greys')
    norm_edge = Normalize(vmin=labels.min(), vmax=labels.max())

    # 展平 label
    label_repeated = np.repeat(labels, N)

    # 创建画布
    fig, ax = plt.subplots(figsize=(8, 6))

    # 按 epoch 分组绘制
    for n in range(N):
        points = features[:, n, :]  # shape (B, 2)
        edge_colors = cmap_edge(labels)  # shape (B, 4)
        ax.scatter(
            points[:, 0], points[:, 1],
            c=[colors_inner[n]],  # 每组一个颜色
            edgecolors=edge_colors,
            linewidths=1.2,
            s=60,
            marker='o',
            label=f"Epoch {n + 1}"
        )

    # 添加 colorbar 显示 label 强度
    sm = plt.cm.ScalarMappable(cmap=cmap_edge, norm=norm_edge)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Label Intensity")

    ax.set_title("Visualization of Samples Across Epochs")
    ax.set_xlabel("Feature Dimension 1")
    ax.set_ylabel("Feature Dimension 2")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(title="Epoch")
    plt.tight_layout()
    return plt


if __name__ == '__main__':
    data = torch.load(r"/home/fuen/DeepLearningProjects/TimeSeriesProject/train/model_result/IMDSSN_dataFD002_cut10.3engine_cut20.03rul_cut30.8random_1/visual_data.pt")
    sample = data["tsne"]
    label = data["labels"]
    features = data["features"]
    _, N, _ = sample.shape
    for i in range(N):
        p = plot_epochwise_samples_with_label_edges(sample[:, i:i+1], label)
        p.show()
