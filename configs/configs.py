from train import BaseConfig, build_flag
import torch


FEI_STATE_PATH = "FEI_encResNet"

class TrainConfig(BaseConfig):
    def __init__(self):
        super(TrainConfig, self).__init__()
        self.batch_size = 64
        self.epoch = 300
        self.early_stopping = 0
        self.lr = 1e-3

        # # C-MAPSS dataset
        self.dataset = "FD002"
        self.window_size = 30
        self.in_features = 14
        self.data_cut = ['engine', 'rul', 'random']
        self.data_ratio = [0.3, 0.03, 0.8]
        self.sample_step = 15

        # Output Layer
        self.output_layer_mode = "norm"  # "norm"/"zero"

        # Bearing XJTU dataset
        # self.dataset = "OP_A"
        # self.window_size = 1024
        # self.in_features = 2
        # self.data_cut = ['rul', 'random']
        # self.data_ratio = [0.1, 0.1]
        # self.sample_step = 32768

        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        self.visual_epoch = 150
        self.feature_distance_epoch = 5


class DualMixerConfig(TrainConfig):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 32
        self.layer_num = 6
        self.dropout = 0.

    @property
    def model_flag(self):
        return build_flag("DualMixerRUL",
                          data=self.dataset,
                          cut=self.data_cut,
                          ratio=self.data_ratio,
                          step=self.sample_step)


class PretrainedCNNConfig(TrainConfig):
    def __init__(self):
        super().__init__()
        self.embed_dim = 64
        self.extc_dim = 128
        self.inner_dim = 16
        self.dynamic_token = False
        self.saved_stat = FEI_STATE_PATH
        # self.saved_stat = None
        self.token_num = 3

        self.fusing = False
        self.fine_tune = 'all'  # all/bias/false/input
        # Output Layer
        self.output_layer_mode = "norm"  # "norm"/"zero"

    @property
    def model_flag(self):
        return build_flag("FeiCNN",
                          data=self.dataset,
                          cut=self.data_cut,
                          ratio=self.data_ratio,
                          step=self.sample_step,
                          fuse=self.fusing,
                          fineTune=self.fine_tune,
                          pretrained=True if self.saved_stat is not None else False,
                          outpuT=self.output_layer_mode)


class RandomCNNConfig(PretrainedCNNConfig):
    def __init__(self):
        super().__init__()
        self.saved_stat = None
        # Output Layer
        self.output_layer_mode = "norm"  # "norm"/"zero"

    @property
    def model_flag(self):
        return build_flag("RandomCNN",
                          data=self.dataset,
                          cut=self.data_cut,
                          ratio=self.data_ratio,
                          step=self.sample_step,
                          fuse=self.fusing,
                          fineTune=self.fine_tune,
                          pretrained=False,
                          outpuT=self.output_layer_mode,)


class LinearPretrainedCNNConfig(PretrainedCNNConfig):
    def __init__(self):
        super().__init__()
        self.fine_tune = 'false'
        # Output Layer
        self.output_layer_mode = "norm"  # "norm"/"zero"

    @property
    def model_flag(self):
        return build_flag("LinearCNN",
                          data=self.dataset,
                          cut=self.data_cut,
                          ratio=self.data_ratio,
                          step=self.sample_step,
                          fuse=self.fusing,
                          fineTune=self.fine_tune,
                          pretrained=True,
                          outpuT=self.output_layer_mode,)


class PeftMuTSConfig(TrainConfig):
    def __init__(self):
        super().__init__()
        self.embed_dim = 64
        self.side_dim = [[self.embed_dim, 128, 1],
                         [128, 32, 1],
                         [128, 32, 1],
                         [256, 4, 1],
                         [256, 4, 1],
                         [512, 2, 1],
                         [512, 2, 1],
                         [1024, 1, 1]]  # The element is: [SideNet: in_dim, SideNet: hidden_dim]

        self.saved_stat = FEI_STATE_PATH
        # self.saved_stat = None
        self.fine_tune = False
        self.fusion = True
        # Output Layer
        self.output_layer_mode = "zero"  # "norm"/"zero"
        self.shift = False
        self.shift_init_factor = 1e-5

    @property
    def model_flag(self):
        return build_flag("ParamsTune_SideCNN",
                          data=self.dataset,
                          cut=self.data_cut,
                          ratio=self.data_ratio,
                          step=self.sample_step,
                          fineTune=self.fine_tune,
                          pretrained=True if self.saved_stat is not None else False,
                          fusion=self.fusion,
                          outpuT=self.output_layer_mode,
                          shift=self.shift_init_factor if self.shift else self.shift)

class PeftMuTSNoPretrainConfig(PeftMuTSConfig):
    def __init__(self):
        super().__init__()
        self.embed_dim = 64

        self.saved_stat = None
        self.fine_tune = False
        self.fusion = True


class DAMCNNConfig(TrainConfig):
    def __init__(self):
        super().__init__()
        # Output Layer
        self.output_layer_mode = "norm"  # "norm"/"zero"

    @property
    def model_flag(self):
        return build_flag("DAMCNN",
                          data=self.dataset,
                          cut=self.data_cut,
                          ratio=self.data_ratio,
                          outpuT=self.output_layer_mode,)


class IMDSSNConfig(TrainConfig):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 512
        self.encoder_nums = 1
        self.n_heads = 2
        self.pe = True
        # Output Layer
        self.output_layer_mode = "norm"  # "norm"/"zero"

    @property
    def model_flag(self):
        return build_flag("IMDSSN",
                          data=self.dataset,
                          cut=self.data_cut,
                          ratio=self.data_ratio,)
