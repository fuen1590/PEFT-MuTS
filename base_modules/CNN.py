import torch
import torch.nn as nn


def get_output_size(resnet_size: str):
    if resnet_size == "tiny":
        return 128
    elif resnet_size == "small":
        return 256
    elif resnet_size == "norm":
        return 512
    else:
        return 1024

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, out_act=True):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=kernel_size // 2,
                      bias=False),
            nn.BatchNorm1d(out_channel),
            nn.GELU(),
            nn.Conv1d(in_channels=out_channel,
                      out_channels=out_channel,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=kernel_size // 2,
                      bias=False),
            nn.BatchNorm1d(out_channel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels=in_channel,
                          out_channels=out_channel,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm1d(out_channel),
            )
        self.out_act = nn.GELU() if out_act else None

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = self.out_act(out) if self.out_act is not None else out
        return out


class InputEmbedding(nn.Module):
    """
    This block processing the time series to the embedding space, as a standard input layer of ResNet.
    The input size should be (Batch, Length, in_channel).
    """
    def __init__(self, in_channels, out_channels):
        super(InputEmbedding, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        class Transpose(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(-1, -2)

        self.embedding = nn.Sequential(
            Transpose(),
            nn.Conv1d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(self.out_channels),
            nn.GELU(),
            nn.AvgPool1d(2, 2),
            Transpose()
        )

    def forward(self, x):
        """

        :param x: time series with shape (Batch, Length, Channel)
        :return: embeddings with shape (Batch, L/2, in_channel)
        """
        return self.embedding(x)


class ResNet(nn.Module):
    def __init__(self,
                 in_channels=16,
                 size="tiny",
                 stride=2):
        super(ResNet, self).__init__()
        self.in_channel = in_channels
        if size == "tiny":
            self.layer1 = self.make_layer(ResBlock, 16, 2, stride=stride)
            self.layer2 = self.make_layer(ResBlock, 32, 2, stride=stride)
            self.layer3 = self.make_layer(ResBlock, 64, 2, stride=stride)
            self.layer4 = self.make_layer(ResBlock, 128, 2, stride=stride, out_act=True)
        elif size == "small":
            self.layer1 = self.make_layer(ResBlock, 32, 2, stride=stride)
            self.layer2 = self.make_layer(ResBlock, 64, 2, stride=stride)
            self.layer3 = self.make_layer(ResBlock, 128, 2, stride=stride)
            self.layer4 = self.make_layer(ResBlock, 256, 2, stride=stride, out_act=True)
        elif size == "norm":
            self.layer1 = self.make_layer(ResBlock, 64, 2, stride=stride)
            self.layer2 = self.make_layer(ResBlock, 128, 2, stride=stride)
            self.layer3 = self.make_layer(ResBlock, 256, 2, stride=stride)
            self.layer4 = self.make_layer(ResBlock, 512, 2, stride=stride, out_act=True)
        elif size == "big":
            self.layer1 = self.make_layer(ResBlock, 128, 2, stride=stride)
            self.layer2 = self.make_layer(ResBlock, 256, 2, stride=stride)
            self.layer3 = self.make_layer(ResBlock, 512, 2, stride=stride)
            self.layer4 = self.make_layer(ResBlock, 1024, 2, stride=stride, out_act=True)
        elif size == "large-50":
            self.layer1 = self.make_layer(ResBlock, 128, 3, stride=stride)
            self.layer2 = self.make_layer(ResBlock, 256, 4, stride=stride)
            self.layer3 = self.make_layer(ResBlock, 512, 6, stride=stride)
            self.layer4 = self.make_layer(ResBlock, 1024, 3, stride=stride, out_act=True)
        elif size == "large-101":
            self.layer1 = self.make_layer(ResBlock, 128, 3, stride=stride)
            self.layer2 = self.make_layer(ResBlock, 256, 4, stride=stride)
            self.layer3 = self.make_layer(ResBlock, 512, 23, stride=stride)
            self.layer4 = self.make_layer(ResBlock, 1024, 3, stride=stride, out_act=True)
        else:
            raise NotImplementedError("Unsupported ResNet size: {}".format(size))

    def make_layer(self, block, channels, num_blocks, stride, out_act=True):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channels, stride=stride, out_act=out_act))
            self.in_channel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: input series with shape (Batch, Length, in_channel).
        """
        x = x.transpose(-1, -2)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.transpose(-1, -2)
        return out

    def forward_per_layers(self, x):
        """
        :param x: input series with shape (Batch, Length, in_channel).

        :return: output, features from every layer.
        """
        x = x.transpose(-1, -2)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        out = f4.transpose(-1, -2)
        return  out, [f1.transpose(-1, -2), f2.transpose(-1, -2), f3.transpose(-1, -2), f4.transpose(-1, -2)]

    def get_layers(self):
        layers = []
        for block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for layer in block:
                layers.append(layer)
        return layers


if __name__ == '__main__':
    model1 = ResNet(in_channels=64, size="big", stride=2)
    saved = torch.load(r"/home/fuen/DeepLearningProjects/TimeSeriesProject/train/model_result/FMEI_encResNet_mMin0.0_mMax0.7_mom995_inDim64/model_ck.pt")
    params = saved["encoder"]
    model1.load_state_dict(params)
    x = torch.randn((1, 20, 64))
    y = model1(x)
