"""Groups the defined basic layer types used for the project"""
from collections import OrderedDict
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """Construct a squeeze excitation block for use in a residual block.

    This uses the squeeze excitation block as proposed by `Hu et al. 2017`_.
    A larger reduction ratio reduces computational cost, but may impact generalization performance.
    The generalization performance does not change monotonically with increasing reduction ratio.
    It is therefore recommended to treat it as a tunable hyperparameter.

    Parameters
    ----------
    channels : int
        Number of input channels
    ratio : int
        Reduction ratio of the bottleneck

    .. _Hu et al. 2017:
        https://arxiv.org/abs/1709.01507

    """

    def __init__(self, channels, ratio):
        super().__init__()
        self.dense_linear_1 = nn.Linear(channels, channels // ratio)
        self.dense_linear_2 = nn.Linear(channels // ratio, 2 * channels)

    def forward(self, x):
        x_before = x
        n, c, _, _ = x.size()

        x = F.adaptive_avg_pool2d(x, 1).view(n, c)
        x = self.dense_linear_1(x)
        x = F.relu(x, inplace=True)
        x = self.dense_linear_2(x)

        x = x.view(n, 2 * c, 1, 1)
        scale, shift = x.chunk(2, dim=1)

        x = scale.sigmoid() * x_before + shift
        return x


class ResidualBlock2D(nn.Sequential):
    """Construct a residual block to be used in a convolutional architecture.

    This class makes use of the squeeze excitation block to perform dynamic
    channel-wise feature recalibration.

    Parameters
    ----------
    channels : int
        Number of input channels
    kernel_size : int
        Size of the convolving kernels
    ratio : int
        Reduction ratio of the bottleneck

    """

    def __init__(self, channels, kernel_size, se_ratio):
        super().__init__(
            OrderedDict(
                [
                    ("conv_layer_1", nn.Conv2d(channels, channels, kernel_size, padding=1, bias=False)),
                    ("batch_norm_1", nn.BatchNorm2d(channels)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("conv_layer_2", nn.Conv2d(channels, channels, kernel_size, padding=1, bias=False)),
                    ("batch_norm_2", nn.BatchNorm2d(channels)),
                    ("squeeze_ex", SqueezeExcitation(channels, se_ratio)),
                ]
            )
        )

    def forward(self, x):
        x_before = x
        x = super().forward(x)

        # Add skip connection and apply second relu
        x = x + x_before
        x = F.relu(x, inplace=True)
        return x


class MultiHeadAttention(nn.Module):
    """Construct a multi-head attention block to be used in a Transformer architecture.

    Parameters
    ----------
    channels : int
        Number of input channels
    heads : int
        Number of attention heads
    reduction_ratio : int
        Reduction ratio for the attention channels: att = heads * channels // reduction_ratio

    .. _Battaglia et al. 2018:
        https://arxiv.org/abs/1806.01261

    """

    def __init__(self, channels, heads, reduction_ratio=1):
        super().__init__()
        self.n_heads = heads
        self.n_att_channels = heads * channels // reduction_ratio

        self.conv_in = nn.Conv1d(in_channels=channels, out_channels=3 * self.n_att_channels, kernel_size=1)
        self.conv_out = nn.Conv1d(in_channels=self.n_att_channels, out_channels=channels, kernel_size=1)

        self.scale = 1.0 / np.sqrt(channels // reduction_ratio)
        # Additionally account for Xavier/Glorot initialization:
        self.scale /= np.sqrt(2 / (channels + 3 * self.n_att_channels))
        self.scale *= np.sqrt(2 / (channels + self.n_att_channels))

    def forward(self, x):
        n, c, h, w = x.size()

        # First stack the feature planes into vectors:
        length = h * w
        x = x.view(n, c, length)

        # Apply 1d convolutions and scale to obtain Q, K and V:
        scaled = self.conv_in(x) * self.scale
        queries, keys, values = scaled.view(n, self.n_heads, -1, length).chunk(3, dim=2)

        # Compute attention weights through dot product & softmax:
        att_weights = queries.transpose(2, 3).matmul(keys).view(n, self.n_heads, -1)
        att_weights = F.softmax(att_weights, dim=2).view(n, self.n_heads, length, length)

        # Multiply attention weights with values and reshape to obtain original shape:
        output = self.conv_out(values.matmul(att_weights).view(n, -1, length)).view(n, c, h, w)
        return output
