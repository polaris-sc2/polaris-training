"""Groups the defined basic layer types used for the project"""
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


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
