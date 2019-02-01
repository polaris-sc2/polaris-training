"""
@file: layers.py
Created on 02.02.19
@project: LeelaStar
@author: kiudee

Groups the defined basic layer types used for the project
"""
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """Construct a squeeze excitation block for use in a residual block.

    This uses the squeeze excitation block as proposed by `Hu et al. 2017`_.

    .. _Hu et al. 2017:
        https://arxiv.org/abs/1709.01507

    """

    def __init__(self, channels, ratio):
        super().__init__()
        self.dense_linear_1 = nn.Linear(channels, channels // ratio)
        self.relu = nn.ReLU(inplace=True)
        self.dense_linear_2 = nn.Linear(channels // ratio, 2 * channels)

    def forward(self, x):
        x_before = x
        n, c, _, _ = x.size()

        x = F.adaptive_avg_pool2d(x, 1).view(n, c)
        x = self.dense_linear_1(x)
        x = self.relu(x)
        x = self.dense_linear_2(x)

        x = x.view(n, 2 * c, 1, 1)
        scale, shift = x.chunk(2, dim=1)

        x = scale.sigmoid() * x_before + shift
        return x


class ResidualBlock(nn.Sequential):
    """Construct a residual block to be used in a convolutional architecture.

    This class makes use of the squeeze excitation block to performs dynamic
    channel-wise feature recalibration.
    """

    def __init__(self, channels, kernel_size, se_ratio):
        super().__init__(
            dict(
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
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x_before = x
        x = super().forward(x)

        # Add skip connection and apply second relu
        x = x + x_before
        x = self.relu2(x)
        return x
