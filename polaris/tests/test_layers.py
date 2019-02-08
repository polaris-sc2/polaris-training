""" Group all tests cases for layers"""

import pytest
import torch

from polaris.network.layers import SqueezeExcitation, ResidualBlock2D


def test_squeeze_excitation():
    X = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])
    se = SqueezeExcitation(channels=1, ratio=1)
    se.dense_linear_1.weight.data = torch.tensor([[4.0]])
    se.dense_linear_1.bias.data = torch.tensor([[2.0]])
    se.dense_linear_2.weight.data = torch.tensor([[-0.1], [2.0]])
    se.dense_linear_2.bias.data = torch.tensor([0.1, -3])

    output = se(X)
    expected = torch.tensor([[[[41.109, 41.218, 41.327], [41.436, 41.545, 41.655], [41.764, 41.873, 41.982]]]])
    assert pytest.approx(expected.detach().numpy(), abs=1e-3) == output.detach().numpy()


def test_residual_block():
    X = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])
    rb = ResidualBlock2D(channels=1, kernel_size=3, se_ratio=1)
    rb.conv_layer_1.weight.data = torch.tensor([[[[0.0, 1, 0.0], [1, 2, 1], [0.0, 1, 0.0]]]])
    rb.conv_layer_2.weight.data = torch.tensor([[[[0.0, 1, 0.0], [1, 1, 1], [0.0, 1, 0.0]]]])
    rb.batch_norm_1.weight.data = torch.tensor([0.1])
    rb.batch_norm_2.weight.data = torch.tensor([1.0])
    rb.squeeze_ex.dense_linear_1.weight.data = torch.tensor([[0.0]])
    rb.squeeze_ex.dense_linear_1.bias.data = torch.tensor([[0.0]])
    rb.squeeze_ex.dense_linear_2.weight.data = torch.tensor([[1.0], [1.0]])
    rb.squeeze_ex.dense_linear_2.bias.data = torch.tensor([1.0, 0.0])

    output = rb(X)
    expected = torch.tensor([[[[0.000, 1.351, 2.282], [3.535, 5.685, 6.340], [7.018, 9.076, 9.823]]]])
    assert pytest.approx(expected.detach().numpy(), abs=1e-3) == output.detach().numpy()
