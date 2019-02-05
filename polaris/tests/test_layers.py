import pytest
import torch

from polaris.network.layers import SqueezeExcitation


def test_squeeze_excitation():
    X = torch.tensor([[[[1.0, 2, 3], [4, 5, 6], [7, 8, 9]]]])
    se = SqueezeExcitation(channels=1, ratio=1)
    se.dense_linear_1.weight.data = torch.tensor([[4.0]])
    se.dense_linear_1.bias.data = torch.tensor([[2.0]])
    se.dense_linear_2.weight.data = torch.tensor([[-0.1], [2.0]])
    se.dense_linear_2.bias.data = torch.tensor([0.1, -3])

    output = se(X)
    expected = torch.tensor([[[[41.109, 41.218, 41.327], [41.436, 41.545, 41.655], [41.764, 41.873, 41.982]]]])
    assert pytest.approx(expected.detach().numpy(), abs=1e-3) == output.detach().numpy()
