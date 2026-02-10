"""Tests for building models by subclassing talos' TorchModule.

These tests validate a common workflow where a user defines a model that
inherits from `ta.TorchModule` directly (rather than wrapping an existing
`torch.nn.Module`). The resulting object should behave like a standard
`torch.nn.Module` with respect to:
  - parameter/state_dict registration
  - train()/eval() behavior
  - forward pass correctness (basic shape checks)
"""

import pytest
import torch
from torch import nn

from examples.model.pytorch.init_as_talos_model import MLP


def test_subclassed_torchmodule_behaves_like_nn_module():
    """Ensure a TorchModule subclass exposes core nn.Module semantics."""
    model = MLP(input_size=32 * 32 * 3, hidden_size=16, num_classes=10)

    assert isinstance(model, nn.Module)

    params = list(model.parameters())
    assert len(params) > 0

    sd = model.state_dict()
    assert isinstance(sd, dict)
    assert len(sd) > 0

    model.eval()
    assert model.training is False

    model.train()
    assert model.training is True

    x = torch.randn(4, 32 * 32 * 3)
    y = model(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape == (4, 10)


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_forward_accepts_expected_input_rank(batch_size):
    """Ensure the model forwards rank-2 inputs [batch, features] correctly."""
    model = MLP(input_size=12, hidden_size=6, num_classes=5)

    x = torch.randn(batch_size, 12)
    y = model(x)
    assert y.shape == (batch_size, 5)
