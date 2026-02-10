# tests/unit/model/pytorch/test_wrapping.py

"""
These tests validate talos' core "wrapping" contract:

  - A user writes a plain `torch.nn.Module` (e.g., an MLP).
  - They wrap it via `ta.Model(nn_model)` (or equivalently `TorchModel(model=nn_model)`).
  - The wrapped object should still behave like an `nn.Module`:
      * it owns parameters via `parameters()` / `state_dict()`
      * train()/eval() toggles propagate to the wrapped module
      * forward works and produces correctly-shaped output

This is intentionally a lightweight unit test: it does not train, save, or use CUDA.
"""

import pytest
import torch
from torch import nn

import talos as ta
from examples.model.pytorch.init_from_torch_module import MLP


def test_wraps_nn_module_and_preserves_module_semantics():
  """Ensure wrapping a user `nn.Module` preserves core `nn.Module` semantics."""
  # 1) Arrange: create a vanilla torch module similar to what a user would write.
  nn_model = MLP(input_size=32 * 32 * 3, hidden_size=16, num_classes=10)

  # 2) Act: wrap via talos' public API.
  model = ta.Model(nn_model)

  # 3) Assert: wrapper should expose core nn.Module surface area.
  # A) It should be an nn.Module itself (or at least behave as one).
  assert isinstance(model, nn.Module)

  # B) It should contain the wrapped module as a registered child module.
  # This ensures parameters/buffers are discoverable via state_dict/parameters.
  children = dict(model.named_children())
  assert "_model" in children
  assert children["_model"] is nn_model

  # C) It should have parameters and a non-empty state_dict consistent with torch modules.
  params = list(model.parameters())
  assert len(params) > 0
  sd = model.state_dict()
  assert isinstance(sd, dict)
  assert len(sd) > 0

  # D) train()/eval() should propagate to the wrapped module.
  model.eval()
  assert model.training is False
  assert nn_model.training is False

  model.train()
  assert model.training is True
  assert nn_model.training is True

  # E) Forward pass: check the wrapper can run inference and shape matches expectation.
  x = torch.randn(4, 32 * 32 * 3)
  y = model(x)
  assert isinstance(y, torch.Tensor)
  assert y.shape == (4, 10)


def test_state_dict_keys_are_prefixed_by_wrapper_child_name():
  """Ensure wrapper state_dict keys reflect the registered child module name."""
  # This test documents an important implication of wrapping: since the user module
  # becomes a child module named "_model", parameters are typically stored under
  # the "_model." prefix in the wrapper state_dict.
  nn_model = MLP(input_size=8, hidden_size=4, num_classes=3)
  model = ta.Model(nn_model)

  keys = list(model.state_dict().keys())
  assert any(k.startswith("_model.") for k in keys), (
    "Expected wrapper state_dict keys to be prefixed with '_model.' because "
    "the wrapped nn.Module is registered as child module '_model'."
  )


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_forward_accepts_expected_input_rank(batch_size):
  """Ensure the wrapper forwards rank-2 inputs [batch, features] correctly."""
  # This MLP expects rank-2 input: [batch, features].
  nn_model = MLP(input_size=12, hidden_size=6, num_classes=5)
  model = ta.Model(nn_model)

  x = torch.randn(batch_size, 12)
  y = model(x)
  assert y.shape == (batch_size, 5)
