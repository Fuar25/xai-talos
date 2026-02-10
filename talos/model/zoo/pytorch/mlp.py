from __future__ import annotations

from collections.abc import Callable

import talos as ta
import torch
from torch import nn



class MLP(ta.TorchModule):
  """A configurable multi-layer perceptron (MLP).

  Supports optional activation, normalization, dropout, and residual connections.
  Blocks can be post-activation (Linear→Act→Norm) or pre-activation (Act→Norm→Linear).
  """

  def __init__(
      self,
      in_features: int,
      hidden_features: list[int] | tuple[int, ...],
      out_features: int,
      *,
      activation: str | nn.Module | Callable = "relu",
      dropout: float = 0.0,
      norm: str | None = None,
      bias: bool = True,
      activate_last: bool = False,
      last_activation: str | nn.Module | Callable | None = None,
      preact: bool = False,
      residual: bool = False,
  ):
    """Initialize an MLP.

    Args:
      in_features: Input feature dimension.
      hidden_features: Hidden layer sizes.
      out_features: Output feature dimension.
      activation: Activation for hidden layers (string, module, or callable).
      dropout: Dropout probability (applied only when an activation is present).
      norm: Normalization type: "bn", "ln", or None.
      bias: Whether Linear layers use bias.
      activate_last: Whether to apply activation/norm/dropout to the final layer.
      last_activation: Activation for final layer if `activate_last` is True.
      preact: If True, use Act→Norm→Linear blocks; else Linear→Act→Norm.
      residual: If True, add residual connections where shapes match.
    """
    super().__init__()

    self.in_features = int(in_features)
    self.hidden_features = list(hidden_features)
    self.out_features = int(out_features)
    self.dropout = float(dropout)
    self.norm = norm
    self.bias = bool(bias)
    self.activate_last = bool(activate_last)
    self.preact = bool(preact)
    self.residual = bool(residual)

    act = self._make_activation(activation)
    last_act = self._make_activation(last_activation) if last_activation is not None else act

    self._dims = [self.in_features] + self.hidden_features + [self.out_features]
    blocks: list[nn.Module] = []
    self._layer_specs: list[tuple[bool, bool]] = []

    for layer_i in range(len(self._dims) - 1):
      in_dim = self._dims[layer_i]
      out_dim = self._dims[layer_i + 1]
      is_last = (layer_i == len(self._dims) - 2)

      has_act = (not is_last) or self.activate_last
      layer_act = (last_act if is_last else act) if has_act else None

      linear = nn.Linear(in_dim, out_dim, bias=self.bias)
      # (1) BatchNorm must match feature size of the tensor it sees.
      norm_features = in_dim if self.preact else out_dim
      layer_norm = self._make_norm(self.norm, norm_features) if self.norm else None

      blocks.extend(self._layer_block(linear, layer_act, layer_norm))
      self._layer_specs.append((bool(layer_act is not None), bool(layer_norm is not None)))

    self.net = nn.Sequential(*blocks)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Run the MLP forward pass."""
    idx = 0

    for layer_i in range(len(self._dims) - 1):
      residual_in = x
      has_act, has_norm = self._layer_specs[layer_i]

      # (1) Keep `idx` consumption aligned with `_layer_block` order.
      if self.preact:
        if has_act:
          x = self.net[idx](x)
          idx += 1
        if has_norm:
          x = self.net[idx](x)
          idx += 1
        x = self.net[idx](x)
        idx += 1
        if self.dropout > 0.0 and has_act:
          x = self.net[idx](x)
          idx += 1
      else:
        x = self.net[idx](x)
        idx += 1
        if has_act:
          x = self.net[idx](x)
          idx += 1
        if has_norm:
          x = self.net[idx](x)
          idx += 1
        if self.dropout > 0.0 and has_act:
          x = self.net[idx](x)
          idx += 1

      if self.residual and x.shape == residual_in.shape:
        x = x + residual_in

    return x

  def _make_activation(self, activation: str | nn.Module | Callable) -> nn.Module:
    """Create an activation module from a string, module, or callable."""
    if isinstance(activation, nn.Module):
      return activation
    if isinstance(activation, str):
      name = activation.lower()
      if name == "relu":
        return nn.ReLU()
      if name == "gelu":
        return nn.GELU()
      if name in {"silu", "swish"}:
        return nn.SiLU()
      if name == "tanh":
        return nn.Tanh()
      if name == "sigmoid":
        return nn.Sigmoid()
      if name == "leaky_relu":
        return nn.LeakyReLU()
      raise ValueError(f"Unknown activation: {activation}")
    if callable(activation):
      module = activation()
      if isinstance(module, nn.Module):
        return module

      class _CallableActivation(nn.Module):
        """Wrap a callable activation into an nn.Module."""
        def __init__(self, fn):
          super().__init__()
          self._fn = fn

        def forward(self, x):
          return self._fn(x)

      return _CallableActivation(activation)
    raise TypeError("activation must be a str, nn.Module, or callable")

  def _make_norm(self, norm: str, features: int) -> nn.Module:
    """Create a normalization module."""
    name = norm.lower()
    if name == "bn":
      return nn.BatchNorm1d(features)
    if name == "ln":
      return nn.LayerNorm(features)
    raise ValueError(f"Unknown norm: {norm}")

  def _layer_block(
      self,
      linear: nn.Module,
      act: nn.Module | None,
      norm: nn.Module | None,
  ) -> list[nn.Module]:
    """Build one layer block in the configured pre/post-activation order."""
    modules: list[nn.Module] = []
    if self.preact:
      if act is not None:
        modules.append(act)
      if norm is not None:
        modules.append(norm)
      modules.append(linear)
    else:
      modules.append(linear)
      if act is not None:
        modules.append(act)
      if norm is not None:
        modules.append(norm)
    if self.dropout > 0.0 and act is not None:
      modules.append(nn.Dropout(self.dropout))
    return modules


if __name__ == "__main__":
  mlp = ta.model.torch_zoo.MLP(
    784, [128, 64, 64], 10,
    activation='gelu',
    residual=True,
  )

  mlp.summary(784)
  print(mlp)