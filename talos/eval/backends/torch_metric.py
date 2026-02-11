"""PyTorch backend metrics."""

from __future__ import annotations

from talos.eval.talos_metric import (
  TalosMetric, MSE, MAE, CrossEntropy, BinaryCrossEntropy, Accuracy)
from talos.utils.backends.pytorch import has_torch, torch


class TorchMetric(TalosMetric):
  """Base class for torch metrics. Adds numpy() back door."""

  def numpy(self, outputs, targets):
    """Compute metric using numpy (back door).

    Accepts both numpy arrays and torch tensors — tensors are auto-converted.
    """
    if has_torch:
      if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
      if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    return self._np_compute(outputs, targets)


# region: Concrete Torch Metrics

class TorchMSE(TorchMetric, MSE):
  """Mean Squared Error (PyTorch)."""

  def __call__(self, outputs, targets):
    return torch.mean((outputs - targets) ** 2)


class TorchMAE(TorchMetric, MAE):
  """Mean Absolute Error (PyTorch)."""

  def __call__(self, outputs, targets):
    return torch.mean(torch.abs(outputs - targets))


class TorchCrossEntropy(TorchMetric, CrossEntropy):
  """Cross Entropy (PyTorch). Uses torch.nn.functional for numerical stability."""

  def __call__(self, outputs, targets):
    return torch.nn.functional.cross_entropy(outputs, targets.long())


class TorchBinaryCrossEntropy(TorchMetric, BinaryCrossEntropy):
  """Binary Cross Entropy (PyTorch)."""

  def __call__(self, outputs, targets):
    return torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)


class TorchAccuracy(TorchMetric, Accuracy):
  """Classification Accuracy (PyTorch). Non-differentiable."""

  def __call__(self, outputs, targets):
    preds = torch.argmax(outputs, dim=-1)
    return (preds == targets.long()).float().mean()

# endregion: Concrete Torch Metrics


# region: Registry

TORCH_METRIC_REGISTRY = {
  'mse': TorchMSE,
  'mae': TorchMAE,
  'cross_entropy': TorchCrossEntropy,
  'bce': TorchBinaryCrossEntropy,
  'accuracy': TorchAccuracy,
}

# endregion: Registry


# region: APIs

def get_torch_metric(spec) -> TorchMetric:
  """Resolve a metric specification into a TorchMetric instance.

  Args:
    spec: Metric specification. Supported forms:
      (1) string name (e.g., 'mse') → registry lookup
      (2) TorchMetric subclass → instantiate
      (3) TorchMetric instance → return as-is
  """
  # (1) String → registry lookup.
  if isinstance(spec, str):
    name = spec.strip().lower()
    cls = TORCH_METRIC_REGISTRY.get(name)
    if cls is None:
      supported = ', '.join(sorted(TORCH_METRIC_REGISTRY.keys()))
      raise ValueError(f"Unknown metric '{name}'. Supported: {supported}")
    return cls()
  # (2) Class → instantiate.
  if isinstance(spec, type) and issubclass(spec, TorchMetric):
    return spec()
  # (3) Instance → return as-is.
  if isinstance(spec, TorchMetric):
    return spec
  raise TypeError(
    f"Expected metric name (str), TorchMetric class, or TorchMetric instance, "
    f"got {type(spec).__name__}")

# endregion: APIs
