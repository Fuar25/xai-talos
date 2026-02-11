"""Backend-agnostic, numpy-based metrics."""

from __future__ import annotations

import numpy as np


class TalosMetric:
  """Base class for all metrics.

  Attributes:
    name: Metric identifier (e.g., 'mse', 'cross_entropy').
    differentiable: Whether the metric can be used as a training loss.
    direction: 'minimize' or 'maximize' — used by alchemy and early stopping.
  """

  name: str = ''
  differentiable: bool = False
  direction: str = 'minimize'

  def __call__(self, outputs, targets):
    """Compute metric using numpy."""
    return self._np_compute(outputs, targets)

  def _np_compute(self, outputs, targets):
    """Core numpy computation. Override in concrete metrics."""
    raise NotImplementedError

  def __repr__(self):
    return f"{self.__class__.__name__}(direction='{self.direction}')"


# region: Concrete Metrics

class MSE(TalosMetric):
  """Mean Squared Error: mean((y_hat - y)^2)."""

  name = 'mse'
  differentiable = True
  direction = 'minimize'

  def _np_compute(self, outputs, targets):
    return np.mean((outputs - targets) ** 2)


class MAE(TalosMetric):
  """Mean Absolute Error: mean(|y_hat - y|)."""

  name = 'mae'
  differentiable = True
  direction = 'minimize'

  def _np_compute(self, outputs, targets):
    return np.mean(np.abs(outputs - targets))


class CrossEntropy(TalosMetric):
  """Cross Entropy for multi-class classification.

  Expects outputs as logits (N, C) and targets as class indices (N,).
  """

  name = 'cross_entropy'
  differentiable = True
  direction = 'minimize'

  def _np_compute(self, outputs, targets):
    # (1) Softmax for numerical stability.
    shifted = outputs - np.max(outputs, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=-1, keepdims=True)
    # (2) Negative log-likelihood.
    n = outputs.shape[0]
    return -np.mean(np.log(probs[np.arange(n), targets.astype(int)] + 1e-12))


class BinaryCrossEntropy(TalosMetric):
  """Binary Cross Entropy.

  Expects outputs as logits (N,) and targets as 0/1 labels (N,).
  """

  name = 'bce'
  differentiable = True
  direction = 'minimize'

  def _np_compute(self, outputs, targets):
    # (1) Sigmoid.
    probs = 1.0 / (1.0 + np.exp(-outputs))
    # (2) Binary cross entropy.
    eps = 1e-12
    return -np.mean(
      targets * np.log(probs + eps) + (1 - targets) * np.log(1 - probs + eps))


class Accuracy(TalosMetric):
  """Classification accuracy.

  Expects outputs as logits/probs (N, C) and targets as class indices (N,).
  """

  name = 'accuracy'
  differentiable = False
  direction = 'maximize'

  def _np_compute(self, outputs, targets):
    preds = np.argmax(outputs, axis=-1)
    return np.mean(preds == targets.astype(int))

# endregion: Concrete Metrics
