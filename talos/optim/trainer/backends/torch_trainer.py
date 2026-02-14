"""PyTorch backend trainer."""

from __future__ import annotations

from typing import Any

import numpy as np

from talos.eval.backends.torch_metric import get_torch_metric
from talos.model.backends.pytorch.torch_model import TorchModel
from talos.optim.trainer.talos_trainer import TalosTrainer
from talos.utils.backends.pytorch import has_torch, torch


class TorchTrainer(TalosTrainer):
  """A gradient-based trainer for PyTorch models."""
  model: TorchModel

  def __init__(self, model: TorchModel, optimizer: Any = "sgd",
               loss_fn=None, **kwargs):
    """Initialize the PyTorch trainer."""
    super().__init__(model=model, optimizer=optimizer, loss_fn=loss_fn, **kwargs)

  # region: Backend-Specific Methods

  def _validate_optimizer(self, **configs) -> None:
    """Resolve optimizer spec so `self.optimizer` is a training-ready torch optimizer."""
    # (1) Backend availability.
    if not has_torch:
      raise ImportError("PyTorch is not installed. TorchTrainer is unavailable.")

    # (2) Normalize optimizer specification.
    name, cls, instance = self._normalize_optimizer_spec()

    # (3) Optimizer instance → validate and use directly.
    if instance is not None:
      if not isinstance(instance, torch.optim.Optimizer):
        raise TypeError("!! optimizer instance must be a torch.optim.Optimizer")
      self.optimizer = instance
      return

    # (4) Resolve optimizer class (from class or name).
    if cls is not None:
      if not issubclass(cls, torch.optim.Optimizer):
        raise TypeError("!! optimizer class must subclass torch.optim.Optimizer")
      opt_cls = cls
    else:
      registry = {
        "adadelta": torch.optim.Adadelta,
        "adagrad": torch.optim.Adagrad,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
        "sgd": torch.optim.SGD,
      }
      opt_cls = registry.get(name)
      if opt_cls is None:
        supported = ", ".join(sorted(registry.keys()))
        raise ValueError(f"!! unknown optimizer '{name}', supported: {supported}")

    # (5) Instantiate optimizer for the training loop.
    self.optimizer = opt_cls(self.model.parameters(), **configs)

  def _prepare_batch(self, X, Y):
    """Convert numpy arrays to torch tensors on the model's device."""
    device = next(self.model.parameters()).device
    X = torch.tensor(X, dtype=torch.float32, device=device)
    if Y is not None:
      Y = torch.tensor(Y, dtype=torch.float32, device=device)
    return X, Y

  def _backward_and_update(self, loss) -> None:
    """Zero gradients, compute gradients, and update parameters."""
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

  def _resolve_metric(self, spec):
    """Resolve metric spec using torch metric registry."""
    return get_torch_metric(spec)

  def _validate(self, val_set, val_metrics, iteration):
    """Run validation with model.eval() and torch.no_grad()."""
    self.model.eval()
    with torch.no_grad():
      super()._validate(val_set, val_metrics, iteration)
    self.model.train()

  # endregion: Backend-Specific Methods


if __name__ == "__main__":
  import talos as ta

  # Data
  N, D_in, D_out = 100, 10, 1  # Configure dataset
  X, Y = np.random.rand(N, D_in), np.random.rand(N, D_out)
  train_set, test_set = ta.Dataset(X, Y, 'Dummy').split(1, 1)
  train_set.report(), test_set.report()

  # Model
  hidden_features = [32, 32]  # Configure model architecture
  model = ta.model.torch_zoo.MLP(D_in, hidden_features, D_out)
  model.summary(D_in)

  # Optimization
  trainer = TorchTrainer(model, loss_fn='mse')
  trainer.config.print_every = 20
  trainer.config.validate_every = 100
  # trainer.config.val_ratio = 0.1
  # trainer.config.patience = 3
  trainer.train(train_set, max_iterations=500)
