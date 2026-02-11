"""Gradient-based, backend-agnostic trainer."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from talos.data.talos_data import TalosData
from talos.model.talos_model import TalosModel
from talos.utils import Nomear


class TalosTrainer(Nomear):
  """A base class for gradient-based trainers."""

  SCOPE = "talos.trainer"

  def __init__(self, model: TalosModel, optimizer: Any = 'sgd',
               loss_fn=None, **optimizer_configs):
    """Train a model on a dataset using a specified optimizer.

    Args:
      model: The model to be trained.
      optimizer: Optimizer specification. Supported forms:
        (1) string name (e.g., "adam")
        (2) optimizer class (backend specific)
        (3) optimizer instance (backend specific)
      loss_fn: Loss function specification (string, class, or instance).
        If provided, stored as the first entry in self.loss_functions.
      **optimizer_configs: Optimizer kwargs (e.g., lr=0.01).
    """
    # (1) Core training components.
    self.model: TalosModel = model
    self.optimizer = optimizer

    # (2) Ensure `self.optimizer` is ready for the training loop.
    self._validate_optimizer(**optimizer_configs)

    # (3) Register initial loss function if provided.
    if loss_fn is not None:
      loss_instance = self._resolve_metric(loss_fn)
      self.loss_functions[loss_instance.name] = loss_instance

  # region: Properties

  @Nomear.property()
  def loss_functions(self): return OrderedDict()

  # endregion: Properties

  # region: APIs

  def train(self, train_set: TalosData, max_iterations, batch_size=-1,
            loss_fn=None, *args, **kwargs):
    """Train the model on a dataset.

    Args:
      train_set: Training dataset.
      max_iterations: Number of training iterations.
      batch_size: Batch size. -1 for full batch (default).
      loss_fn: Train-time loss override (takes priority over self.loss_functions).
    """
    # (0) Resolve loss function.
    loss_fn = self._get_loss_function(loss_fn)

    for i in range(max_iterations):
      # (1) Sample data batch and convert to backend format.
      batch = train_set.sample(batch_size)
      X, Y = self._prepare_batch(batch.X, batch.Y)

      # (2) Forward pass.
      outputs = self.model.forward(X)

      # (3) Compute loss.
      loss = loss_fn(outputs, Y)

      # (4) Backward pass + parameter update (backend-specific).
      self._backward_and_update(loss)

  # endregion: APIs

  # region: Backend-Specific Methods

  def _validate_optimizer(self, **configs) -> None:
    """Validate/resolve optimizer so `self.optimizer` is ready for the training loop."""
    raise NotImplementedError

  def _prepare_batch(self, X, Y):
    """Convert batch data to backend-specific format. Default: pass-through."""
    return X, Y

  def _backward_and_update(self, loss) -> None:
    """Compute gradients and update model parameters."""
    raise NotImplementedError

  def _resolve_metric(self, spec):
    """Resolve metric spec into a metric instance (backend-specific)."""
    raise NotImplementedError

  # endregion: Backend-Specific Methods

  # region: Utilities

  def _get_loss_function(self, spec=None):
    """Get loss function for training.

    Priority: (1) train-time arg → (2) self.loss_functions 1st entry.
    """
    # (1) Train-time override.
    if spec is not None:
      return self._resolve_metric(spec)
    # (2) First entry in loss_functions.
    if self.loss_functions:
      return next(iter(self.loss_functions.values()))
    raise ValueError(
      "No loss function specified. Provide loss_fn at init or train time.")

  def _normalize_optimizer_spec(self) -> tuple[str | None, type | None, Any | None]:
    """Normalize `self.optimizer` into (name, cls, instance)."""
    # (1) Name.
    if isinstance(self.optimizer, str):
      name = self.optimizer.strip().lower()
      if not name:
        raise ValueError("!! optimizer name must be a non-empty string")
      return name, None, None
    # (2) Class.
    if isinstance(self.optimizer, type):
      return None, self.optimizer, None
    # (3) Instance.
    return None, None, self.optimizer

  # endregion: Utilities
