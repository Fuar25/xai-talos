"""Gradient-based, backend-agnostic trainer."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from talos.data.talos_data import TalosData
from talos.model.talos_model import TalosModel
from talos.utils import Nomear
from talos.utils.config import Config as ConfigBase
from talos.utils.console import Console
from talos.optim.trainer.history import TrainingHistory

_console = Console()


class TrainState:
  """Per-session training state. Created fresh for each train() call."""

  def __init__(self):
    # (1) Early stopping state.
    self.early_stop_metric = None
    self.patience_counter = 0
    # (2) Checkpointing state.
    self.best_checkpoint = None


class TalosTrainer(Nomear):
  """A base class for gradient-based trainers."""

  SCOPE = "talos.trainer"

  class Config(ConfigBase):
    batch_size: int = ConfigBase.Integer(
      default=-1, description='Batch size. -1 for full batch.')
    max_iterations: int | None = ConfigBase.Integer(
      default=None, description='Number of training iterations.', positive=True)
    early_stop: bool = ConfigBase.Boolean(
      default=False, description='Enable early stopping.')
    patience: int = ConfigBase.Integer(
      default=10, description='Early stopping patience.', positive=True)
    validate_every: int = ConfigBase.Integer(
      default=100, description='Validate every N iterations.', positive=True)
    val_ratio: float | None = ConfigBase.Float(
      default=None, description='Auto-split ratio for validation set.',
      positive=True)
    val_metrics: str | None = ConfigBase.String(
      default=None,
      description='Comma/semicolon-separated metric names for validation.')
    print_every: int = ConfigBase.Integer(
      default=100, description='Print training progress every N iterations.',
      positive=True)
    save_best: bool = ConfigBase.Boolean(
      default=True, description='Save best model weights during validation.')

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

    # (4) Initialize per-session state (created fresh for each train() call).
    self._state = TrainState()

  # region: Properties

  @Nomear.property()
  def loss_functions(self): return OrderedDict()

  @Nomear.property()
  def config(self):
    return type(self).Config(name='trainer')

  @Nomear.property()
  def history(self): return TrainingHistory()

  # endregion: Properties

  # region: APIs

  def train(self, train_set: TalosData, max_iterations=None, batch_size=None,
            loss_fn=None, val_set: TalosData = None, val_metrics=None,
            *args, **kwargs):
    """Train the model on a dataset.

    Args:
      train_set: Training dataset.
      max_iterations: Number of training iterations. Falls back to config.
      batch_size: Batch size. -1 for full batch. Falls back to config.
      loss_fn: Train-time loss override (takes priority over self.loss_functions).
      val_set: Validation dataset. If not provided but config.val_ratio is set,
        auto-splits from train_set.
      val_metrics: Validation metrics (list of metric specs or instances).
        First metric is used for early stopping. Falls back to config.val_metrics
        string, then to [loss_fn].
    """
    # (0.1) Resolve parameters: signature args (priority) → config → error.
    max_iterations = self._resolve_param(
      'max_iterations', max_iterations, required=True)
    batch_size = self._resolve_param('batch_size', batch_size)
    validate_every = self.config.validate_every

    # (0.2) Resolve loss function.
    loss_fn = self._get_loss_function(loss_fn)

    # (0.3) Initialize per-session training state.
    self._state = TrainState()

    # (0.4) Resolve validation set and metrics.
    train_set, val_set = self._resolve_val_set(train_set, val_set)
    val_metrics = self._resolve_val_metrics(val_metrics, loss_fn)

    # (0.5) Check for config warnings.
    self._check_config_warnings(val_set)

    # (0.6) Resolve print frequency.
    print_every = self.config.print_every
    _console.show_status(f'Training started (max_iterations={max_iterations})')

    stopped_early = False
    for i in range(max_iterations):
      # (1) Sample data batch and convert to backend format.
      batch = train_set.sample(batch_size)
      X, Y = self._prepare_batch(batch.X, batch.Y)

      # (2) Forward pass.
      outputs = self.model.forward(X)

      # (3) Compute loss.
      loss = loss_fn(outputs, Y)
      # (3.1) Add model-specific loss (e.g., physics residual for PINNs).
      model_loss = self.model.model_loss(X, outputs, Y)
      if model_loss is not None:
        loss = loss + model_loss

      # (4) Backward pass + parameter update (backend-specific).
      self._backward_and_update(loss)

      # (5) Record training loss.
      self.history.record(loss_fn, iteration=i, value=loss, group='train')

      # (5.1) Print progress.
      if (i + 1) % print_every == 0:
        self._print_progress(i, loss_fn)

      # (6) Validation.
      if val_set is not None and (i + 1) % validate_every == 0:
        self._validate(val_set, val_metrics, i)
        # (6.1) Print validation results.
        self._print_validation(val_metrics, i)

      # (7) Check stopping criteria.
      if self._should_stop():
        stopped_early = True
        break

    # (8) Restore best checkpoint if available.
    if self.config.save_best and self._state.best_checkpoint is not None:
      self._restore_checkpoint(self._state.best_checkpoint)
      best_iter, best_val = self.history.best(self._state.early_stop_metric)
      _console.show_status(
        f'Restored best model (iter {best_iter + 1}, '
        f'{self._state.early_stop_metric} = {best_val:.6g})')

    # (9) Print training summary.
    if stopped_early:
      _console.show_status(
        f'Early stopping at iter {i + 1} (patience={self.config.patience})')
    else:
      _console.show_status(f'Training complete (iter {i + 1})')

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

  def _save_checkpoint(self):
    """Save model state to memory. Returns checkpoint object."""
    raise NotImplementedError

  def _restore_checkpoint(self, checkpoint):
    """Restore model state from checkpoint."""
    raise NotImplementedError

  def _validate(self, val_set, val_metrics, iteration):
    """Run validation on val_set. Override in backends for gradient/mode handling."""
    # (1) Compute and record validation metrics.
    X, Y = self._prepare_batch(val_set.X, val_set.Y)
    outputs = self.model.forward(X)
    for metric in val_metrics:
      val = metric(outputs, Y)
      self.history.record(metric, iteration=iteration, value=val, group='val')
    # (2) Check improvement and update state.
    improved = self.history.improved(self._state.early_stop_metric)
    if improved:
      self._state.patience_counter = 0
      if self.config.save_best:
        self._state.best_checkpoint = self._save_checkpoint()
    else:
      self._state.patience_counter += 1

  # endregion: Backend-Specific Methods

  # region: Utilities

  def _check_config_warnings(self, val_set):
    """Warn about config knobs that have no effect in the current setup."""
    if val_set is None:
      if self.config.early_stop:
        _console.warn('early_stop=True has no effect without a validation set')
      if self.config.save_best:
        _console.warn('save_best=True has no effect without a validation set')

  def _print_progress(self, iteration, loss_fn):
    """Print iteration + training loss."""
    loss_key = f'train/{loss_fn.name}'
    loss_val = self.history.latest(loss_key)
    _console.show_status(f'Iter {iteration + 1} | {loss_key} = {loss_val:.6g}')

  def _print_validation(self, val_metrics, iteration):
    """Print validation metrics and notify on new best."""
    for metric in val_metrics:
      key = f'val/{metric.name}'
      val = self.history.latest(key)
      _console.supplement(f'{key} = {val:.6g}')
    # (1) Check if early stopping metric improved (or is first entry).
    es_key = self._state.early_stop_metric
    if es_key:
      track = self.history._tracks.get(es_key, [])
      if len(track) == 1 or self.history.improved(es_key):
        best_iter, best_val = self.history.best(es_key)
        _console.supplement(
          f'[Best] {es_key} = {best_val:.6g} (iter {best_iter + 1})')

  def _should_stop(self):
    """Check stopping criteria. Returns True if training should halt."""
    # (1) Early stopping.
    if self.config.early_stop:
      if self._state.patience_counter >= self.config.patience:
        return True
    return False

  def _resolve_val_set(self, train_set, val_set):
    """Resolve validation set.

    Priority: (1) explicit val_set → (2) auto-split via config.val_ratio → (3) None.
    """
    # (1) Explicit val_set takes priority.
    if val_set is not None:
      return train_set, val_set
    # (2) Auto-split if val_ratio is configured.
    val_ratio = self.config.val_ratio
    if val_ratio is not None:
      train_set, val_set = train_set.split(1 - val_ratio, val_ratio)
      return train_set, val_set
    # (3) No validation.
    return train_set, None

  def _resolve_val_metrics(self, val_metrics, loss_fn):
    """Resolve validation metrics.

    Priority: (1) train() arg → (2) config.val_metrics string → (3) [loss_fn].
    Also sets `_state.early_stop_metric` for early stopping (first metric).
    """
    # (1) Explicit list from train() signature.
    if val_metrics is not None:
      resolved = [self._resolve_metric(m) if isinstance(m, str) else m
                  for m in val_metrics]
    # (2) Config string (comma or semicolon separated).
    elif (config_str := self.config.val_metrics) is not None:
      import re
      names = [n.strip() for n in re.split(r'[,;]', config_str) if n.strip()]
      resolved = [self._resolve_metric(n) for n in names]
    # (3) Default to loss function.
    else:
      resolved = [loss_fn]
    # (4) Set early stopping key (first val metric).
    self._state.early_stop_metric = f'val/{resolved[0].name}'
    return resolved

  def _resolve_param(self, name, arg_value, required=False):
    """Resolve parameter: signature arg (priority) → config value.

    Args:
      name: Config knob name.
      arg_value: Value from train() signature (None means not provided).
      required: If True, raise error when both arg and config are None.
    """
    # (1) Signature arg takes priority (if explicitly provided).
    if arg_value is not None:
      return arg_value
    # (2) Fall back to config.
    config_value = getattr(self.config, name, None)
    if config_value is not None:
      return config_value
    # (3) Error if required.
    if required:
      raise ValueError(
        f"'{name}' must be specified via train() argument or config.")
    return arg_value

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
