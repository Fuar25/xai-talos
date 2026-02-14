"""Utilities for tutorial 01: Simple function fitting."""
import matplotlib.pyplot as plt
import talos as ta


def get_axis(figsize=(6, 3)):
  """Create a matplotlib axis with the specified figure size."""
  fig, ax = plt.subplots(figsize=figsize)
  return ax


def plot_points(ax, X, Y, color, label):
  """Plot points on the given axis."""
  ax.scatter(X, Y, c=color, label=label, s=10, alpha=0.6)


def finalize_plot(ax, title='Plot', xlabel='x', ylabel='y'):
  """Finalize the plot with title, labels, legend, and grid."""
  ax.set_title(title)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.legend()
  ax.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.show()


def plot_data(train_set, test_set=None, Y_pred=None, eval=False, title=None):
  """Plot training and test data, optionally with predictions.

  Args:
    train_set: TalosData for training set.
    test_set: TalosData for test set (optional).
    Y_pred: Predictions (optional, requires test_set).
    eval: If True, plot predictions vs ground truth on test set.
    title: Custom plot title (optional).
  """
  ax = get_axis((6, 3))

  if eval:
    # Evaluation mode: plot predictions vs ground truth
    if test_set is None or Y_pred is None:
      raise ValueError("eval=True requires both test_set and Y_pred")
    plot_points(ax, test_set.X, test_set.Y, 'green', 'Ground Truth')
    plot_points(ax, test_set.X, Y_pred, 'red', 'Prediction')
    default_title = 'Model Prediction vs Ground Truth (Test Set)'
  else:
    # Data visualization mode
    plot_points(ax, train_set.X, train_set.Y, 'blue', 'Train Set')
    if test_set is not None:
      plot_points(ax, test_set.X, test_set.Y, 'green', 'Test Set')
    default_title = 'Dataset Visualization'

  finalize_plot(ax, title=title or default_title)


def get_trainer(model, loss_fn='mse', early_stop=False, patience=10,
                val_ratio=None, validate_every=100, print_every=100, **kwargs):
  """Create a TorchTrainer with common configuration.

  Args:
    model: The model to train.
    loss_fn: Loss function specification.
    early_stop: Enable early stopping.
    patience: Early stopping patience.
    val_ratio: Validation split ratio.
    validate_every: Validate every N iterations.
    print_every: Print progress every N iterations.
    **kwargs: Additional optimizer configs.

  Returns:
    Configured TorchTrainer instance.
  """
  trainer = ta.TorchTrainer(model, loss_fn=loss_fn, **kwargs)
  trainer.config.early_stop = early_stop
  trainer.config.patience = patience
  if val_ratio is not None:
    trainer.config.val_ratio = val_ratio
  trainer.config.validate_every = validate_every
  trainer.config.print_every = print_every
  return trainer
