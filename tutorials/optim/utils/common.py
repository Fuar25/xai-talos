"""Shared utilities for tutorials/optim."""
import math

import numpy as np
import matplotlib.pyplot as plt


def generate_data(n=200, x_min=0.0, x_max=None, noise=0.0, seed=None):
  """Generate sin(x) data as numpy arrays.

  Returns:
    X: shape (n, 1), evenly spaced over [x_min, x_max]
    Y: shape (n, 1), sin(X) + optional Gaussian noise
  """
  x_max = x_max if x_max is not None else 2 * math.pi
  if seed is not None:
    np.random.seed(seed)
  X = np.linspace(x_min, x_max, n).reshape(-1, 1)
  Y = np.sin(X).reshape(-1, 1)
  if noise > 0:
    Y = Y + np.random.normal(0, noise, Y.shape)
  return X, Y


def plot_loss_curve(history, key='train/mse', title=None):
  """Plot a single training or validation loss curve."""
  iters = history.iterations(key)
  vals  = history.values(key)
  fig, ax = plt.subplots(figsize=(7, 3))
  ax.plot(iters, vals, linewidth=1.2)
  ax.set_xlabel('Iteration')
  ax.set_ylabel('MSE')
  ax.set_title(title or key)
  ax.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.show()


def plot_loss_curves(history, keys, labels=None, title=None):
  """Plot multiple loss curves on the same axis."""
  labels = labels or keys
  fig, ax = plt.subplots(figsize=(7, 3))
  for key, label in zip(keys, labels):
    iters = history.iterations(key)
    vals  = history.values(key)
    ax.plot(iters, vals, linewidth=1.2, label=label)
  ax.set_xlabel('Iteration')
  ax.set_ylabel('MSE')
  ax.set_title(title or 'Loss curves')
  ax.legend()
  ax.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.show()


def plot_predictions(X, Y_true, Y_pred=None, train_X=None, train_Y=None, title=None):
  """Compare model predictions against ground truth.

  Args:
    X: x-values for the curves, shape (n, 1).
    Y_true: Ground truth y-values, shape (n, 1).
    Y_pred: Model predictions, shape (n, 1). Omit to show data only.
    train_X: Optional training point x-values (scatter).
    train_Y: Optional training point y-values (scatter).
    title: Plot title.
  """
  # Sort by X so line plots connect points in order (X may be shuffled)
  sort_idx = np.argsort(X.ravel())
  X_sorted      = X[sort_idx]
  Y_true_sorted = Y_true[sort_idx]
  Y_pred_sorted = Y_pred[sort_idx] if Y_pred is not None else None

  fig, ax = plt.subplots(figsize=(7, 3))
  if train_X is not None:
    ax.scatter(train_X, train_Y, s=8, alpha=0.4, color='gray',
               label='Train data', zorder=3)
  ax.plot(X_sorted, Y_true_sorted, color='steelblue', linewidth=1.5, label='Ground truth')
  if Y_pred_sorted is not None:
    ax.plot(X_sorted, Y_pred_sorted, color='tomato', linestyle='--', linewidth=1.5,
            label='Prediction')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_title(title or 'Predictions vs Ground Truth')
  ax.legend()
  ax.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.show()


def get_trainer(model, loss_fn='mse', lr=1e-3, print_every=500):
  """Create a TorchTrainer with Adam optimizer and common defaults.

  Args:
    model: The model to train.
    loss_fn: Loss function spec (default 'mse').
    lr: Learning rate (default 1e-3).
    print_every: Print progress every N iterations.

  Returns:
    Configured TorchTrainer instance.
  """
  import talos as ta
  trainer = ta.TorchTrainer(model, optimizer='adam', loss_fn=loss_fn, lr=lr)
  trainer.config.print_every = print_every
  return trainer
