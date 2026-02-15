"""Utilities for tutorial 03: Solving a PDE (1D diffusion equation)."""
import numpy as np
import matplotlib.pyplot as plt
import talos as ta


# region: Exact solution and data generation

def exact_solution(x, t):
  """Exact solution: y(x,t) = exp(-t) * sin(pi*x)."""
  return np.exp(-t) * np.sin(np.pi * x)


def generate_icbc_data(n_ic=200, n_bc=100, x_min=-1, x_max=1, t_min=0, t_max=1):
  """Generate initial condition and boundary condition training points.

  Returns:
    X_train: (N, 2) array of (x, t) points from IC and BC.
    Y_train: (N, 1) array of exact solution values at those points.
  """
  # (1) Initial condition: y(x, 0) = sin(pi*x).
  x_ic = np.linspace(x_min, x_max, n_ic).reshape(-1, 1)
  t_ic = np.zeros((n_ic, 1))
  y_ic = np.sin(np.pi * x_ic)

  # (2) Boundary conditions: y(x_min, t) = 0, y(x_max, t) = 0.
  t_bc = np.linspace(t_min, t_max, n_bc).reshape(-1, 1)
  # Left boundary.
  x_left = np.full((n_bc, 1), x_min)
  y_left = np.zeros((n_bc, 1))
  # Right boundary.
  x_right = np.full((n_bc, 1), x_max)
  y_right = np.zeros((n_bc, 1))

  # (3) Combine all IC+BC points.
  X_train = np.vstack([
    np.hstack([x_ic, t_ic]),
    np.hstack([x_left, t_bc]),
    np.hstack([x_right, t_bc]),
  ])
  Y_train = np.vstack([y_ic, y_left, y_right])
  return X_train, Y_train


def generate_test_grid(n_x=200, n_t=100, x_min=-1, x_max=1, t_min=0, t_max=1):
  """Generate a dense test grid with exact solution values.

  Returns:
    X_test: (n_x*n_t, 2) array of (x, t) points.
    Y_test: (n_x*n_t, 1) array of exact solution values.
    x_grid: 1D array of x values (for plotting).
    t_grid: 1D array of t values (for plotting).
  """
  x_grid = np.linspace(x_min, x_max, n_x)
  t_grid = np.linspace(t_min, t_max, n_t)
  T, X = np.meshgrid(t_grid, x_grid)  # X: (n_x, n_t), T: (n_x, n_t)
  X_test = np.hstack([X.flatten()[:, None], T.flatten()[:, None]])
  Y_test = exact_solution(X_test[:, 0:1], X_test[:, 1:2])
  return X_test, Y_test, x_grid, t_grid

# endregion: Exact solution and data generation

# region: Plotting

def plot_contour(x_grid, t_grid, values, title='', vmin=None, vmax=None):
  """Plot a filled contour map of values on the (t, x) grid.

  Args:
    x_grid: 1D array of x values.
    t_grid: 1D array of t values.
    values: (n_x*n_t,) or (n_x, n_t) array.
    title: Plot title.
    vmin, vmax: Color scale limits.
  """
  n_x, n_t = len(x_grid), len(t_grid)
  Z = values.reshape(n_x, n_t) if values.ndim == 1 else values
  T, X = np.meshgrid(t_grid, x_grid)

  fig, ax = plt.subplots(figsize=(6, 4))
  kwargs = {}
  if vmin is not None: kwargs['vmin'] = vmin
  if vmax is not None: kwargs['vmax'] = vmax
  cp = ax.contourf(T, X, Z, levels=20, cmap='rainbow', **kwargs)
  fig.colorbar(cp, ax=ax)
  ax.set_xlabel('t')
  ax.set_ylabel('x')
  ax.set_title(title)
  plt.tight_layout()
  plt.show()


def plot_comparison(x_grid, t_grid, Y_exact, Y_pred, title_prefix=''):
  """Plot exact solution, PINN prediction, and absolute error side by side.

  Args:
    x_grid: 1D array of x values.
    t_grid: 1D array of t values.
    Y_exact: (n_x*n_t, 1) exact solution.
    Y_pred: (n_x*n_t, 1) PINN prediction.
    title_prefix: Optional prefix for titles.
  """
  n_x, n_t = len(x_grid), len(t_grid)
  T, X = np.meshgrid(t_grid, x_grid)
  Z_exact = Y_exact.reshape(n_x, n_t)
  Z_pred = Y_pred.reshape(n_x, n_t)
  Z_err = np.abs(Z_exact - Z_pred)

  vmin = min(Z_exact.min(), Z_pred.min())
  vmax = max(Z_exact.max(), Z_pred.max())

  fig, axes = plt.subplots(1, 3, figsize=(15, 4))
  panels = [
    (Z_exact, 'Exact Solution', dict(vmin=vmin, vmax=vmax)),
    (Z_pred, 'PINN Prediction', dict(vmin=vmin, vmax=vmax)),
    (Z_err, 'Absolute Error', dict()),
  ]
  for ax, (Z, label, kw) in zip(axes, panels):
    cp = ax.contourf(T, X, Z, levels=20, cmap='rainbow', **kw)
    fig.colorbar(cp, ax=ax)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_title(f'{title_prefix}{label}')

  plt.tight_layout()
  plt.show()


def plot_icbc_points(X_train, x_min=-1, x_max=1, t_min=0, t_max=1):
  """Visualize IC and BC training points on the (t, x) domain.

  Args:
    X_train: (N, 2) array of (x, t) training points.
  """
  fig, ax = plt.subplots(figsize=(6, 4))

  # (1) Classify points by type.
  is_ic = X_train[:, 1] == t_min
  is_left = X_train[:, 0] == x_min
  is_right = X_train[:, 0] == x_max

  ax.scatter(X_train[is_ic, 1], X_train[is_ic, 0],
             c='blue', s=10, label='IC: y(x, 0)')
  ax.scatter(X_train[is_left, 1], X_train[is_left, 0],
             c='red', s=10, label=f'BC: y({x_min}, t)')
  ax.scatter(X_train[is_right, 1], X_train[is_right, 0],
             c='green', s=10, label=f'BC: y({x_max}, t)')

  ax.set_xlabel('t')
  ax.set_ylabel('x')
  ax.set_title('Training Points (IC + BC)')
  ax.legend()
  ax.set_xlim(t_min - 0.05, t_max + 0.05)
  ax.set_ylim(x_min - 0.1, x_max + 0.1)
  ax.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.show()

# endregion: Plotting

# region: Trainer helper

def get_trainer(model, loss_fn='mse', optimizer='adam', lr=1e-3,
                print_every=2000, **kwargs):
  """Create a TorchTrainer configured for PDE problems.

  Args:
    model: The model to train.
    loss_fn: Loss function specification.
    optimizer: Optimizer name.
    lr: Learning rate.
    print_every: Print progress every N iterations.
    **kwargs: Additional optimizer configs.

  Returns:
    Configured TorchTrainer instance.
  """
  trainer = ta.TorchTrainer(model, optimizer=optimizer, loss_fn=loss_fn,
                            lr=lr, **kwargs)
  trainer.config.print_every = print_every
  return trainer

# endregion: Trainer helper
