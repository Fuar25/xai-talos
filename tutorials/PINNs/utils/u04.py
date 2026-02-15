"""Utilities for tutorial 04: Discrete Time PINNs (Allen-Cahn equation)."""
import numpy as np
import matplotlib.pyplot as plt
import talos as ta


# region: IRK weights

def irk_weights(q):
  """Compute implicit Runge-Kutta weights for q-stage Gauss-Legendre method.

  Uses polynomial collocation at Gauss-Legendre nodes on [0, 1].

  Returns:
    a: (q, q) array — RK coefficient matrix.
    b: (q,) array — quadrature weights.
    c: (q,) array — stage nodes in [0, 1].
  """
  # (1) Gauss-Legendre nodes/weights on [-1, 1], shifted to [0, 1].
  nodes, weights = np.polynomial.legendre.leggauss(q)
  c = (nodes + 1) / 2
  b = weights / 2

  # (2) Compute a_ij via polynomial collocation.
  # Lagrange basis polynomials integrated from 0 to c_i.
  a = np.zeros((q, q))
  for i in range(q):
    for j in range(q):
      # (2.1) Build Lagrange basis polynomial L_j(x) = prod_{k!=j} (x - c_k)/(c_j - c_k).
      # Integrate L_j from 0 to c_i using numpy polynomial tools.
      poly = np.array([1.0])
      for k in range(q):
        if k == j:
          continue
        # Multiply by (x - c_k) / (c_j - c_k).
        factor = np.array([1.0, -c[k]]) / (c[j] - c[k])
        poly = np.convolve(poly, factor)
      # (2.2) Integrate the polynomial from 0 to c_i.
      # Antiderivative: integrate each term.
      anti = np.polyint(poly)
      a[i, j] = np.polyval(anti, c[i]) - np.polyval(anti, 0.0)

  return a, b, c

# endregion: IRK weights

# region: Reference solution

def solve_allen_cahn(n_x=512, t_span=(0.0, 1.0), t_eval=None):
  """Solve the Allen-Cahn equation using method of lines.

  u_t = 0.0001 * u_xx + 5*(u - u^3), x in [-1, 1], periodic BCs.
  IC: u(x,0) = x^2 * cos(pi*x).

  Uses Chebyshev spectral differentiation for spatial derivatives and
  scipy RK45 for time integration.

  Returns:
    x_grid: (n_x,) spatial grid.
    t_eval: (n_t,) time points.
    u_ref: (n_x, n_t) reference solution.
  """
  from scipy.integrate import solve_ivp

  # (1) Spatial grid (uniform, periodic).
  x_grid = np.linspace(-1, 1, n_x, endpoint=False)
  dx = x_grid[1] - x_grid[0]

  # (2) Spectral differentiation via FFT for periodic domain.
  def rhs(t, u):
    # (2.1) Second derivative via finite differences (periodic).
    u_xx = np.zeros_like(u)
    u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
    u_xx[0] = (u[1] - 2*u[0] + u[-1]) / dx**2
    u_xx[-1] = (u[0] - 2*u[-1] + u[-2]) / dx**2
    return 0.0001 * u_xx + 5 * (u - u**3)

  # (3) Initial condition.
  u0 = x_grid**2 * np.cos(np.pi * x_grid)

  # (4) Time integration.
  if t_eval is None:
    t_eval = np.linspace(t_span[0], t_span[1], 201)
  sol = solve_ivp(rhs, t_span, u0, method='RK45', t_eval=t_eval,
                  rtol=1e-8, atol=1e-10, max_step=1e-3)
  u_ref = sol.y  # (n_x, n_t)
  return x_grid, t_eval, u_ref

# endregion: Reference solution

# region: Data generation

def generate_training_data(x_grid, u_ref_snapshot, n_train=200):
  """Subsample N points from reference solution at a single time snapshot.

  Args:
    x_grid: (n_x,) spatial grid.
    u_ref_snapshot: (n_x,) reference solution at time t_n.
    n_train: Number of training points.

  Returns:
    X_train: (n_train, 1) array of x values.
    Y_train: (n_train, 1) array of u(x, t_n) values.
    indices: Selected indices (for visualization).
  """
  indices = np.sort(np.random.choice(len(x_grid), n_train, replace=False))
  X_train = x_grid[indices].reshape(-1, 1)
  Y_train = u_ref_snapshot[indices].reshape(-1, 1)
  return X_train, Y_train, indices

# endregion: Data generation

# region: Trainer helper

def get_trainer(model, optimizer='adam', lr=1e-3, print_every=2000, **kwargs):
  """Create a TorchTrainer configured for discrete PINN (no data loss).

  Args:
    model: The model to train.
    optimizer: Optimizer name.
    lr: Learning rate.
    print_every: Print progress every N iterations.

  Returns:
    Configured TorchTrainer instance.
  """
  trainer = ta.TorchTrainer(model, optimizer=optimizer, lr=lr, **kwargs)
  trainer.config.print_every = print_every
  return trainer

# endregion: Trainer helper

# region: Plotting

def plot_solution(x_grid, t_grid, u_ref, title='Allen-Cahn Reference Solution'):
  """Plot the full spatiotemporal reference solution as a contour map.

  Args:
    x_grid: (n_x,) spatial grid.
    t_grid: (n_t,) time grid.
    u_ref: (n_x, n_t) solution array.
  """
  T, X = np.meshgrid(t_grid, x_grid)
  fig, ax = plt.subplots(figsize=(8, 4))
  cp = ax.contourf(T, X, u_ref, levels=40, cmap='rainbow')
  fig.colorbar(cp, ax=ax, label='u(x, t)')
  ax.set_xlabel('t')
  ax.set_ylabel('x')
  ax.set_title(title)
  plt.tight_layout()
  plt.show()


def plot_training_data(x_grid, u_snapshot, x_train, y_train, t_n):
  """Plot the reference snapshot and the subsampled training points.

  Args:
    x_grid: (n_x,) full spatial grid.
    u_snapshot: (n_x,) reference solution at t_n.
    x_train: (n_train, 1) training x values.
    y_train: (n_train, 1) training u values.
    t_n: Time of the snapshot.
  """
  fig, ax = plt.subplots(figsize=(8, 3))
  ax.plot(x_grid, u_snapshot, 'b-', linewidth=1.5, label=f'Reference at t={t_n}')
  ax.scatter(x_train.flatten(), y_train.flatten(), c='red', s=15, zorder=5,
             label=f'Training data ({len(x_train)} pts)')
  ax.set_xlabel('x')
  ax.set_ylabel('u')
  ax.set_title(f'Training Data at t = {t_n}')
  ax.legend()
  ax.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.show()


def plot_prediction(x_grid, u_exact, u_pred, t_pred):
  """Plot predicted vs exact solution at a single time.

  Args:
    x_grid: (n_x,) spatial grid.
    u_exact: (n_x,) exact solution at t_pred.
    u_pred: (n_x,) PINN prediction at t_pred.
    t_pred: Time of prediction.
  """
  fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))

  # (1) Comparison.
  axes[0].plot(x_grid, u_exact, 'b-', linewidth=2, label='Reference')
  axes[0].plot(x_grid, u_pred, 'r--', linewidth=2, label='PINN prediction')
  axes[0].set_xlabel('x')
  axes[0].set_ylabel('u')
  axes[0].set_title(f'Solution at t = {t_pred}')
  axes[0].legend()
  axes[0].grid(True, alpha=0.3)

  # (2) Pointwise error.
  error = np.abs(u_exact - u_pred)
  axes[1].semilogy(x_grid, error, 'k-', linewidth=1)
  axes[1].set_xlabel('x')
  axes[1].set_ylabel('|error|')
  axes[1].set_title(f'Pointwise Absolute Error at t = {t_pred}')
  axes[1].grid(True, alpha=0.3)

  plt.tight_layout()
  plt.show()

# endregion: Plotting
