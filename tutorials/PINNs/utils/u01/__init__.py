import matplotlib.pyplot as plt


def get_axis(fig_size) -> plt.Axes:
  return plt.subplots(figsize=fig_size)[1]


def plot_points(ax: plt.Axes, X, Y, color, label):
  """Plot points on the given axis."""
  ax.scatter(X, Y, color=color, label=label, alpha=0.5)


def finalize_plot(ax: plt.Axes, title: str):
  """Finalize the plot by setting labels, title, legend, and showing the plot."""
  # Set grid to True
  ax.grid(True)
  ax.legend()
  ax.set_title(title)