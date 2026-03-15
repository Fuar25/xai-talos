"""Utilities for tutorial 03: Hyperparameter Search."""
import matplotlib.pyplot as plt

from .common import generate_data, get_trainer


def plot_search_comparison(results):
  """Bar chart comparing best val/mse from different search strategies.

  Args:
    results: dict mapping strategy name (str) to best score (float).
  """
  names  = list(results.keys())
  scores = list(results.values())
  fig, ax = plt.subplots(figsize=(6, 3))
  colors = ['steelblue', 'seagreen', 'tomato']
  bars = ax.bar(names, scores, color=colors[:len(names)])
  for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(scores) * 0.01,
            f'{score:.4f}', ha='center', va='bottom', fontsize=9)
  ax.set_ylabel('Best val/mse')
  ax.set_title('Search Strategy Comparison')
  ax.grid(True, alpha=0.3, axis='y')
  plt.tight_layout()
  plt.show()


__all__ = ['generate_data', 'get_trainer', 'plot_search_comparison']
