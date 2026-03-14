"""Hyperparameter search via exhaustive grid enumeration."""

import numpy as np

from talos.utils.config import Config as ConfigBase
from ..talos_alchemy import AlchemySearch
from ..space import ContinuousParam, IntParam, CategoricalParam


class GridSearch(AlchemySearch):
  """Hyperparameter search via exhaustive grid enumeration.

  Each param axis is discretized to n_points values; all combinations
  are evaluated. The total number of trials equals the product of axis
  lengths (n_trials argument is ignored).
  """

  class Config(AlchemySearch.Config):
    n_points: int = ConfigBase.Integer(
      default=5, positive=True,
      description='Grid resolution for continuous and integer params.')

  def _create_sampler(self, space: dict):
    import optuna
    return optuna.samplers.GridSampler(self._build_optuna_space(space))

  def _build_optuna_space(self, space: dict) -> dict:
    """Build GridSampler search space: {param_name: [candidate_values]}."""
    result = {}
    n = self.config.n_points
    for k, p in space.items():
      if isinstance(p, ContinuousParam):
        pts = (np.logspace(np.log10(p.low), np.log10(p.high), n)
               if p.log else np.linspace(p.low, p.high, n))
        result[k] = list(pts)
      elif isinstance(p, IntParam):
        all_vals = list(range(p.low, p.high + 1))
        result[k] = (all_vals if len(all_vals) <= n
                     else [round(v) for v in np.linspace(p.low, p.high, n)])
      elif isinstance(p, CategoricalParam):
        result[k] = list(p.choices)
    return result
