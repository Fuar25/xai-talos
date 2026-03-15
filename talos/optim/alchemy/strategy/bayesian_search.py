"""Hyperparameter search via Bayesian optimization (TPE)."""

from talos.utils.config import Config as ConfigBase
from ..talos_alchemy import AlchemySearch


class BayesianSearch(AlchemySearch):
  """Hyperparameter search via Bayesian optimization (Tree-structured Parzen
  Estimator).

  The first n_initial trials use random sampling to bootstrap the surrogate
  model; subsequent trials are guided by TPE.
  """

  class Config(AlchemySearch.Config):
    n_initial: int = ConfigBase.Integer(
      default=10, positive=True,
      description='Random bootstrap trials before TPE sampling begins.')

  def _create_sampler(self, space: dict):
    import optuna
    return optuna.samplers.TPESampler(
      n_startup_trials=self.config.n_initial)
