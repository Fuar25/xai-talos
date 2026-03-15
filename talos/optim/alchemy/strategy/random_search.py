"""Hyperparameter search via random sampling."""

from ..talos_alchemy import AlchemySearch


class RandomSearch(AlchemySearch):
  """Hyperparameter search via uniform random sampling."""

  def _create_sampler(self, space: dict):
    import optuna
    return optuna.samplers.RandomSampler()
