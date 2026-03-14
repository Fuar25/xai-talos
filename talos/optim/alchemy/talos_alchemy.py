"""Hyperparameter search base class backed by Optuna."""

from talos.utils import Nomear
from talos.utils.config import Config as ConfigBase
from talos.utils.console import Console
from talos.optim.alchemy.space import ContinuousParam, IntParam, CategoricalParam
from talos.optim.alchemy.history import SearchHistory

_console = Console()


class AlchemySearch(Nomear):
  """Abstract base for Optuna-backed hyperparameter search.

  Subclasses implement _create_sampler() to select a sampling strategy.
  """

  SCOPE = 'talos.alchemy'

  class Config(ConfigBase):
    n_trials:  int  = ConfigBase.Integer(
      default=20, description='Number of search trials.', positive=True)
    direction: str  = ConfigBase.String(
      default='minimize',
      description="Optimization direction: 'minimize' or 'maximize'.")
    verbose:   bool = ConfigBase.Boolean(
      default=True, description='Print search progress.')

  def __init__(self):
    pass

  @Nomear.property()
  def config(self):
    return type(self).Config(name='alchemy')

  def _create_sampler(self, space: dict):
    """Return an optuna.samplers instance. Override in subclasses."""
    pass

  def run(self, objective, space, n_trials=None) -> SearchHistory:
    """Run hyperparameter search.

    Args:
      objective: Callable (params: dict) -> float. Encapsulates all training
        details and returns a scalar score for the given params.
      space: Dict mapping param names to Param instances.
      n_trials: Number of trials. Overrides config.n_trials if provided.

    Returns:
      SearchHistory containing all trial records and best result.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # (1) Resolve config.
    n_trials = n_trials if n_trials is not None else self.config.n_trials
    sampler  = self._create_sampler(space)
    study    = optuna.create_study(
      direction=self.config.direction, sampler=sampler)

    # (2) Wrap user objective for Optuna.
    def _optuna_objective(trial):
      params = self._suggest(trial, space)
      return float(objective(params))

    # (3) Verbose callback.
    def _callback(study, trial):
      if self.config.verbose:
        _console.show_status(
          f'Trial {trial.number + 1}/{n_trials}: {trial.params}')
        _console.supplement(f'score = {trial.value:.6g}')

    _console.show_status(f'Search started (n_trials={n_trials})')
    study.optimize(_optuna_objective, n_trials=n_trials, callbacks=[_callback])
    _console.show_status(f'Search complete. Best score = {study.best_value:.6g}')

    # (4) Convert Optuna study to SearchHistory.
    history = SearchHistory(direction=self.config.direction)
    for t in sorted(study.trials, key=lambda t: t.number):
      history.record(t.params, t.value)
    return history

  def _suggest(self, trial, space: dict) -> dict:
    """Map space dict to Optuna trial suggestions."""
    params = {}
    for k, p in space.items():
      if isinstance(p, ContinuousParam):
        params[k] = trial.suggest_float(k, p.low, p.high, log=p.log)
      elif isinstance(p, IntParam):
        params[k] = trial.suggest_int(k, p.low, p.high)
      elif isinstance(p, CategoricalParam):
        params[k] = trial.suggest_categorical(k, p.choices)
    return params
