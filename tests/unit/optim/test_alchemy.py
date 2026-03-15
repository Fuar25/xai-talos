# SPDX-License-Identifier: MIT
"""
Unit tests for talos/optim/alchemy/.

Tests cover:
- ContinuousParam, IntParam, CategoricalParam (space.py)
- sample_space (space.py)
- SearchHistory (history.py)
- AlchemySearch (talos_alchemy.py)
- End-to-end search integration
"""

import pytest

from talos.optim.alchemy.space import (
  ContinuousParam, IntParam, CategoricalParam, sample_space,
)
from talos.optim.alchemy.history import SearchHistory
from talos.optim.alchemy import AlchemySearch, RandomSearch, GridSearch, BayesianSearch


# region: Fixtures

@pytest.fixture
def minimize_history():
  """SearchHistory configured for minimization."""
  return SearchHistory(direction='minimize')


@pytest.fixture
def maximize_history():
  """SearchHistory configured for maximization."""
  return SearchHistory(direction='maximize')


@pytest.fixture
def mixed_space():
  """Search space with one param of each type."""
  return {
    'lr':     ContinuousParam(1e-4, 1e-1),
    'layers': IntParam(1, 5),
    'act':    CategoricalParam(['relu', 'tanh', 'gelu']),
  }


# endregion: Fixtures

# region: TestContinuousParam

class TestContinuousParam:
  """Tests for ContinuousParam."""

  def test_sample_is_float(self):
    """sample() returns a float."""
    p = ContinuousParam(0.0, 1.0)
    assert isinstance(p.sample(), float)

  def test_sample_in_range(self):
    """sample() returns a value within [low, high]."""
    p = ContinuousParam(0.1, 0.9)
    v = p.sample()
    assert 0.1 <= v <= 0.9

  def test_sample_many_in_range(self):
    """Repeated sampling always stays within [low, high]."""
    p = ContinuousParam(1e-4, 1e-1)
    for _ in range(1000):
      v = p.sample()
      assert 1e-4 <= v <= 1e-1

  def test_log_sample_is_float(self):
    """log=True: sample() still returns a float."""
    p = ContinuousParam(1e-4, 1e-1, log=True)
    assert isinstance(p.sample(), float)

  def test_log_sample_in_range(self):
    """log=True: sampled value is within [low, high]."""
    p = ContinuousParam(1e-4, 1e-1, log=True)
    v = p.sample()
    assert 1e-4 <= v <= 1e-1

  def test_log_vs_linear_distribution(self):
    """Log sampling concentrates values toward low end vs linear."""
    # (1) Each distribution sampled 2000 times.
    linear = ContinuousParam(1e-4, 1.0, log=False)
    log    = ContinuousParam(1e-4, 1.0, log=True)

    n = 2000
    midpoint = (1e-4 + 1.0) / 2  # 0.50005
    linear_below = sum(1 for _ in range(n) if linear.sample() < midpoint)
    log_below    = sum(1 for _ in range(n) if log.sample() < midpoint)

    # (2) Linear ~50% below midpoint; log should be much more skewed toward low end.
    assert abs(linear_below / n - 0.5) < 0.05   # linear is approximately uniform
    assert log_below / n > 0.90                  # log is heavily skewed low


# endregion: TestContinuousParam

# region: TestIntParam

class TestIntParam:
  """Tests for IntParam."""

  def test_sample_is_int(self):
    """sample() returns an int."""
    p = IntParam(1, 10)
    assert isinstance(p.sample(), int)

  def test_sample_in_range(self):
    """sample() returns a value within [low, high] (both endpoints inclusive)."""
    p = IntParam(3, 7)
    v = p.sample()
    assert 3 <= v <= 7

  def test_sample_many_in_range(self):
    """Repeated sampling always stays within [low, high]."""
    p = IntParam(0, 100)
    for _ in range(1000):
      v = p.sample()
      assert 0 <= v <= 100


# endregion: TestIntParam

# region: TestCategoricalParam

class TestCategoricalParam:
  """Tests for CategoricalParam."""

  def test_sample_from_choices(self):
    """sample() returns a value from the choices list."""
    choices = ['relu', 'tanh', 'gelu']
    p = CategoricalParam(choices)
    assert p.sample() in choices

  def test_sample_many_from_choices(self):
    """Repeated sampling always returns a value from choices."""
    choices = [32, 64, 128, 256]
    p = CategoricalParam(choices)
    for _ in range(1000):
      assert p.sample() in choices


# endregion: TestCategoricalParam

# region: TestSampleSpace

class TestSampleSpace:
  """Tests for sample_space()."""

  def test_returns_all_keys(self, mixed_space):
    """Sampled dict has exactly the same keys as the input space."""
    sample = sample_space(mixed_space)
    assert set(sample.keys()) == set(mixed_space.keys())

  def test_mixed_space_values_in_range(self, mixed_space):
    """Sampled values from a mixed space are valid for their param types."""
    sample = sample_space(mixed_space)
    assert isinstance(sample['lr'], float)
    assert 1e-4 <= sample['lr'] <= 1e-1
    assert isinstance(sample['layers'], int)
    assert 1 <= sample['layers'] <= 5
    assert sample['act'] in ['relu', 'tanh', 'gelu']


# endregion: TestSampleSpace

# region: TestSearchHistory

class TestSearchHistory:
  """Tests for SearchHistory."""

  def test_empty_on_init(self, minimize_history):
    """Fresh history has no trials."""
    assert minimize_history.trials == []

  def test_best_trial_empty(self, minimize_history):
    """best_trial/best_params/best_score all return None when empty."""
    assert minimize_history.best_trial  is None
    assert minimize_history.best_params is None
    assert minimize_history.best_score  is None

  def test_record_appends(self, minimize_history):
    """record() increases the trial count by one."""
    minimize_history.record({'lr': 0.01}, 0.5)
    assert len(minimize_history.trials) == 1

  def test_record_trial_id(self, minimize_history):
    """trial_id starts at 0 and increments with each record."""
    r0 = minimize_history.record({'lr': 0.01}, 0.5)
    r1 = minimize_history.record({'lr': 0.001}, 0.3)
    assert r0.trial_id == 0
    assert r1.trial_id == 1

  def test_record_stores_params_score(self, minimize_history):
    """TrialRecord stores params and score correctly."""
    params = {'lr': 0.01, 'layers': 3}
    score  = 0.42
    record = minimize_history.record(params, score)
    assert record.params == params
    assert record.score  == score

  def test_best_minimize(self, minimize_history):
    """direction='minimize': best_trial has the lowest score."""
    minimize_history.record({'lr': 0.1},  score=0.9)
    minimize_history.record({'lr': 0.01}, score=0.2)   # best
    minimize_history.record({'lr': 0.001},score=0.5)

    assert minimize_history.best_score  == 0.2
    assert minimize_history.best_params == {'lr': 0.01}

  def test_best_maximize(self, maximize_history):
    """direction='maximize': best_trial has the highest score."""
    maximize_history.record({'lr': 0.1},  score=0.6)
    maximize_history.record({'lr': 0.01}, score=0.95)  # best
    maximize_history.record({'lr': 0.001},score=0.3)

    assert maximize_history.best_score  == 0.95
    assert maximize_history.best_params == {'lr': 0.01}

  def test_trials_returns_copy(self, minimize_history):
    """trials property returns a copy; mutating it doesn't affect internal state."""
    minimize_history.record({'lr': 0.01}, 0.5)
    copy = minimize_history.trials
    copy.clear()
    assert len(minimize_history.trials) == 1

  def test_repr_empty(self, minimize_history):
    """__repr__ on empty history returns 'SearchHistory(empty)'."""
    assert repr(minimize_history) == 'SearchHistory(empty)'

  def test_repr_with_trials(self, minimize_history):
    """__repr__ with trials mentions trial count and best score."""
    minimize_history.record({'lr': 0.01}, 0.42)
    r = repr(minimize_history)
    assert '1' in r
    assert '0.42' in r


# endregion: TestSearchHistory

# region: TestAlchemySearch

class TestAlchemySearch:
  """Tests for AlchemySearch.run() via the RandomSearch concrete subclass."""

  def test_run_returns_history(self, mixed_space):
    """run() returns a SearchHistory instance."""
    history = RandomSearch().run(lambda p: 0.5, mixed_space, n_trials=3)
    assert isinstance(history, SearchHistory)

  def test_run_n_trials_count(self, mixed_space):
    """Number of recorded trials equals n_trials."""
    n = 5
    history = RandomSearch().run(lambda p: 0.5, mixed_space, n_trials=n)
    assert len(history.trials) == n

  def test_run_objective_called_each_trial(self, mixed_space):
    """objective is called exactly n_trials times."""
    call_count = [0]

    def objective(params):
      call_count[0] += 1
      return 0.0

    n = 7
    RandomSearch().run(objective, mixed_space, n_trials=n)
    assert call_count[0] == n

  def test_run_params_from_space(self, mixed_space):
    """Params passed to objective contain all keys from space."""
    received = []

    def objective(params):
      received.append(params)
      return 0.0

    RandomSearch().run(objective, mixed_space, n_trials=3)
    for params in received:
      assert set(params.keys()) == set(mixed_space.keys())

  def test_run_uses_config_n_trials(self, mixed_space):
    """Without n_trials argument, config.n_trials is used."""
    alchemy = RandomSearch()
    alchemy.config.n_trials = 4
    history = alchemy.run(lambda p: 0.5, mixed_space)
    assert len(history.trials) == 4

  def test_run_override_n_trials(self, mixed_space):
    """Explicit n_trials argument overrides config.n_trials."""
    alchemy = RandomSearch()
    alchemy.config.n_trials = 20
    history = alchemy.run(lambda p: 0.5, mixed_space, n_trials=3)
    assert len(history.trials) == 3


# endregion: TestAlchemySearch

# region: TestAlchemyIntegration

class TestAlchemyIntegration:
  """End-to-end search flow tests using a deterministic mock objective."""

  def test_full_search_minimize(self):
    """best_score equals the minimum score across all trials (minimize)."""
    # (1) ContinuousParam with a known range — objective returns lr directly.
    space = {'lr': ContinuousParam(1e-4, 1e-1)}

    scores = []
    def objective(params):
      score = params['lr']
      scores.append(score)
      return score

    alchemy = RandomSearch()
    history = alchemy.run(objective, space, n_trials=10)

    assert history.best_score == min(scores)

  def test_full_search_maximize(self):
    """best_score equals the maximum score across all trials (maximize)."""
    space = {'lr': ContinuousParam(1e-4, 1e-1)}

    scores = []
    def objective(params):
      score = params['lr']
      scores.append(score)
      return score

    alchemy = RandomSearch()
    alchemy.config.direction = 'maximize'
    history = alchemy.run(objective, space, n_trials=10)

    assert history.best_score == max(scores)


# endregion: TestAlchemyIntegration

# region: TestRandomSearch

class TestRandomSearch:
  """Tests for RandomSearch._create_sampler()."""

  def test_sampler_type(self, mixed_space):
    """_create_sampler() returns a RandomSampler instance."""
    import optuna
    sampler = RandomSearch()._create_sampler(mixed_space)
    assert isinstance(sampler, optuna.samplers.RandomSampler)


# endregion: TestRandomSearch

# region: TestGridSearch

class TestGridSearch:
  """Tests for GridSearch._build_optuna_space() and _create_sampler()."""

  def test_build_space_continuous_linear(self):
    """Linear ContinuousParam: n_points values spanning [low, high]."""
    gs = GridSearch()
    space = {'lr': ContinuousParam(0.1, 1.0)}
    result = gs._build_optuna_space(space)
    pts = result['lr']
    assert len(pts) == gs.config.n_points
    assert abs(pts[0]  - 0.1) < 1e-9
    assert abs(pts[-1] - 1.0) < 1e-9

  def test_build_space_continuous_log(self):
    """Log ContinuousParam: n_points values with geometric progression."""
    gs = GridSearch()
    space = {'lr': ContinuousParam(1e-4, 1e-1, log=True)}
    result = gs._build_optuna_space(space)
    pts = result['lr']
    assert len(pts) == gs.config.n_points
    assert abs(pts[0]  - 1e-4) < 1e-12
    assert abs(pts[-1] - 1e-1) < 1e-9
    # (1) Adjacent ratios should be approximately constant (geometric).
    ratios = [pts[i + 1] / pts[i] for i in range(len(pts) - 1)]
    assert max(ratios) / min(ratios) < 1.001

  def test_build_space_int_small_range(self):
    """IntParam with range <= n_points: all integers in [low, high] returned."""
    gs = GridSearch()
    space = {'layers': IntParam(1, 4)}  # range 1-4 = 4 values <= default n_points=5
    result = gs._build_optuna_space(space)
    assert result['layers'] == [1, 2, 3, 4]

  def test_build_space_int_large_range(self):
    """IntParam with range > n_points: exactly n_points rounded integers returned."""
    gs = GridSearch()
    gs.config.n_points = 3
    space = {'layers': IntParam(1, 100)}
    result = gs._build_optuna_space(space)
    pts = result['layers']
    assert len(pts) == 3
    assert all(isinstance(v, int) for v in pts)
    assert all(1 <= v <= 100 for v in pts)

  def test_build_space_categorical(self):
    """CategoricalParam: all choices preserved in order."""
    gs = GridSearch()
    choices = ['relu', 'tanh', 'gelu']
    space = {'act': CategoricalParam(choices)}
    result = gs._build_optuna_space(space)
    assert result['act'] == choices

  def test_n_points_config(self):
    """Changing n_points changes the number of grid values for continuous params."""
    space = {'lr': ContinuousParam(0.0, 1.0)}
    gs3 = GridSearch()
    gs3.config.n_points = 3
    gs7 = GridSearch()
    gs7.config.n_points = 7
    assert len(gs3._build_optuna_space(space)['lr']) == 3
    assert len(gs7._build_optuna_space(space)['lr']) == 7

  def test_sampler_type(self, mixed_space):
    """_create_sampler() returns a GridSampler instance."""
    import optuna
    sampler = GridSearch()._create_sampler(mixed_space)
    assert isinstance(sampler, optuna.samplers.GridSampler)

  def test_grid_covers_all_combinations(self):
    """With a 2x2 categorical grid and n_trials=4, all 4 combos are visited."""
    space = {
      'act':     CategoricalParam(['relu', 'tanh']),
      'dropout': CategoricalParam([0.1, 0.5]),
    }
    visited = []

    def objective(params):
      visited.append((params['act'], params['dropout']))
      return 0.0

    gs = GridSearch()
    gs.config.verbose = False
    gs.run(objective, space, n_trials=4)

    expected = {('relu', 0.1), ('relu', 0.5), ('tanh', 0.1), ('tanh', 0.5)}
    assert set(visited) == expected


# endregion: TestGridSearch

# region: TestBayesianSearch

class TestBayesianSearch:
  """Tests for BayesianSearch._create_sampler() and run()."""

  def test_sampler_type(self, mixed_space):
    """_create_sampler() returns a TPESampler instance."""
    import optuna
    sampler = BayesianSearch()._create_sampler(mixed_space)
    assert isinstance(sampler, optuna.samplers.TPESampler)

  def test_run_returns_history(self, mixed_space):
    """run() returns a SearchHistory instance."""
    bs = BayesianSearch()
    bs.config.verbose = False
    history = bs.run(lambda p: 0.5, mixed_space, n_trials=3)
    assert isinstance(history, SearchHistory)

  def test_run_n_trials(self, mixed_space):
    """Number of recorded trials equals n_trials."""
    bs = BayesianSearch()
    bs.config.verbose = False
    n = 5
    history = bs.run(lambda p: 0.5, mixed_space, n_trials=n)
    assert len(history.trials) == n


# endregion: TestBayesianSearch
