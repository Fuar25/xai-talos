"""Search history for hyperparameter optimization."""

from dataclasses import dataclass


@dataclass
class TrialRecord:
  trial_id: int
  params:   dict
  score:    float


class SearchHistory:
  """Records trial results from a hyperparameter search."""

  def __init__(self, direction: str):
    # (1) Optimization direction: 'minimize' or 'maximize'.
    self._direction = direction
    self._trials: list[TrialRecord] = []

  def record(self, params: dict, score: float) -> TrialRecord:
    record = TrialRecord(len(self._trials), params, score)
    self._trials.append(record)
    return record

  @property
  def trials(self) -> list[TrialRecord]:
    return list(self._trials)

  @property
  def best_trial(self) -> TrialRecord | None:
    if not self._trials:
      return None
    fn = min if self._direction == 'minimize' else max
    return fn(self._trials, key=lambda r: r.score)

  @property
  def best_params(self) -> dict | None:
    t = self.best_trial
    return t.params if t else None

  @property
  def best_score(self) -> float | None:
    t = self.best_trial
    return t.score if t else None

  def __repr__(self):
    n = len(self._trials)
    if not n:
      return 'SearchHistory(empty)'
    return f'SearchHistory({n} trials, best={self.best_score:.6g})'
