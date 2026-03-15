"""Hyperparameter search space descriptors."""

import math
import random


class ContinuousParam:
  """Continuous float hyperparameter with optional log-space sampling."""

  def __init__(self, low: float, high: float, log: bool = False):
    self.low, self.high, self.log = low, high, log

  def sample(self) -> float:
    if self.log:
      return math.exp(random.uniform(math.log(self.low), math.log(self.high)))
    return random.uniform(self.low, self.high)


class IntParam:
  """Integer hyperparameter (both endpoints inclusive)."""

  def __init__(self, low: int, high: int):
    self.low, self.high = low, high

  def sample(self) -> int:
    return random.randint(self.low, self.high)


class CategoricalParam:
  """Categorical hyperparameter chosen from a fixed set of values."""

  def __init__(self, choices: list):
    self.choices = list(choices)

  def sample(self):
    return random.choice(self.choices)


def sample_space(space: dict) -> dict:
  """Draw one random sample from a space dict.

  All values in `space` must be Param instances.
  Fixed hyperparameters belong outside the space as named constants.
  """
  return {k: v.sample() for k, v in space.items()}
