# SPDX-License-Identifier: MIT
"""
TalosTask module is responsible for managing task-related functionalities,
  including:
  (1) data manipulation;
  (2) test model performance with specified metrics;
"""

from talos.utils import Nomear


class TalosTask(Nomear):
  """A base class for Talos tasks. """

  def __init__(self, name: str = "TalosTask", **kwargs):
    """Initialize the TalosTask with a name.

    Args:
        name (str): The name of the task. Defaults to "TalosTask".
    """
    self.name = name
    self.configs = kwargs