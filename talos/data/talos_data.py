# SPDX-License-Identifier: MIT
"""
TalosData is the basic protocol for data in Talos.
Subclasses of TalosData can be converted to each other.

Each instance of TalosData mainly contains:
(1) X, a collection of data ready to by input to a model;
    Each datum in X can be anything, such as an image, a sentence, a graph, etc.
    X can be
    (1.1) np.ndarray, which is a common format for tabular data, images, etc.;
    (1.2) list of objects, each object can be:
        (1.2.1) an object ready to be input to a model;
        (1.2.2) a callable that generates an object ready to be input to a model,
           which is useful for large datasets that cannot fit into memory.
(2, optional) metadata, which describes the properties of each datum in X,
    such as the label of an image, the diagnosis of a patient, etc.
"""

from talos.utils import Nomear

import random
import numpy as np


class TalosData(Nomear):
  """A base class for Talos data. """

  SCOPE = 'talos.data'

  def __init__(self, name: str = "TalosData", **kwargs):
    """Initialize the TalosData with a name.

    Args:
        name (str): The name of the data. Defaults to "TalosData".
    """
    self.name = name
    self.configs = kwargs

    self.X = None
    self.Y = None

    self.metadata = None  # should be a list of dicts with the same length as X

  # region: Properties

  @property
  def size(self) -> int: return len(self.X)

  # endregion: Properties

  # region: Wrapping

  @classmethod
  def wrap(cls, X, Y=None, name=None, **kwargs) -> "TalosData":
    """
    """
    # TODO:
    dataset = TalosData(name=name, **kwargs)
    dataset.X = X
    dataset.Y = Y

    return dataset

  # endregion: Wrapping

  # region: APIs

  def split(self, *sizes, names=None, shuffle=True, stratify=None):
    """Split self into multiple TalosData instances according to the specified sizes.

    Usage example:
      train_set, test_set = self.split(5, 5, names=["train", "test"], shuffle=True)

    TODO: currently stratify is not implemented.
    """
    # 1. Sanity checks before splitting
    # 1.1 Make sure X is not None and has a valid length
    if self.X is None: raise ValueError('!! X must not be None for split')
    total_size = len(self.X)

    # 1.2 Validate aligned sizes
    if self.Y is not None and len(self.Y) != total_size:
      raise ValueError('!! Y must have the same length as X')
    if self.metadata is not None and len(self.metadata) != total_size:
      raise ValueError('!! metadata must have the same length as X')

    # 1.3 Validate names
    if names is not None:
      if not isinstance(names, (tuple, list)):
        raise TypeError('!! names must be a tuple or list of strings')
      if len(names) != len(sizes):
        raise ValueError('!! length of name list and sizes list does not match')

    # 1.4 Validate sizes
    # 1.4.1 Handle the case when sizes are passed in as a single list or tuple
    if len(sizes) == 0: raise ValueError('!! split sizes not specified')
    elif len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
      # in case sizes are passed in as a single list or tuple, ]
      #   e.g., split([5, 5], names=[...])
      sizes = sizes[0]

    # 1.4.2 Validate that each size is a non-negative integer
    sizes = list(sizes)
    for size in sizes:
      if not isinstance(size, (int, float)) or size < 0:
        raise ValueError('!! size must be a non-negative number')

    # 1.4.3 If the sum of sizes does not equal the total size,
    #   treat sizes as ratios and convert them to absolute sizes
    size_sum = sum(sizes)
    if size_sum != total_size:
      # treat sizes as ratios, e.g., [9, 1] for 9:1 split
      if size_sum == 0: raise ValueError('!! split sizes cannot all be zero')
      base_sizes = [(size * total_size) // size_sum for size in sizes]
      remainder = total_size - sum(base_sizes)
      for i in range(remainder): base_sizes[i % len(base_sizes)] += 1
      sizes = base_sizes

    # 2. Build indices
    indices = list(range(total_size))
    if shuffle: random.shuffle(indices)

    # 2.1 Helper for subsetting
    def _subset(data, idx):
      if data is None: return None
      if isinstance(data, np.ndarray): return data[idx]
      if isinstance(data, list): return [data[i] for i in idx]
      if isinstance(data, tuple): return tuple(data[i] for i in idx)
      try: return data[idx]
      except Exception: return [data[i] for i in idx]

    # 3. Slice and build datasets
    data_sets, cursor = (), 0
    for i, size in enumerate(sizes):
      if size == 0: continue
      idx = indices[cursor:cursor + size]
      cursor += size

      data_set = TalosData(name=f'{self.name} ({i+1}/{len(sizes)})', **self.configs)
      data_set.X = _subset(self.X, idx)
      data_set.Y = _subset(self.Y, idx)
      data_set.metadata = _subset(self.metadata, idx)
      if names is not None: data_set.name = names[i]
      data_sets += (data_set,)

    return data_sets

  def report(self):
    """Report the basic information of the dataset.
    """
    print(f":: Dataset `{self.name}` (size: {self.size})")
    if self.X is not None:
      print(f"   - X: type={type(self.X)}, shape={getattr(self.X, 'shape', 'N/A')}")
    if self.Y is not None:
      print(f"   - Y: type={type(self.Y)}, shape={getattr(self.Y, 'shape', 'N/A')}")

  # endregion: APIs
