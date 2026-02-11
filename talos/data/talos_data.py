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
    e.g., metadata = [{"pid": "P001", "diagnosis": "insomnia", "gender": "male"},
                      {"pid": "P002", "gender": "female"}, ...]
    note that if stratify-based splitting ignores metadata fields that are not present in all data points,
    e.g., "diagnosis" in the above example.
"""

from talos.utils import Nomear, check_type, INT_TYPES

import numpy as np
from collections import defaultdict


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
  def wrap(cls, X, Y=None, metadata=None, name=None, **kwargs) -> "TalosData":
    """Wrap data arrays into a TalosData instance.

    Args:
      X: Input data (np.ndarray or list).
      Y: Optional labels/targets.
      metadata: Optional list of dicts with same length as X.
      name: Optional name for the dataset.
      **kwargs: Additional configuration.

    Returns:
      TalosData instance wrapping the provided data.
    """
    dataset = TalosData(name=name, **kwargs)
    dataset.X = X
    dataset.Y = Y
    dataset.metadata = metadata

    return dataset

  # endregion: Wrapping

  # region: APIs

  def sample(self, batch_size: int | None) -> "TalosData":
    """Sample a subset of data points.

    Args:
      batch_size: Number of data points to sample. If None or -1, return self.

    Returns:
      A new TalosData containing the sampled subset, or `self` when batch_size is None/-1.
    """
    if self.X is None:
      raise ValueError('!! X must not be None for sample')

    # (1) Return all data without copying.
    if batch_size is None or batch_size == -1:
      return self

    # (2) Validate batch_size
    batch_size = check_type(batch_size, INT_TYPES, positive=True)

    total_size = len(self.X)
    batch_size = min(batch_size, total_size)
    indices = np.random.choice(total_size, size=batch_size, replace=False).tolist()
    return self.get_subset(indices, name=f"{self.name} (sample:{batch_size})")

  def split(self, *sizes, names=None, shuffle=True,
            stratify=None) -> tuple["TalosData", ...]:
    """Split self into multiple TalosData instances according to the specified sizes.

    Usage example:
      train_set, test_set = self.split(5, 5, names=["train", "test"], shuffle=True)
      train_set, test_set = self.split(0.8, 0.2, stratify='label')  # stratify by metadata field
      train_set, test_set = self.split(0.8, 0.2, stratify=True)  # stratify by Y

    Args:
      *sizes: Split sizes (ints or floats). If sum != total, treated as ratios.
      names: Optional list of names for the splits.
      shuffle: Whether to shuffle before splitting.
      stratify: Stratification strategy:
        - None: No stratification (default)
        - True: Stratify by Y (Y must not be None)
        - str: Stratify by metadata field name
    """
    # (1) Validate base data.
    if self.X is None:
      raise ValueError('!! X must not be None for split')
    total_size = len(self.X)

    # (1.1) Validate aligned optional fields.
    if self.Y is not None and len(self.Y) != total_size:
      raise ValueError('!! Y must have the same length as X')
    if self.metadata is not None and len(self.metadata) != total_size:
      raise ValueError('!! metadata must have the same length as X')

    # (1.2) Validate stratify parameter
    if stratify is not None:
      if stratify is True:
        if self.Y is None:
          raise ValueError('!! Y must not be None when stratify=True')
      elif isinstance(stratify, str):
        if self.metadata is None:
          raise ValueError(f'!! metadata must not be None when stratify="{stratify}"')
        # Check if the field exists in all metadata entries
        for i, meta in enumerate(self.metadata):
          if not isinstance(meta, dict) or stratify not in meta:
            raise ValueError(
              f'!! metadata field "{stratify}" not found in metadata entry {i}')
      else:
        raise TypeError('!! stratify must be None, True, or a string (metadata field name)')

    # (2) Normalize `sizes` input.
    if len(sizes) == 0:
      raise ValueError('!! split sizes not specified')
    if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
      sizes = sizes[0]

    # (2.1) Validate sizes using check_type
    sizes = check_type(list(sizes), list, inner_type=(int, float), non_negative=True)

    # (2.2) Validate `names` if provided.
    if names is not None:
      names = check_type(names, (tuple, list))
      if len(names) != len(sizes):
        raise ValueError('!! length of name list and sizes list does not match')

    # (3) Resolve absolute sizes:
    # (3.1) If sum(sizes) == total_size, treat as absolute counts.
    # (3.2) Otherwise treat as ratios (e.g., 9:1) and distribute the remainder.
    size_sum = sum(sizes)
    if size_sum != total_size:
      if size_sum == 0:
        raise ValueError('!! split sizes cannot all be zero')
      base_sizes = [int((size * total_size) // size_sum) for size in sizes]
      remainder = int(total_size - sum(base_sizes))
      for i in range(remainder):
        base_sizes[i % len(base_sizes)] += 1
      sizes = base_sizes
    else:
      # Ensure sizes are integers even when they sum to total_size
      sizes = [int(s) for s in sizes]

    # (4) Build an index plan (optionally shuffled, optionally stratified).
    if stratify is None:
      # (4.1) Simple split without stratification
      indices = list(range(total_size))
      if shuffle: np.random.shuffle(indices)

      # (5.1) Materialize subsets using `get_subset`.
      data_sets = ()
      cursor = 0
      for i, size in enumerate(sizes):
        size = int(size)
        if size == 0: continue

        idx = indices[cursor:cursor + size]
        cursor += size

        data_set_name = names[i] if names is not None else f'{self.name} ({i+1}/{len(sizes)})'
        data_sets += (self.get_subset(idx, name=data_set_name),)

    else:
      # (4.2) Stratified split
      # (4.2.1) Get stratification labels
      if stratify is True:
        strat_labels = self.Y
      else:  # stratify is a string (metadata field name)
        strat_labels = [meta[stratify] for meta in self.metadata]

      # (4.2.2) Group indices by stratification label
      label_to_indices = defaultdict(list)
      for idx in range(total_size):
        label = strat_labels[idx]
        # Convert numpy types to Python types for hashability
        if hasattr(label, 'item'):
          label = label.item()
        label_to_indices[label].append(idx)

      # (4.2.3) Shuffle within each stratum if needed
      if shuffle:
        for label in label_to_indices:
          np.random.shuffle(label_to_indices[label])

      # (4.2.4) Split each stratum proportionally
      split_indices = [[] for _ in sizes]
      for label, label_indices in label_to_indices.items():
        n_samples = len(label_indices)
        # Calculate proportional sizes for this stratum
        strat_sizes = [(int(size) / total_size) * n_samples for size in sizes]
        # Convert to integers with remainder distribution
        strat_sizes_int = [int(s) for s in strat_sizes]
        remainder = int(n_samples - sum(strat_sizes_int))
        for i in range(remainder):
          strat_sizes_int[i % len(strat_sizes_int)] += 1

        # Distribute indices from this stratum to splits
        cursor = 0
        for i, strat_size in enumerate(strat_sizes_int):
          split_indices[i].extend(label_indices[cursor:cursor + strat_size])
          cursor += strat_size

      # (4.2.5) Shuffle the combined splits if needed (to mix strata)
      if shuffle:
        for idx_list in split_indices:
          np.random.shuffle(idx_list)

      # (5.2) Materialize stratified subsets
      data_sets = ()
      for i, idx in enumerate(split_indices):
        if len(idx) == 0: continue
        data_set_name = names[i] if names is not None else f'{self.name} ({i+1}/{len(sizes)})'
        data_sets += (self.get_subset(idx, name=data_set_name),)

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

  # region: Private Methods

  def get_subset(self, indices, name=None) -> "TalosData":
    """Create a subset of this dataset by indices.

    Args:
      indices: Indices to take (list/tuple/np.ndarray of ints).
      name: Optional name for the returned subset.

    Returns:
      A new TalosData containing X (and optional Y/metadata) at the given indices.
    """
    if self.X is None:
      raise ValueError('!! X must not be None for get_subset')

    total_size = len(self.X)

    # (1) Validate and normalize indices using check_type
    if indices is None:
      raise ValueError('!! indices must not be None')

    # Convert numpy array or tuple to list
    if isinstance(indices, np.ndarray):
      indices = indices.tolist()
    indices = check_type(indices, list, inner_type=INT_TYPES, non_negative=True)

    # (1.1) Check bounds
    for i, idx in enumerate(indices):
      if idx >= total_size:
        raise IndexError(f'!! index {idx} out of range [0, {total_size})')

    # (1.2) Validate aligned optional fields
    if self.Y is not None and len(self.Y) != total_size:
      raise ValueError('!! Y must have the same length as X')
    if self.metadata is not None and len(self.metadata) != total_size:
      raise ValueError('!! metadata must have the same length as X')

    def _subset_data(data):
      if data is None:
        return None
      if isinstance(data, np.ndarray):
        return data[indices]
      if isinstance(data, list):
        return [data[i] for i in indices]
      if isinstance(data, tuple):
        return tuple(data[i] for i in indices)
      try:
        return data[indices]
      except Exception:
        return [data[i] for i in indices]

    subset = TalosData(name=name or f"{self.name} (subset:{len(indices)})", **self.configs)
    subset.X = _subset_data(self.X)
    subset.Y = _subset_data(self.Y)
    subset.metadata = _subset_data(self.metadata)
    return subset

  # endregion: Private Methods
