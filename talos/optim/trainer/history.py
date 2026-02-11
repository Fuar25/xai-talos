"""Training history with track-based recording."""

from __future__ import annotations

from collections import OrderedDict

from talos.eval.talos_metric import TalosMetric


class TrainingHistory:
  """Track-based training history. Each metric is an independent time series."""

  def __init__(self):
    self._tracks: OrderedDict[str, list[tuple[int, float]]] = OrderedDict()
    self._directions: dict[str, str] = {}

  # region: Recording

  def record(self, metric: TalosMetric, iteration: int, value, group: str = 'train'):
    """Record a metric value at a given iteration.

    Args:
      metric: TalosMetric instance (provides name and direction).
      iteration: Current training iteration.
      value: Metric value (scalar).
      group: Track group ('train', 'val', etc.).
    """
    # (1) Convert tensor/numpy to float.
    if hasattr(value, 'item'):
      value = value.item()
    value = float(value)

    # (2) Build track key and record.
    key = f'{group}/{metric.name}'
    if key not in self._tracks:
      self._tracks[key] = []
      self._directions[key] = metric.direction
    self._tracks[key].append((iteration, value))

  # endregion: Recording

  # region: Querying

  def __getitem__(self, key: str) -> list[tuple[int, float]]:
    """Get full track as list of (iteration, value) pairs."""
    if key not in self._tracks:
      raise KeyError(f"No track '{key}'. Available: {list(self._tracks.keys())}")
    return self._tracks[key]

  def latest(self, key: str) -> float | None:
    """Get the most recent value for a track."""
    if key not in self._tracks or not self._tracks[key]:
      return None
    return self._tracks[key][-1][1]

  def values(self, key: str) -> list[float]:
    """Get all values for a track (y-axis for plotting)."""
    return [v for _, v in self[key]]

  def iterations(self, key: str) -> list[int]:
    """Get all iteration numbers for a track (x-axis for plotting)."""
    return [i for i, _ in self[key]]

  def best(self, key: str) -> tuple[int, float] | None:
    """Get the best (iteration, value) based on metric direction."""
    if key not in self._tracks or not self._tracks[key]:
      return None
    direction = self._directions[key]
    if direction == 'minimize':
      return min(self._tracks[key], key=lambda x: x[1])
    return max(self._tracks[key], key=lambda x: x[1])

  def improved(self, key: str) -> bool:
    """Check if the latest value improved over the previous best.

    Returns False if fewer than 2 entries exist.
    """
    track = self._tracks.get(key, [])
    if len(track) < 2:
      return False
    direction = self._directions[key]
    latest_val = track[-1][1]
    # (1) Best among all entries except the latest.
    prev_entries = track[:-1]
    if direction == 'minimize':
      prev_best = min(v for _, v in prev_entries)
      return latest_val < prev_best
    prev_best = max(v for _, v in prev_entries)
    return latest_val > prev_best

  # endregion: Querying

  # region: Utilities

  @property
  def tracks(self) -> list[str]:
    """List all track keys."""
    return list(self._tracks.keys())

  def __repr__(self):
    parts = []
    for key in self._tracks:
      n = len(self._tracks[key])
      latest = self.latest(key)
      parts.append(f'{key}: {n} entries, latest={latest:.6g}')
    body = '; '.join(parts) if parts else 'empty'
    return f'TrainingHistory({body})'

  # endregion: Utilities
