"""Utilities for improving experiment reproducibility.

This module provides best-effort seeding across common Python ML libraries.
"""

import random


def set_seed(seed: int = 42) -> None:
  """Set seeds for common RNG sources.

  This is a best-effort utility. Optional dependencies (e.g., NumPy, PyTorch,
  TensorFlow) are handled via try/except so importing talos does not require them.

  Args:
    seed: The seed value to set. Defaults to 42.
  """
  # (1) Python standard library RNG.
  random.seed(seed)

  # (2) NumPy RNG (optional dependency).
  try:
    import numpy as np  # third-party
    np.random.seed(seed)
  except Exception:
    pass

  # (3) PyTorch RNG (optional dependency).
  try:
    import torch  # third-party
    torch.manual_seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)

    # (3.1) Best-effort determinism helpers. Some environments may not support these.
    try:
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
    except Exception:
      pass

    # (3.2) PyTorch 1.8+: enforce deterministic algorithms where possible.
    try:
      torch.use_deterministic_algorithms(True)
    except Exception:
      pass
  except Exception:
    pass

  # (4) TensorFlow RNG (optional dependency).
  try:
    import tensorflow as tf  # third-party
    tf.random.set_seed(seed)
  except Exception:
    pass
