"""
Models sub-package: Neural network architectures and model definitions.

This module keeps heavyweight optional dependencies (e.g., PyTorch) out of the
import path by lazily importing backend-specific subpackages on demand.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, TYPE_CHECKING

from .talos_model import TalosModel

__all__ = [
  "TalosModel",
  "torch_zoo",
]


# Lazy imports for avoiding heavy dependencies at import time.
if TYPE_CHECKING:
  # This does not run at runtime, so it does not defeat lazy loading.
  from .zoo import pytorch as torch_zoo


def __getattr__(name: str) -> Any:
  """Dynamically resolve module attributes for optional/lazy imports.

  Args:
    name: Attribute name requested from this module.

  Returns:
    The resolved attribute value.

  Raises:
    AttributeError: If the attribute is not defined.
  """
  # (1) Lazily expose the PyTorch model zoo as `talos.model.torch_zoo`.
  if name == "torch_zoo":
    module = import_module(".zoo.pytorch", package=__name__)
    globals()[name] = module
    return module

  raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
  """Return a sorted list of available attributes for IDE/introspection support."""
  return sorted(list(globals().keys()) + ["torch_zoo"])