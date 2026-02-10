"""xai-talos: A decoupled, modular deep learning framework.

talos provides abstractions for deep learning research by isolating:
- Data (data manipulation and loading)
- Evaluation (metrics)
- Model (neural network architecture, algorithm workflow, and model deployment)
- Optimization (gradient-based optimization and hyperparameter tuning)
Each component can be used independently or together.

Philosophy of consistency
-------------------------
I. APIs should be clear and easy to use.
II. The framework should be educational with smooth learning curve.
III. The repo should be easy to contribute to.
"""

import importlib
from typing import Any, TYPE_CHECKING

__version__ = "0.1.0"

# (1) Import level-1 APIs: d-e-m-o and u
from . import data
from . import eval
from . import model
from . import optim

from . import utils

# (2) Import level-2 APIs: wrapping functions and global attributes
# (2.1) Data
# (2.1.1) Data wrapping function
Dataset = data.TalosData.wrap

# (2.3) Model
# (2.3.1) Model wrapping function
Model = model.TalosModel.wrap

# (2.5) Utilities
from .utils import set_seed

# (3) Create miscellaneous attributes
# (3.1) Global attributes
work_dir = utils.file_manager.get_main_file_dir()

# (4) Lazy imports (PEP 562) for avoiding heavy dependencies at import time.
if TYPE_CHECKING:
  # This does not run at runtime, so it does not defeat lazy loading.
  from .model.backends.pytorch.torch_model import TorchModel as TorchModule

def __getattr__(name: str) -> Any:
  """Lazily resolve selected module attributes on first access."""
  if name == "TorchModule":
    torch_model_module = importlib.import_module(
      ".model.backends.pytorch.torch_model",
      __name__,
    )
    torch_module = torch_model_module.TorchModel
    globals()["TorchModule"] = torch_module
    return torch_module
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__() -> list[str]:
  """Return custom dir\(\) output including lazy attributes."""
  return sorted(list(globals().keys()) + ["TorchModule"])