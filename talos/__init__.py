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

__version__ = "0.1.0"

# Imports
from . import data
from . import eval
from . import model
from . import optim

from . import utils

# Global attributes
work_dir = utils.file_manager.get_main_file_dir()

# Data wrapping function
Dataset = data.TalosData.wrap

# Model wrapping function
Model = model.TalosModel.wrap
from .model.backends.pytorch.torch_model import TorchModel as TorchModule

