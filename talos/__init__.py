"""
xai-talos: A decoupled, modular deep learning framework.

This framework provides clean abstractions for deep learning research by isolating:
- Data (data manipulation and loading)
- Model (neural network architecture and algorithm pipeline)
- Optimization (gradient-based optimization and hyperparameter tuning)
"""

__version__ = "0.1.0"

# Imports
from . import data
from . import model
from . import optim
from . import utils

# Global attributes
work_dir = utils.file_manager.get_main_file_dir()

# Data wrapping function
Dataset = data.TalosData.wrap

# Model wrapping function
Model = model.TalosModel.wrap

