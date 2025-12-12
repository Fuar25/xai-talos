"""
xai-talos: A decoupled, modular deep learning framework.

This framework provides clean abstractions for deep learning research by isolating:
- Data (Task)
- Architecture (Model)  
- Optimization (Trainer)
"""

__version__ = "0.1.0"

# Imports
from . import tasks
from . import models
from . import trainers
from . import utils

# Global attributes
work_dir = utils.file_manager.get_main_file_dir()

# Model wrapping function
wrap = models.TalosModel.wrap

