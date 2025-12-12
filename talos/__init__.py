"""
xai-talos: A decoupled, modular deep learning framework.

This framework provides clean abstractions for deep learning research by isolating:
- Data (Task)
- Architecture (Model)  
- Optimization (Trainer)
"""

__version__ = "0.1.0"

from . import tasks
from . import models
from . import trainers
from . import utils

__all__ = ["tasks", "models", "trainers", "utils"]
