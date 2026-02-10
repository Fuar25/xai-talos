# SPDX-License-Identifier: MIT
"""
TorchModel Class
"""

from talos.model import TalosModel
from typing import Any, Optional, Union

try:
  import torch
  _HAS_TORCH = True
except ImportError:
  torch = None
  _HAS_TORCH = False

_BASE_MODULE = torch.nn.Module if _HAS_TORCH else object


class TorchModel(_BASE_MODULE, TalosModel):

  def __init__(self, model=None, name: str = "TorchModel",
               model_dir=None, **kwargs):
    # Initialize the base classes
    if _HAS_TORCH: torch.nn.Module.__init__(self)
    TalosModel.__init__(self, name=name, model_dir=model_dir, **kwargs)

    # Store the wrapped model if provided, otherwise use self as the model
    self._model: torch.nn.Module = model

  # region: APIs

  def summary(self, input_size: Union[tuple, list]) -> None:
    """Print a summary of the PyTorch model architecture.

    Args:
        input_size (Optional[tuple]): The size of the input tensor (excluding batch size).
                                      If None, a default size will be used.
    """

    if not _HAS_TORCH:
      raise ImportError("PyTorch is not installed. Cannot print model summary.")

    try:
      from torchsummary import summary
    except ImportError:
      raise ImportError("torchsummary is not installed." 
                        " Please install it to use the summary feature.")

    # Use self if self.model is None, otherwise use self.model for summary
    model = self._model if self._model is not None else self
    device = next(model.parameters()).device
    summary(model.to(device), tuple(input_size))

  def _save(self, file_path: str) -> None:
    torch.save(self.state_dict(), file_path)

  def _load(self, file_path: str, **kwargs) -> None:
    map_location = kwargs.get('map_location', 'cpu')
    state = torch.load(file_path, map_location=map_location)
    self.load_state_dict(state)

  # endregion: APIs

  # region: Delegation

  def forward(self, *args: Any, **kwargs: Any) -> Any:
    """Delegate the forward pass to the wrapped torch.nn.Module."""
    model = self._model if self._model is not None else self
    return model(*args, **kwargs)

  # endregion: Delegation
