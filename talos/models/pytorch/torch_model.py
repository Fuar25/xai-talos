# SPDX-License-Identifier: MIT
"""
TorchModel Class
"""

from talos.models import TalosModel
from typing import Any, Optional, Union

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False


class TorchModel(TalosModel):

  def __init__(self, model, name: str = "TorchModel", model_dir=None, **kwargs):
    super().__init__(name=name, **kwargs)
    self.model: torch.nn.Module = model

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

    device = next(self.model.parameters()).device
    summary(self.model.to(device), tuple(input_size))

  def _save(self, file_path: str) -> None:
    torch.save(self.model.state_dict(), file_path)

  def _load(self, file_path: str, **kwargs) -> None:
    map_location = kwargs.get('map_location', 'cpu')
    state = torch.load(file_path, map_location=map_location)
    self.model.load_state_dict(state)

  # endregion: APIs

  # region: Delegation

  def forward(self, *args: Any, **kwargs: Any) -> Any:
    """Delegate the forward pass to the wrapped torch.nn.Module."""
    return self.model(*args, **kwargs)


  def __getattr__(self, item: str) -> Any:
    """
    Delegate attribute access to the inner torch.nn.Module when the
    attribute is not found on this wrapper or TalosModel.
    """
    # Called only if normal attribute lookup fails.
    model = self.__dict__.get("model", None)
    if model is not None and hasattr(model, item):
      return getattr(model, item)
    raise AttributeError(
      f"{self.__class__.__name__!s} has no attribute {item!r}")


  def to(self, *args: Any, **kwargs: Any) -> "TorchModel":
    """Optionally override TalosModel.to to also move the inner torch model."""
    if hasattr(super(), "to"): super().to(*args, **kwargs)
    self.model.to(*args, **kwargs)
    return self

  # endregion: Delegation
