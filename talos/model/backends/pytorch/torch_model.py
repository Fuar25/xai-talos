# SPDX-License-Identifier: MIT
"""
TorchModel Class
"""

from typing import Any, Union

from talos.model import TalosModel
from talos.utils.backends.pytorch import has_torch, torch
from talos.utils import check_type

_BASE_MODULE = torch.nn.Module if has_torch else object


class TorchModel(_BASE_MODULE, TalosModel):

  def __init__(self, model=None, name: str = "TorchModel",
               model_dir=None, **kwargs):
    """Initialize a TorchModel wrapper.

    Args:
      model: The `torch.nn.Module` to wrap.
      name: Logical model name.
      model_dir: Optional directory for artifacts.
      **kwargs: Forwarded to `TalosModel`.
    """
    # (1) Initialize the base classes.
    if has_torch: torch.nn.Module.__init__(self)
    TalosModel.__init__(self, name=name, model_dir=model_dir, **kwargs)

    # (2) Store the wrapped model.
    self._model: "torch.nn.Module" = model

  # region: APIs

  def summary(self, input_size: Union[tuple, list, int]) -> None:
    """Print a summary of the PyTorch model architecture.
  
    Args:
      input_size: The size of the input tensor (excluding batch size).
        (1) Accepts a `tuple` or `list` of dims, or a single positive `int`
        which will be converted to a one-element `tuple`.
    """
    # (1) Ensure PyTorch is available.
    if not has_torch:
      raise ImportError("PyTorch is not installed. Cannot print model summary.")
  
    # (1.2) Import the optional dependency `torchsummary`.
    try:
      from torchsummary import summary
    except ImportError as exc:
      raise ImportError(
        "torchsummary is not installed. Please install it to use the summary feature."
      ) from exc

    # (2) Normalize `input_size` to a tuple of positive ints.
    input_size = check_type(input_size, tuple, inner_type=int, positive=True)

    # (3) Summarize the wrapped module when present; otherwise summarize self.
    model = self._model if self._model is not None else self

    # (3.1) Prefer CUDA when available; otherwise use CPU.
    preferred_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # (3.2) Force a single device for the summary call (fixes mixed-device errors).
    try:
      model = model.to(preferred_device)
      device = str(preferred_device)
    except Exception:
      model = model.to("cpu")
      device = "cpu"

    summary(model, tuple(input_size), device=device)

  def _save(self, file_path: str) -> None:
    """Save model state to disk."""
    if not has_torch:
      raise ImportError("PyTorch is not installed. Cannot save model.")
    torch.save(self.state_dict(), file_path)

  def _load(self, file_path: str, **kwargs) -> None:
    """Load model state from disk."""
    if not has_torch:
      raise ImportError("PyTorch is not installed. Cannot load model.")
    map_location = kwargs.get("map_location", "cpu")
    state = torch.load(file_path, map_location=map_location)
    self.load_state_dict(state)

  # endregion: APIs

  # region: Delegation

  def forward(self, *args: Any, **kwargs: Any) -> Any:
    """Delegate the forward pass to the wrapped torch.nn.Module."""
    model = self._model if self._model is not None else self
    return model(*args, **kwargs)

  def _predict(self, X):
    """Run inference with numpy/tensor input, returns numpy array."""
    import numpy as np
    device = next(self.parameters()).device
    # (1) Convert numpy to tensor.
    if isinstance(X, np.ndarray):
      X = torch.tensor(X, dtype=torch.float32, device=device)
    if X.ndim == 1: X = X.unsqueeze(0)
    # (2) Eval mode + no_grad.
    self.eval()
    with torch.no_grad():
      out = self.forward(X)
    self.train()
    # (3) Return as numpy.
    return out.cpu().numpy()

  # endregion: Delegation
