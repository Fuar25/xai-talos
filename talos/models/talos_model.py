# SPDX-License-Identifier: MIT
"""
Talos Model Module
"""

from talos.utils import Nomear


class TalosModel(Nomear):
  """A base class for Talos models. """

  def __init__(self, name: str = "TalosModel"):
    """Initialize the TalosModel with a name.

    Args:
        name (str): The name of the model. Defaults to "TalosModel".
    """
    self.name = name
    self.model = None

  # region: Wrapping

  @classmethod
  def wrap(cls, model):
    """Wrap a given model into a TalosModel subclass based on its type.
    """
    try:
      import torch
      if isinstance(model, torch.nn.Module):
        cls.print("Detected PyTorch model. Wrapping with TorchModel ...")
        from talos.models.pytorch.torch_model import TorchModel
        return TorchModel(model)
    except:
      pass

    try:
      import tensorflow as tf
      if isinstance(model, tf.keras.Model):
        cls.print("Detected TensorFlow Keras model. Wrapping with KerasModel.")
        # TODO
        return None
    except:
      pass

    raise TypeError(f"Unsupported model type for TalosModel: {type(model)}")

  # endregion: Wrapping


  # region: APIs

  def summary(self, *args, **kwargs) -> None:
    """Print a summary of the model architecture."""
    raise NotImplementedError(
      "The summary method must be implemented in subclasses.")

  # endregion: APIs
