# SPDX-License-Identifier: MIT
"""
Talos Model Module
"""

from talos.utils import Nomear
from talos.utils.config import Config as ConfigBase

import os


class TalosModel(Nomear):
  """A base class for Talos models, supporting different backends such as PyTorch.

  Current Addons:
  (1) Path management for saving and loading models, with a default directory
      structure under the Talos work directory:

      |-- work_dir/
        |-- models/
          |-- {model_name}/
            |-- ckpt_path_1.pt
            |-- ckpt_path_2.pt
        |-- data/
        |-- model_1.py
        |-- model_2.py

  (2) Coming soon ...
  """

  SCOPE = 'talos.model'

  class Config(ConfigBase):
    pass  # No knobs yet; subclasses extend via inheritance.

  CHECK_POINT_EXT = ".pt"

  def __init__(self, name: str = "TalosModel", model_dir=None, **kwargs):
    """Initialize the TalosModel with a name.

    Args:
        name (str): The name of the model. Defaults to "TalosModel".
    """
    self.name = name
    self.model = None
    self.configs = kwargs

    self._model_dir = model_dir

  # region: Properties

  @Nomear.property()
  def config(self):
    return type(self).Config(name='model')

  @property
  def model_dir(self) -> str:
    """Get the directory where the model is saved.
    """
    if self._model_dir is not None:
      # Use the user-specified model directory if provided
      model_dir = self._model_dir
    else:
      # Default to the Talos work directory
      import talos
      model_dir = os.path.join(talos.work_dir, "models", self.name)

    # Create the directory if it does not exist
    os.makedirs(model_dir, exist_ok=True)

    return model_dir

  @model_dir.setter
  def model_dir(self, dir_path: str) -> None:
    """Set the directory where the model is saved."""
    self._model_dir = dir_path

  # endregion: Properties

  # region: Wrapping

  @classmethod
  def wrap(cls, model, name=None, work_dir=None, **kwargs):
    """Wrap a given model into a TalosModel subclass based on its type.
    Note that the default file structure is:

    |-- work_dir/
      |-- models/
        |-- {model_name}/
          |-- ckpt_path_1.pt
          |-- ckpt_path_2.pt
      |-- data/
      |-- model_1.py
      |-- model_2.py

    Args:
        model: The model instance to be wrapped.
        name (str): The name of the model (model_name).
                    If None, a default name will be used.
        work_dir (str): The working directory where the model will be saved.
    """
    # If name is not provided, use the class name of the model as the default name
    if name is None: name = type(model).__name__

    kwargs['name'] = name
    kwargs['model_dir'] = work_dir

    try:
      import torch
      if isinstance(model, torch.nn.Module):
        cls.print("Detected PyTorch model. Wrapping with TorchModel ...")
        from talos.model.backends.pytorch.torch_model import TorchModel
        return TorchModel(model, **kwargs)
    except:
      pass

    try:
      from talos.model.backends import tensorflow as tf
      if isinstance(model, tf.keras.Model):
        cls.print("Detected TensorFlow Keras model. Wrapping with KerasModel.")
        # TODO
        return None
    except:
      pass

    raise TypeError(f"Unsupported model type for TalosModel: {type(model)}")

  # endregion: Wrapping

  # region: Abstract Methods

  def forward(self, *args, **kwargs):
    raise NotImplementedError

  def model_loss(self, X, outputs, Y):
    """Compute model-specific loss (e.g., PDE residual for PINNs).

    Override in subclasses to inject custom loss terms. Called by the trainer
    after the data loss; non-None return values are added to the total loss.

    Args:
      X: Input tensor (needed for autograd-based derivative computation).
      outputs: Model forward pass outputs.
      Y: Target tensor (may be None for self-supervised losses).

    Returns:
      Loss tensor, or None if no model-specific loss.
    """
    return None

  def _predict(self, X):
    """Backend-specific prediction. Override in subclasses."""
    return self.forward(X)

  # endregion: Abstract Methods

  # region: APIs

  def predict(self, input):
    """Run inference on numpy array, tensor, or TalosData.

    Args:
      input: numpy array, backend tensor, or TalosData instance.

    Returns:
      Prediction result (format depends on backend).
    """
    from talos.data.talos_data import TalosData
    if isinstance(input, TalosData):
      input = input.X
    return self._predict(input)

  def summary(self, *args, **kwargs) -> None:
    """Print a summary of the model architecture."""
    raise NotImplementedError(
      "The summary method must be implemented in subclasses.")

  def save(self, ckpt_suffix: str = None, model_dir: str = None) -> None:
    """Save the model to the specified filepath. File name format:
      {model_dir}/{model_name}{ckpt_suffix}{CHECK_POINT_EXT}

    Args:
        ckpt_suffix (str): Optional suffix to append to the checkpoint filename.
        model_dir (str): Optional directory to save the model. If None, uses
                         the default model directory.
    """
    if ckpt_suffix is None: ckpt_suffix = '_no_suffix'
    if model_dir is None: model_dir = self.model_dir
    file_path = os.path.join(
      model_dir, f'{self.name}{ckpt_suffix}{self.CHECK_POINT_EXT}')
    self._save(file_path)
    self.print(f'Model saved to `{file_path}`.')

  def _save(self, file_path: str) -> None:
    """Save the model to the specified filepath."""
    raise NotImplementedError(
      "The save method must be implemented in subclasses.")

  def load(self, file_path: str = None, **kwargs) -> 'TalosModel':
    """Load the model from the specified filepath."""
    if file_path is None:
      model_dir = self.model_dir
      ckpts = [
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir) if f.endswith(self.CHECK_POINT_EXT)]

      if not ckpts: raise FileNotFoundError(
        f"No checkpoint files found in {model_dir}")

      file_path = max(ckpts, key=os.path.getmtime)

    self._load(file_path, **kwargs)
    self.print(f'Model loaded from `{file_path}`.')

  def _load(self, file_path: str, **kwargs) -> 'TalosModel':
    """Load the model from the specified filepath."""
    raise NotImplementedError(
      "The load method must be implemented in subclasses.")

  # endregion: APIs
