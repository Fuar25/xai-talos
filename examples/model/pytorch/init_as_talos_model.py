"""Example: define a model by subclassing talos' TorchModule.

This script is intended to be runnable on its own. It demonstrates a workflow
where a user defines a model that directly inherits from `ta.TorchModule`
(so it is already a Talos + PyTorch module) and then uses it like a regular
`torch.nn.Module`.

Run:
  python -m examples.model.pytorch.init_as_talos_model
"""

from torch import nn

import talos as ta


class MLP(ta.TorchModule):
  """A simple MLP implemented as a Talos TorchModule."""

  def __init__(self, input_size: int = 3072, hidden_size: int = 512, num_classes: int = 10):
    """Initialize the MLP.

    Args:
      input_size: Number of input features (e.g., 32*32*3 for CIFAR-like data).
      hidden_size: Width of the hidden layer.
      num_classes: Number of output classes.
    """
    super().__init__(name="MLP")
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    """Run a forward pass.

    Args:
      x: Input tensor of shape [batch, input_size].

    Returns:
      Logits tensor of shape [batch, num_classes].
    """
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    return out


def main() -> None:
  """Run the example."""
  model = MLP()

  # Prefer CUDA when available, but keep the script runnable on CPU-only setups.
  try:
    model = model.cuda()
  except Exception:
    pass

  model.summary([1, 32 * 32 * 3])
  print("\nDirectly printing the model (should show the MLP architecture):")
  print(model)


if __name__ == "__main__":
  main()

