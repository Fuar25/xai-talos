# import something


def check_torch():
  # Make sure PyTorch is installed
  try:
    import torch
  except ImportError:
    print("!! PyTorch is not installed. Please install it to run the code.")
    return

  # If PyTorch is installed with CUDA support, print the version and available devices
  print(f":: PyTorch version: {torch.__version__}")
  if torch.cuda.is_available():
    print(f":: CUDA is available. Number of CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
      print(f"   - Device {i}: {torch.cuda.get_device_name(i)}")
  else:
    print(" ! CUDA is not available. Running on CPU.")


def add_necessary_paths():
  import sys, os

  # Add the parent directory of talos, i.e., xai-talos
  # Note the path of this file is xai-talos\tutorials\PINNs\utils\__init__.py
  current_dir = os.path.dirname(os.path.abspath(__file__))
  project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
  if project_root not in sys.path:
    sys.path.insert(0, project_root)
