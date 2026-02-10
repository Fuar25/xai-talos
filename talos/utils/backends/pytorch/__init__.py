"""PyTorch backend utilities.

This module is imported lazily (only when a user needs PyTorch features), so it may
safely attempt to import torch without requiring it for the rest of the package.

If torch is available, this module can optionally print a one-time, human-friendly
backend report (version + CUDA/GPU info). Disable it by setting:
  TALOS_TORCH_BACKEND_REPORT=0
"""

import os

try:
  import torch
  has_torch = True
except ImportError:
  torch = None
  has_torch = False


_has_reported = False


def _format_bool(value: bool) -> str:
  """Format a boolean as a human-friendly string."""
  return "yes" if value else "no"


def _report_disabled() -> bool:
  """Return True if backend reporting is disabled via environment variable."""
  return os.getenv("TALOS_TORCH_BACKEND_REPORT", "1").strip() in {
    "0", "false", "False", "no", "NO",
  }


def _safe_get_gpu_names() -> list[str]:
  """Best-effort retrieval of GPU device names."""
  if not has_torch:
    return []
  try:
    count = torch.cuda.device_count()
  except Exception:
    return []
  names: list[str] = []
  for i in range(count):
    try:
      names.append(torch.cuda.get_device_name(i))
    except Exception:
      names.append(f"cuda:{i}")
  return names


def _report_torch_backend_once() -> None:
  """Print a one-time PyTorch backend report (best-effort)."""
  global _has_reported
  if _has_reported:
    return

  # (1) Allow users/CI to disable printing.
  if _report_disabled():
    _has_reported = True
    return

  # (2) Report missing optional dependency explicitly (still one-line + compact).
  if not has_torch:
    print("[talos.pytorch] torch=unavailable; cuda=no; GPU 0/0")
    _has_reported = True
    return

  # (3) Gather details defensively (no hard failures).
  torch_version = getattr(torch, "__version__", "unknown")

  try:
    cuda_available = bool(torch.cuda.is_available())
  except Exception:
    cuda_available = False

  try:
    cuda_version = torch.version.cuda if getattr(torch, "version", None) is not None else None
  except Exception:
    cuda_version = None

  try:
    gpu_count = int(torch.cuda.device_count()) if cuda_available else 0
  except Exception:
    gpu_count = 0

  gpu_names = _safe_get_gpu_names()
  gpu0 = gpu_names[0] if gpu_names else None

  # (4) Print compact, readable, one-line info.
  cuda_part = f"cuda={_format_bool(cuda_available)}"
  if cuda_available and cuda_version:
    cuda_part += f" (v{cuda_version})"

  gpu_part = f"GPU {gpu_count}/{gpu_count}"
  if gpu0:
    gpu_part += f'="{gpu0}"'

  print(f"[talos.pytorch] torch={torch_version}; {cuda_part}; {gpu_part}")
  _has_reported = True


_report_torch_backend_once()
