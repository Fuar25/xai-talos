# talos/utils

Shared utilities used across all talos subpackages. Check here before implementing common functionality.

## 1. Modules

- **`check_type`** (`censor.py`) — Runtime type validation with auto-conversion and constraints. Supports `INT_TYPES`, `FLOAT_TYPES`, `NUMBER_TYPES` (includes NumPy types). Handles: single→tuple/list conversion, string→number, nullable, positivity constraints.
- **`Nomear`** (`nomear.py`) — Base class with cloud (shared) and local (instance) pocket dictionaries. Provides `@Nomear.property()` for lazy initialization.
- **`Config`** (`config.py`) — Typed configuration via knobs (`IntKnob`, `FloatKnob`, `StrKnob`, `BoolKnob`, `CategoricalKnob`). Validates on assignment. YAML serialization/deserialization.
- **`set_seed`** (`reproducibility.py`) — Set random seeds across Python, NumPy, PyTorch, TensorFlow (best-effort).
- **`Console`** (`console/`) — Terminal output with ANSI colors, progress bars with ETA, structured prompts.
- **`ordinal`** (`format/`) — Integer to ordinal string (1→"1st").
- **`file_manager`** (`io/`) — `get_main_file_dir()` returns the entry-point script's directory.
- **`backends/pytorch`** — One-time PyTorch/CUDA environment report on import. Controlled by `TALOS_TORCH_BACKEND_REPORT` env var.
