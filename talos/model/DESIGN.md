# talos/model

Backend-agnostic model management. Wraps framework-specific models (PyTorch, TensorFlow, etc.) behind a unified interface for saving, loading, configuration, and architecture inspection.

## 1. Structure

```
TalosModel (base)              <- save/load, config, model_dir management
└── TorchModel                 <- torch.nn.Module + TalosModel, state_dict persistence
```

**Zoo** — Pre-built architectures:
- `MLP` (PyTorch) — Configurable multi-layer perceptron with activation, dropout, normalization, residual connections, pre/post-activation ordering.

## 2. TalosModel API

- `wrap(cls, model, name=None, work_dir=None, **kwargs)` — Factory: detects backend, returns appropriate subclass (e.g., `torch.nn.Module` → `TorchModel`).
- `save(ckpt_suffix=None, model_dir=None)` — Save model state to `{model_dir}/{name}{suffix}.pt`.
- `load(file_path=None, **kwargs)` — Load model state. If no path given, finds most recent checkpoint in `model_dir`.
- `summary(input_size)` — Print model architecture summary.
- `forward(*args, **kwargs)` — Forward pass (abstract, implemented by backends).
- `predict(input)` — Inference API. Accepts numpy array, tensor, or `TalosData`. 1D input treated as single sample. Delegates to backend `_predict()`.
- `config` — Lazy `Config` property via `@Nomear.property()`. Subclasses extend by defining an inner `Config` class (see `CONFIG_DESIGN.md` s6).
- `model_dir` — Default: `{talos.work_dir}/models/{name}/`. Auto-creates directory.

## 3. TorchModel

Inherits both `torch.nn.Module` and `TalosModel`. Wraps an optional `_model` (any `nn.Module`) and delegates `forward()` to it. `_predict()` handles numpy→tensor conversion, runs in `eval()` + `no_grad()`, returns numpy. Saves/loads via `state_dict()`. Summary via `torchsummary` (optional dependency).

## 4. Zoo Access

Lazy-loaded to avoid importing PyTorch at `import talos.model`:
```python
from talos.model import torch_zoo
mlp = torch_zoo.MLP(784, [128, 64], 10, activation='gelu', residual=True)
```

## 5. Decided

- Factory pattern (`wrap()`) for backend detection — extensible to new backends.
- Lazy PyTorch imports throughout (module-level `__getattr__`, class-level `_BASE_MODULE`).
- Model artifacts stored under `talos.work_dir/models/{name}/`.
- `TorchModel` is both `nn.Module` and `TalosModel` (multiple inheritance).

## 6. TODO

### 6.1. TensorFlow/Keras Backend

`backends/tensorflow/` is a placeholder. Implement `KerasModel` following the same pattern as `TorchModel`.

### 6.2. Expand Model Zoo

Add more pre-built architectures (CNN, Transformer, etc.) under `zoo/pytorch/`.
