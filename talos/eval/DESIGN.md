# talos/eval

Metric definitions and evaluation framework. Does not own training or tuning — provides tools that trainer and alchemy consume.

## 1. Metric Definitions

Callable class instances with numpy base + backend-specific overrides.

```
TalosMetric (numpy base)        ← __call__ uses _np_compute()
├── MSE, MAE, CrossEntropy, BinaryCrossEntropy, Accuracy
└── TorchMetric                 ← adds .numpy(): detach/cpu/convert → _np_compute()
    └── TorchMSE, TorchMAE, TorchCrossEntropy, TorchBinaryCrossEntropy, TorchAccuracy
```

**Class attributes:**
- `name`: identifier (e.g., `'mse'`)
- `differentiable`: whether usable as training loss
- `direction`: `'minimize'` or `'maximize'` (used by alchemy + early stopping)

**Resolution API** — `get_torch_metric(spec)` resolves string/class/instance into a metric. Backed by `TORCH_METRIC_REGISTRY` (dict mapping name → class).
```python
get_torch_metric('mse')          # string → registry lookup → TorchMSE()
get_torch_metric(TorchMSE)       # class → TorchMSE()
get_torch_metric(TorchMSE())     # instance → passthrough
```

## 2. Interaction Map

```
eval.get_torch_metric ──→ TorchTrainer._get_loss_function (loss resolution)
eval.cross_validate ──→ alchemy (CV results for HP selection) [future]
eval.evaluate ──→ user (test-time reporting) [future]
```

## 3. Decided

- Loss functions are **callable class instances**, not plain functions.
- Numpy base implementation + backend overrides (torch, etc.).
- Backend metrics expose `.numpy()` for numpy-based computation from tensor inputs.
- Each backend has its own registry + `get_*_metric()` API.
- Trainer receives loss at init-time (`self.loss_functions`) or train-time override.

## 4. TODO

### 4.1. Evaluation Framework

`evaluate(model, test_data)` → compute metrics, print tables/confusion matrix.

### 4.2. Cross-Validation

`cross_validate(model_def, data, k, ...)` → k-fold CV, returns results for alchemy HP selection.
