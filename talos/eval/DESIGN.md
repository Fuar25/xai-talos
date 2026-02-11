# Eval Design Document

## Design Philosophy

Eval provides **metric definitions** and **evaluation framework**. It does not own
training or tuning — it provides the tools that trainer and alchemy consume.

## Two Layers

### Layer 1: Metric Definitions (Implemented)

Callable class instances with numpy base + backend-specific overrides.

```
TalosMetric (numpy base)        ← __call__ uses _np_compute()
├── MSE, MAE, CrossEntropy, BinaryCrossEntropy, Accuracy
└── TorchMetric                 ← adds .numpy() back door
    └── TorchMSE, TorchMAE, TorchCrossEntropy, TorchBinaryCrossEntropy, TorchAccuracy
```

**Class attributes:**
- `name`: identifier (e.g., `'mse'`)
- `differentiable`: whether usable as training loss
- `direction`: `'minimize'` or `'maximize'` (used by alchemy + early stopping)

**Resolution API** — `get_torch_metric(spec)`:
```python
get_torch_metric('mse')          # string → TorchMSE()
get_torch_metric(TorchMSE)       # class → TorchMSE()
get_torch_metric(TorchMSE())     # instance → passthrough
```

### Layer 2: Evaluation Framework (Future)

- `evaluate(model, test_data)` → compute metrics, print tables/confusion matrix
- `cross_validate(model_def, data, k, ...)` → k-fold CV, returns results for alchemy

## Interaction Map

```
eval.get_torch_metric ──→ TorchTrainer._get_loss_function (loss resolution)
eval.cross_validate ──→ alchemy (CV results for HP selection) [future]
eval.evaluate ──→ user (test-time reporting) [future]
```

## Decided

- Loss functions are **callable class instances**, not plain functions
- Numpy base implementation + backend overrides (torch, etc.)
- Backend metrics expose `.numpy()` back door for alternative computation
- Each backend has its own registry + `get_*_metric()` API
- Trainer receives loss at init-time (`self.loss_functions`) or train-time override
