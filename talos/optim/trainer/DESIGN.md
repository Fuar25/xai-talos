# talos/optim/trainer

## 1. Design Philosophy

Simplicity for beginners, flexibility for experts. The minimal API requires only
**model**, **data**, and **loss** — optimizer and batch size have sensible defaults.

## 2. Minimal Use Case

```python
trainer = TorchTrainer(model, loss_fn='mse')      # default optimizer='sgd'
trainer.train(train_set, max_iterations=100)        # default batch_size=-1 (full batch)
```

## 3. Training Loop

```
Preparation
for each iteration:
  Get batch → predict → compute total loss → backpropagate and update
  Periodically validate → track improvement, save best model
  Stop early if required
Restore best model if saved
```

## 4. Components

### 4.1. Config Knobs

| Knob | Type | Default | Description |
|---|---|---|---|
| `batch_size` | int | -1 | Batch size. -1 for full batch |
| `max_iterations` | int | None | Number of training iterations (positive) |
| `early_stop` | bool | False | Enable early stopping |
| `patience` | int | 10 | Early stopping patience (positive) |
| `validate_every` | int | 100 | Validate every N iterations (positive) |
| `val_ratio` | float | None | Auto-split ratio for validation set (positive) |
| `val_metrics` | str | None | Comma/semicolon-separated metric names for validation |
| `print_every` | int | 100 | Print training progress every N iterations (positive) |
| `save_best` | bool | True | Save best model weights during validation |

### 4.2. TrainingHistory

Track-based recording where each metric is an independent time series keyed by `{group}/{name}` (e.g., `train/mse`, `val/accuracy`).

- `record(metric, iteration, value, group)` — Append a (iteration, value) entry to a track.
- `latest(key)` / `best(key)` — Most recent or best (respects metric direction) entry.
- `values(key)` / `iterations(key)` — All values or iteration numbers for a track (for plotting).
- `improved(key)` — Whether the latest value improved over the previous best.
- `tracks` — List all track keys.

### 4.3. Backend-Specific Methods

| Method | Base (TalosTrainer) | Torch (TorchTrainer) |
|---|---|---|
| `_validate_optimizer` | abstract | Resolves str/class/instance → `torch.optim.Optimizer` |
| `_backward_and_update` | abstract | `zero_grad()` → `backward()` → `step()` |
| `_resolve_metric` | abstract | Delegates to `get_torch_metric(spec)` |
| `_prepare_batch` | pass-through | numpy → `torch.tensor` on model device; 1D → single sample |
| `_validate` | compute metrics + update patience + checkpoint | wraps with `model.eval()` + `no_grad()` |
| `_save_checkpoint` | abstract | `copy.deepcopy(model.state_dict())` |
| `_restore_checkpoint` | abstract | `model.load_state_dict(checkpoint)` |

## 5. Decided

- **Loss ownership**: Eval module defines metrics/losses; trainer consumes them
- **Loss API**: Callable class instances via `get_torch_metric(spec)` (case-insensitive)
- **Loss resolution**: `_get_loss_function` checks train-time arg (priority) → `self.loss_functions` OrderedDict (1st entry) → error if none
- **Model loss**: After computing data loss, the trainer calls `model.model_loss(X, outputs, Y)`. Non-None results are added to the total loss. This enables model-specific losses (e.g., PDE residuals in PINNs) without coupling the trainer to specific model types
- **Loss must be specified**: No default loss — user must provide `loss_fn` at init or train time
- **Default optimizer**: `'sgd'`
- **Default batch_size**: `-1` (full batch)
- **Data → Tensor**: Backend-specific `_prepare_batch()` converts numpy → tensors on model's device
- **`train_set.sample(-1)`**: Returns `self` (full dataset, no copy)
- **Config system**: `Config` with typed knobs via `register_int`, `register_float`, etc.; params resolve via train() arg (priority) → config → error
- **Validation set**: explicit `val_set` (priority) → auto-split via `config.val_ratio` → None
- **Validation metrics**: train() arg (priority) → `config.val_metrics` string → [loss_fn]
- **Early stopping**: `TrainState` holds per-session state (`es_key`, `patience_counter`); `_validate` updates counter; `_should_stop` checks criteria
- **`TrainState`**: Per-session state container created fresh each `train()` call; extensible for progress bar info
- **Model checkpointing**: In-memory only. `save_best=True` by default; saves on validation improvement, restores after training loop. Only activates with a validation set present. Backend implements `_save_checkpoint()` / `_restore_checkpoint()`

## 6. TODO

### 6.1. Model Checkpointing
- ~~In-memory best-model checkpointing~~ — **Done**: `save_best` config knob, `_save_checkpoint()` / `_restore_checkpoint()` abstract methods
- TODO: Disk-based checkpointing for long training runs

### 6.2. Regularization
- Add `regularizer` parameter to `TalosTrainer.__init__` for weight penalties (L1, L2, etc.).
- API: `reg_loss = self.regularizer(self.model)` — depends only on model parameters, not on X/Y/outputs.
- Combined in training loop: `total_loss = data_loss + model_loss + reg_loss`.

### 6.3. Progress Bar & Console Output
- **Done**: Basic console printing via `print_every` config knob. Prints iteration + train loss (`>> Iter N | train/metric = value`), validation metrics (`.. val/metric = value`), and `[Best]` notifications when early stopping metric improves. Start/end banners with early stopping message.
- TODO: Progress bar with ETA (use `Console.print_progress`), verbosity levels
