# Trainer Design Document

## Design Philosophy

Simplicity for beginners, flexibility for experts. The minimal API requires only
**model**, **data**, and **loss** — optimizer and batch size have sensible defaults.

## Minimal Use Case

```python
trainer = TorchTrainer(model, loss_fn='mse')      # default optimizer='sgd'
trainer.train(train_set, max_iterations=100)        # default batch_size=-1 (full batch)
```

## Training Loop

```python
def train(self, train_set, max_iterations, batch_size=-1, loss_fn=None):
  loss_fn = self._get_loss_function(loss_fn)

  for i in range(max_iterations):
    # (1) Sample data and convert to backend format.
    batch = train_set.sample(batch_size)        # batch_size=-1 → full dataset
    X, Y = self._prepare_batch(batch.X, batch.Y)  # numpy → tensor (backend-specific)

    # (2) Forward pass.
    outputs = self.model.forward(X)

    # (3) Compute loss.
    loss = loss_fn(outputs, Y)                  # eval metric instance (callable)

    # (4) Backward + update (backend-specific).
    self._backward_and_update(loss)             # zero_grad → backward → step
```

## Decided

- **Loss ownership**: Eval module defines metrics/losses; trainer consumes them
- **Loss API**: Callable class instances via `get_torch_metric(spec)` (case-insensitive)
- **Loss resolution**: `_get_loss_function` checks train-time arg (priority) → `self.loss_functions` OrderedDict (1st entry) → error if none
- **Loss must be specified**: No default loss — user must provide `loss_fn` at init or train time
- **Default optimizer**: `'sgd'`
- **Default batch_size**: `-1` (full batch)
- **Data → Tensor**: Backend-specific `_prepare_batch()` converts numpy → tensors on model's device
- **`train_set.sample(-1)`**: Returns `self` (full dataset, no copy)

## Backend-Specific Methods

| Method | Base (TalosTrainer) | Torch (TorchTrainer) |
|---|---|---|
| `_validate_optimizer` | abstract | Resolves str/class/instance → `torch.optim.Optimizer` |
| `_backward_and_update` | abstract | `zero_grad()` → `backward()` → `step()` |
| `_resolve_metric` | abstract | Delegates to `get_torch_metric(spec)` |
| `_prepare_batch` | pass-through | numpy → `torch.tensor` on model device |
