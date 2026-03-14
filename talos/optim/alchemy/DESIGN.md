# talos/optim/alchemy

## 1. Design Philosophy

Alchemy sits above the trainer layer. It automates configuration space exploration
via pluggable sampling strategies, making well-tuned models the default rather than
the exception. The minimal API requires only an **objective function** and a **search space**.

## 2. Minimal Use Case

```python
from talos.optim.alchemy import RandomSearch, ContinuousParam, CategoricalParam

space = {
  'lr':          ContinuousParam(1e-4, 1e-1, log=True),
  'hidden_size': CategoricalParam([32, 64, 128]),
}

def objective(params):
  model   = build_model(params['hidden_size'])
  trainer = TorchTrainer(model, loss_fn='mse', lr=params['lr'])
  trainer.train(train_set, val_set=val_set, max_iterations=200)
  return trainer.history.best('val/mse')[1]

history = RandomSearch().run(objective, space, n_trials=20)

print(history.best_params)
print(history.best_score)
```

## 3. Components

### 3.1. Config Knobs (base, all strategies)

| Knob | Type | Default | Description |
|---|---|---|---|
| `n_trials` | int | 20 | Number of search trials (positive) |
| `direction` | str | `'minimize'` | Optimization direction: `'minimize'` or `'maximize'` |
| `verbose` | bool | True | Print trial progress |

### 3.2. Search Space (`space.py`)

All hyperparameters in a space dict must be Param instances. Fixed values belong
outside the space as named constants.

| Class | Description |
|---|---|
| `ContinuousParam(low, high, log=False)` | Uniform or log-uniform float sampling |
| `IntParam(low, high)` | Uniform integer sampling (both endpoints inclusive) |
| `CategoricalParam(choices)` | Random choice from a list |

`sample_space(space)` draws one random sample from the full space dict.

### 3.3. Search Strategies

All strategies share the same `run(objective, space, n_trials)` API.
Optuna is used as the sampling backend.

| Class | Sampler | Extra Config |
|---|---|---|
| `RandomSearch` | `RandomSampler` | — |
| `GridSearch` | `GridSampler` | `n_points` (default 5): grid resolution per continuous/integer param |
| `BayesianSearch` | `TPESampler` | `n_initial` (default 10): random bootstrap trials before TPE |

`GridSearch` ignores `n_trials`; it runs all grid combinations (total = product of axis lengths).

### 3.4. SearchHistory (`history.py`)

Flat list of trial results. Each trial has a `TrialRecord(trial_id, params, score)`.

- `record(params, score)` — Append a trial result.
- `trials` — All trial records.
- `best_trial` / `best_params` / `best_score` — Best result (respects `direction`).

### 3.5. Extension Point

To add a custom strategy, subclass `AlchemySearch` and override `_create_sampler(space)`:

```python
class MySearch(AlchemySearch):
  def _create_sampler(self, space):
    import optuna
    return optuna.samplers.CmaEsSampler()
```

## 4. Decided

- **Objective ownership**: All training details are encapsulated in the user-provided
  `objective` function. Search classes only sample params and record scores.
- **Search space contract**: All values in `space` must be Param instances.
- **Optuna backend**: All strategies delegate to Optuna samplers. The extension point
  is `_create_sampler(space)` — override to plug in any Optuna-compatible sampler.
- **SearchHistory vs TrainingHistory**: Separate classes — TrainingHistory records a
  time series per training run; SearchHistory records a flat list of trial scalars.

## 5. TODO

### 5.1. K-Fold Cross-Validation
- Add `k_folds` config knob once `eval.cross_validate()` is available.
- Each trial runs K-fold CV and reports the mean score.
