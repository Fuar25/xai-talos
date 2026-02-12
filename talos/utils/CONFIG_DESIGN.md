# talos/utils/config

Typed, serializable configuration system. A `Config` holds a collection of **knobs** — typed entries with validation on assignment.

## 1. Knob Types

```
Knob (base)
├── IntKnob         # int, supports constraints (positive, non_negative)
├── FloatKnob       # float, supports constraints (positive, non_negative)
├── StrKnob         # str
├── CategoricalKnob # one of a fixed set of options
└── BoolKnob        # bool
```

All knobs are nullable (any knob can be `None`, meaning "not configured"). `IntKnob`/`FloatKnob` validate via `check_type()`.

### 1.1. Knob Attributes

```python
class Knob:
  name: str           # identifier, e.g., 'batch_size'
  default: Any        # default value (can be None)
  description: str    # human-readable description (default '')
```

## 2. Config API

- **Attribute access**: `__setattr__` validates via knob; `__getattr__` returns value or default.
- **Registration**: `register(knob)`, `register_int()`, `register_float()`, `register_str()`, `register_bool()`, `register_categorical(options=...)`. All accept `default` and `description` (default `''`).
- **Serialization**: `to_dict()` → `OrderedDict`; `to_yaml(path=None)` → string or file; `from_yaml(path, group=None, auto_register=False)` loads from YAML with optional dot-separated group navigation.

## 3. Usage

```python
# (1) Set knobs via attribute assignment (type-checked)
cfg = Config('trainer')
cfg.register_int('batch_size', default=-1)
cfg.register_bool('early_stop', default=False)
cfg.batch_size = 32
cfg.early_stop = True

# (2) Inspect
cfg.to_dict()    # → OrderedDict([('batch_size', 32), ('early_stop', True)])

# (3) YAML serialization
yaml_str = cfg.to_yaml()              # no path → return string
cfg.to_yaml('experiment.yaml')        # path → write to file

# (4) YAML loading (hierarchical group support)
cfg.from_yaml('experiment.yaml')                          # load all
cfg.from_yaml('experiment.yaml', group='trainer')         # navigate to group
cfg.from_yaml('exp.yaml', group='outer.inner')            # nested group
cfg.from_yaml('exp.yaml', auto_register=True)             # auto-register unknown knobs

# (5) Register custom knobs
cfg.register_float('clip_norm', default=None, positive=True)
cfg.register_categorical('scheduler', options=('step', 'cosine', 'linear'))
cfg.register(some_knob_instance)      # accept Knob instance directly
```

## 4. YAML Format

```yaml
trainer:                            # Config.name = 'trainer'
  batch_size: 32
  early_stop: true
  patience: 10

model:                              # Config.name = 'model'
  dropout: 0.5
```

## 5. Priority System

`train()` signature args override config values:
```python
trainer.config.batch_size = 64
trainer.train(train_set, max_iterations=100, batch_size=32)  # 32 wins
```

## 6. Integration Pattern

Modules should own a `.config` property via `@Nomear.property()`, with knobs registered in a `_register_configs()` method. Subclasses extend by overriding `_register_configs()` and calling `super()` first.

```python
class TalosTrainer(Nomear):

  @Nomear.property()
  def config(self):
    cfg = Config(name='trainer')
    self._register_configs(cfg)
    return cfg

  def _register_configs(self, config):
    config.register_int('batch_size', default=-1)
    config.register_bool('early_stop', default=False)


class TorchTrainer(TalosTrainer):

  def _register_configs(self, config):
    super()._register_configs(config)
    config.register_categorical('device', default='auto',
                                options=('auto', 'cpu', 'cuda'))
```

## 7. Decided

- Composition over inheritance: modules own `.config`, not extend Config.
- `Config.name` is used as the top-level YAML group key.
- `from_yaml(auto_register=True)` infers knob type from Python value type via `_infer_knob()`.

## 8. TODO

### 8.1. Unified Config Export

Export all configs (model, data, trainer) into a single YAML for full experiment reproducibility.

Envisioned API:
```python
from talos.utils.config import export_configs
export_configs('experiment_001.yaml', trainer.config, model.config, data.config)
```

```yaml
# experiment_001.yaml
trainer:
  batch_size: 32
  early_stop: true
  patience: 10
model:
  hidden_dim: 256
  dropout: 0.5
data:
  split_ratio: 0.8
```

### 8.2. Hyper-Parameter Search Space

Allow knobs to specify a search space instead of a fixed value. Passing such a config to `talos/optim/alchemy` triggers hyper-parameter search.

Envisioned API:
```python
from talos.utils.config import SearchSpace

trainer.config.set_space('batch_size', SearchSpace([16, 32, 64]))
trainer.config.set_space('learning_rate', SearchSpace(low=1e-5, high=1e-2, log=True))

from talos.optim import alchemy
best_config = alchemy.search(trainer, train_set, val_set)
```
