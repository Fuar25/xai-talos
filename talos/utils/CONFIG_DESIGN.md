# Config Design Document

## Design Philosophy

Config is a **first-class citizen** in talos — a shared, typed, serializable
configuration system used across all modules (trainer, model, data).
Each config is a collection of **knobs** that users can turn.

## Usage

```python
# (1) Minimal — just works with defaults
trainer = TorchTrainer(model, loss_fn='mse')
trainer.train(train_set, max_iterations=10000)

# (2) Config-style — set knobs before training
trainer.config.batch_size = 32
trainer.config.early_stop = True
trainer.config.patience = 10
trainer.train(train_set, max_iterations=10000)

# (3) Inspect all knobs → OrderedDict
trainer.config.to_dict()
# → OrderedDict([('batch_size', 32), ('early_stop', True), ('patience', 10), ...])

# (4) YAML serialization
yaml_str = trainer.config.to_yaml()             # no path → return string
trainer.config.to_yaml('experiment.yaml')        # path → write to file

# (5) YAML loading (hierarchical group support)
trainer.config.from_yaml('experiment.yaml')                       # load all
trainer.config.from_yaml('experiment.yaml', group='trainer')      # load mapping
trainer.config.from_yaml('exp.yaml', group='outer.inner')         # nested mapping
trainer.config.from_yaml('exp.yaml', auto_register=True)          # auto-register unknown knobs

# (6) Pro-user: register custom knobs (no extra imports needed)
trainer.config.register_int('grad_accum_steps', default=1, positive=True,
                            description='Gradient accumulation steps')
trainer.config.register_float('clip_norm', default=None, positive=True,
                              description='Max gradient norm for clipping')
trainer.config.register_bool('use_amp', default=False,
                             description='Enable automatic mixed precision')
trainer.config.register_categorical('scheduler', options=('step', 'cosine', 'linear'),
                                    description='LR scheduler type')
trainer.config.register_str('run_name', default=None,
                            description='Name tag for this training run')
trainer.config.register(some_knob_instance)      # also accept Knob instance directly
```

## Priority System

`train()` signature args override config values:
```python
trainer.config.batch_size = 64
trainer.train(train_set, max_iterations=100, batch_size=32)  # 32 wins
```

## Knob Type Hierarchy

```
Knob (base)
├── IntKnob         # int, supports constraints (positive, non_negative, ...)
├── FloatKnob       # float, supports constraints
├── StrKnob         # str
├── CategoricalKnob # one of a set of string options, e.g., ('r', 'g', 'b')
└── BoolKnob        # special categorical: True / False
```

All knobs are **nullable** — any knob can be set to `None` (meaning "not configured").

### Knob Attributes

```python
class Knob:
  name: str           # identifier, e.g., 'batch_size'
  default: Any        # default value (can be None)
  description: str    # human-readable description
  # Validation uses check_type() internally
```

## Config Class

```python
class Config:
  name: str                         # YAML top-level group (e.g., 'trainer', 'model')

  def __setattr__(name, value)      # type-checks via Knob on assignment
  def __getattr__(name)             # returns value or default

  # Serialization
  def to_dict() -> OrderedDict      # all knobs as name→value
  def to_yaml(path=None)            # path → write file; None → return string
  def from_yaml(path, group=None, auto_register=False)

  # Registration (all accept `description` kwarg)
  def register(knob)                        # accept Knob instance
  def register_int(name, description=None, **kwargs)
  def register_float(name, description=None, **kwargs)
  def register_str(name, description=None, **kwargs)
  def register_bool(name, description=None, **kwargs)
  def register_categorical(name, options, description=None, **kwargs)
```

### YAML Format

```yaml
trainer:                            # Config.name = 'trainer'
  batch_size: 32
  early_stop: true
  patience: 10
  verbose: normal

model:                              # Config.name = 'model'
  dropout: 0.5
```

### Registration Pattern (via Nomear + internal method)

```python
class TalosTrainer(Nomear):

  @Nomear.property()
  def config(self):
    cfg = Config(name='trainer')
    self._register_configs(cfg)
    return cfg

  def _register_configs(self, config):
    """Register base trainer knobs."""
    config.register_int('batch_size', default=-1)
    config.register_bool('early_stop', default=False)
    config.register_int('patience', default=10, positive=True)
    config.register_int('validate_every', default=100, positive=True)


class TorchTrainer(TalosTrainer):

  def _register_configs(self, config):
    """Register torch-specific knobs on top of base."""
    super()._register_configs(config)
    config.register_categorical('device', default='auto',
                                options=('auto', 'cpu', 'cuda'))
```

## Decided

- **Composition**: Modules have a `.config` property, not inherit from Config
- **Nomear.property()**: Config created lazily, knobs registered in `_register_configs()`
- **Subclass extension**: Override `_register_configs()`, call `super()` first
- **Storage**: Dict inside Config (simple, serializable)
- **Location**: `talos/utils/config.py` (single file for now)
- **Config.name**: Used as top-level YAML group key
- **Convenience registration**: `register_int()`, `register_float()`, etc. to avoid imports
- **`to_dict()`**: Returns OrderedDict of all knob name→value pairs
- **`to_yaml(path=None)`**: Returns string if no path, writes file if path given
- **`from_yaml(path, group=None, auto_register=False)`**: Hierarchical group access, optional auto-register
