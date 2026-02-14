"""Typed, serializable configuration system for talos."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


# region: Knob Classes

class Knob:
  """Base class for a single typed configuration entry."""

  def __init__(self, name: str = None, default: Any = None, description: str = ''):
    self.name = name
    self.default = default
    self.description = description

  def validate(self, value):
    """Validate and return value. Override in subclasses."""
    return value

  def __repr__(self):
    return f"{self.__class__.__name__}({self.name!r}, default={self.default!r})"


class IntKnob(Knob):
  """Integer knob with optional constraints."""

  def __init__(self, name=None, default=None, description='',
               positive=False, non_negative=False):
    super().__init__(name, default, description)
    self.positive = positive
    self.non_negative = non_negative

  def validate(self, value):
    if value is None: return None
    from talos.utils import check_type
    return check_type(value, int, positive=self.positive,
                      non_negative=self.non_negative)


class FloatKnob(Knob):
  """Float knob with optional constraints."""

  def __init__(self, name=None, default=None, description='',
               positive=False, non_negative=False):
    super().__init__(name, default, description)
    self.positive = positive
    self.non_negative = non_negative

  def validate(self, value):
    if value is None: return None
    from talos.utils import check_type
    return check_type(value, float, positive=self.positive,
                      non_negative=self.non_negative)


class StrKnob(Knob):
  """String knob."""

  def validate(self, value):
    if value is None: return None
    if not isinstance(value, str):
      raise TypeError(f"Knob '{self.name}' expects str, got {type(value).__name__}")
    return value


class CategoricalKnob(Knob):
  """Categorical knob — value must be one of the allowed string options."""

  def __init__(self, name=None, options: tuple = (), default=None, description=''):
    super().__init__(name, default, description)
    self.options = tuple(options)

  def validate(self, value):
    if value is None: return None
    if value not in self.options:
      raise ValueError(
        f"Knob '{self.name}' must be one of {self.options}, got {value!r}")
    return value

  def __repr__(self):
    return (f"CategoricalKnob({self.name!r}, options={self.options}, "
            f"default={self.default!r})")


class BoolKnob(Knob):
  """Boolean knob."""

  def validate(self, value):
    if value is None: return None
    if not isinstance(value, bool):
      raise TypeError(
        f"Knob '{self.name}' expects bool, got {type(value).__name__}")
    return value

# endregion: Knob Classes


# region: Utilities

def _infer_knob(name: str, value) -> Knob:
  """Infer knob type from a Python value (used by auto_register)."""
  if isinstance(value, bool):
    return BoolKnob(name, default=value)
  if isinstance(value, int):
    return IntKnob(name, default=value)
  if isinstance(value, float):
    return FloatKnob(name, default=value)
  return StrKnob(name, default=value)

# endregion: Utilities


# region: Config Class

class Config:
  """Typed configuration container. A collection of knobs."""

  # Knob-type aliases for declarative field definitions.
  Integer = IntKnob
  Float = FloatKnob
  String = StrKnob
  Boolean = BoolKnob
  Categorical = CategoricalKnob

  # Internal attributes that bypass __setattr__ validation.
  _INTERNAL = ('_name', '_knobs', '_values')

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    # Collect Knob instances declared on this class (not parents).
    cls._fields = {}
    for name in list(vars(cls)):
      value = vars(cls)[name]
      if isinstance(value, Knob):
        cls._fields[name] = value
        value.name = name
        delattr(cls, name)

  def __init__(self, name: str = 'config'):
    object.__setattr__(self, '_name', name)
    object.__setattr__(self, '_knobs', OrderedDict())
    object.__setattr__(self, '_values', {})
    # Auto-register declared fields from MRO (base classes first).
    for klass in reversed(type(self).__mro__):
      for field_name, field in getattr(klass, '_fields', {}).items():
        self.register(field)

  # region: Attribute Access

  def __setattr__(self, name, value):
    """Type-check via registered knob on assignment."""
    knobs = object.__getattribute__(self, '_knobs')
    if name not in knobs:
      raise AttributeError(
        f"Unknown config '{name}'. Use register_*() to add new knobs. "
        f"Available: {list(knobs.keys())}")
    values = object.__getattribute__(self, '_values')
    values[name] = knobs[name].validate(value)

  def __getattr__(self, name):
    """Return knob value or default."""
    knobs = object.__getattribute__(self, '_knobs')
    values = object.__getattribute__(self, '_values')
    if name in values:
      return values[name]
    if name in knobs:
      return knobs[name].default
    raise AttributeError(f"Unknown config '{name}'")

  # endregion: Attribute Access

  # region: Registration

  def register(self, knob: Knob):
    """Register a Knob instance."""
    if not isinstance(knob, Knob):
      raise TypeError(f"Expected Knob instance, got {type(knob).__name__}")
    if knob.name is None:
      raise ValueError("Knob must have a name.")
    knobs = object.__getattribute__(self, '_knobs')
    knobs[knob.name] = knob

  def register_int(self, name, default=None, description='', **kwargs):
    """Register an integer knob."""
    self.register(IntKnob(name, default=default, description=description,
                           **kwargs))

  def register_float(self, name, default=None, description='', **kwargs):
    """Register a float knob."""
    self.register(FloatKnob(name, default=default, description=description,
                              **kwargs))

  def register_str(self, name, default=None, description=''):
    """Register a string knob."""
    self.register(StrKnob(name, default=default, description=description))

  def register_bool(self, name, default=None, description=''):
    """Register a boolean knob."""
    self.register(BoolKnob(name, default=default, description=description))

  def register_categorical(self, name, options, default=None, description=''):
    """Register a categorical knob."""
    self.register(CategoricalKnob(name, options=options, default=default,
                                   description=description))

  # endregion: Registration

  # region: Serialization

  def to_dict(self) -> OrderedDict:
    """Return all knobs as an OrderedDict of name → current value."""
    knobs = object.__getattribute__(self, '_knobs')
    values = object.__getattribute__(self, '_values')
    return OrderedDict(
      (name, values.get(name, knob.default)) for name, knob in knobs.items())

  def to_yaml(self, path=None) -> str | None:
    """Serialize config to YAML.

    Args:
      path: If provided, write to file. Otherwise return YAML string.
    """
    import yaml
    name = object.__getattribute__(self, '_name')
    data = {name: dict(self.to_dict())}
    yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False)
    if path is None:
      return yaml_str
    with open(path, 'w') as f:
      f.write(yaml_str)

  def from_yaml(self, path, group=None, auto_register=False):
    """Load config values from a YAML file.

    Args:
      path: Path to YAML file.
      group: Dot-separated group path (e.g., 'trainer', 'outer.inner').
        If None, load all top-level keys.
      auto_register: If True, auto-register unknown knobs as StrKnob.
    """
    import yaml
    with open(path, 'r') as f:
      data = yaml.safe_load(f)

    # (1) Navigate to target group.
    if group is not None:
      for part in group.split('.'):
        if not isinstance(data, dict) or part not in data:
          raise KeyError(f"Group '{group}' not found in YAML (failed at '{part}')")
        data = data[part]

    if not isinstance(data, dict):
      raise ValueError(f"Expected a mapping, got {type(data).__name__}")

    # (2) Apply values.
    knobs = object.__getattribute__(self, '_knobs')
    for name, value in data.items():
      if name not in knobs:
        if auto_register:
          self.register(_infer_knob(name, value))
        else:
          raise AttributeError(
            f"Unknown config '{name}' in YAML. "
            f"Use auto_register=True to register automatically.")
      setattr(self, name, value)

  # endregion: Serialization

  # region: Display

  def __repr__(self):
    name = object.__getattribute__(self, '_name')
    d = self.to_dict()
    items = ', '.join(f'{k}={v!r}' for k, v in d.items())
    return f"Config[{name}]({items})"

  # endregion: Display

# endregion: Config Class
