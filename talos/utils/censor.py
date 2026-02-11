# Copyright 2020 William Ro. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====-==============================================================-==========
"""This module provides methods related to types of method arguments.
"""
from talos.utils.format import atticus

# region: Numerical Type Definitions

# (1) Try to import numpy for extended numerical type support
try:
  import numpy as np
  _NUMPY_AVAILABLE = True
  # Integer types (Python + NumPy)
  INT_TYPES = (int, np.integer)
  # Float types (Python + NumPy)
  FLOAT_TYPES = (float, np.floating)
  # All numerical types
  NUMBER_TYPES = (int, float, np.number)
except ImportError:
  _NUMPY_AVAILABLE = False
  INT_TYPES = (int,)
  FLOAT_TYPES = (float,)
  NUMBER_TYPES = (int, float)

# endregion: Numerical Type Definitions

# region: Helper Functions

def _is_numerical(value):
  """Check if a value is numerical (int or float, including numpy types)."""
  return isinstance(value, NUMBER_TYPES)

def _validate_constraint(value, constraint_name, constraint_value,
                         check_func, error_msg):
  """Validate a numerical constraint.

  :param value: value to check
  :param constraint_name: name of the constraint (e.g., 'positive')
  :param constraint_value: whether this constraint is enabled
  :param check_func: function that returns True if constraint is satisfied
  :param error_msg: error message template
  """
  if constraint_value and value is not None:
    if not _is_numerical(value):
      raise TypeError(
        f'!! Constraint "{constraint_name}" can only be applied to numerical '
        f'types, got {type(value).__name__}')
    if not check_func(value):
      raise ValueError(error_msg.format(value, constraint_name))

# endregion: Helper Functions


def check_type(input_, type_=None, inner_type=None, nullable=False,
               auto_conversion=True, positive=False, non_negative=False,
               negative=False, non_positive=False):
  """Check the type of the given inputs. This method can also be used as a
  parser for converting strings to numbers.

  When type_ is a groups of more than 1 types, auto conversion will not be
  performed even when auto_conversion is True.

  When input_ is a large group, calling this method frequently may bring
  time overhead which is not neglectable since it involves process to build
  a list of the same scale as the input group.

  Numerical Type Support:
  - Use INT_TYPES for int-like types (int, np.int32, np.int64, etc.)
  - Use FLOAT_TYPES for float-like types (float, np.float32, np.float64, etc.)
  - Use NUMBER_TYPES for any numerical type (int or float, including numpy)

  Examples:
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  >>> check_type(31, int)
  31

  >>> check_type({'adh', 93, 19}, inner_type=str)
  {'adh', '93', '19'}

  >>> check_type([None, 12, 19.0], tuple, inner_type=int, nullable=True)
  (None, 12, 19)

  >>> check_type(3.14, NUMBER_TYPES)
  3.14

  >>> check_type(5, int, positive=True)
  5

  >>> check_type(0, int, non_negative=True)
  0

  Auto-conversion examples:
  >>> check_type(['3.0', 16], tuple, inner_type=NUMBER_TYPES)
  (3.0, 16)

  >>> check_type(128, tuple, inner_type=int, positive=True)
  (128,)

  >>> check_type(['5', '10'], list, inner_type=NUMBER_TYPES)
  [5, 10]
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  :param input_: input to be checked
  :param type_: a type or tuple/list/set of types, None by default.
                When this arg is None, inner_type must be specified and
                input_ is assumed to be in group types.
                Special: use INT_TYPES, FLOAT_TYPES, or NUMBER_TYPES for
                numerical type checking with numpy support.
  :param inner_type: if specified, arg type_ must be None or one of group_types
  :param nullable: case 1: inner_type is None: whether input_ can be None
                   case 2: inner_type is not None: whether value in input_
                           can be None
  :param auto_conversion: whether to perform automatic type conversion.
                          Works only when arg 2 is a type.
  :param positive: if True, validate that numerical values are > 0
  :param non_negative: if True, validate that numerical values are >= 0
  :param negative: if True, validate that numerical values are < 0
  :param non_positive: if True, validate that numerical values are <= 0
  :return: input_, may be converted
  """
  group_types = (tuple, list, set)

  # Sanity check
  if all([type_ is None, inner_type is None]): raise ValueError(
      '!! check_type() arg 2 and 3 can not be None at the same time')

  # Check nullable (logically this line is not necessary)
  if all([nullable, input_ is None, inner_type is None]): return None

  # (1.1) Sanity check for conflicting constraints
  if sum([positive, non_negative, negative, non_positive]) > 1:
    raise ValueError(
      '!! Only one constraint (positive/non_negative/negative/non_positive) '
      'can be specified at a time')

  # Make sure type_/inner_type is legal for isinstance() arg 2.
  # After this code block, type_/inner_type will be a type or a tuple of at
  # least 2 types
  def _check_type_type(_type, arg_ind):
    assert arg_ind in (2, 3)
    if _type is None: return None
    # Handle numerical type tuples (INT_TYPES, FLOAT_TYPES, NUMBER_TYPES)
    if isinstance(_type, tuple) and all([isinstance(t, type) for t in _type]):
      return _type
    if isinstance(_type, group_types):
      type_is_legal = all([isinstance(t, type) for t in _type])
      _type = tuple(_type) if len(_type) > 1 else _type[0]
    else: type_is_legal = isinstance(_type, type)
    if not type_is_legal: raise TypeError(
      '!! check_type() arg {} must be a type or tuple/list/set of types'.format(
        arg_ind))
    return _type

  type_ = _check_type_type(type_, 2)
  inner_type = _check_type_type(inner_type, 3)

  # Define some inner methods for checking input_
  def _safe_convert(src, tgt_type):
    """Safely convert src to tgt_type with validation.

    Special handling for NUMBER_TYPES: converts strings to float/int intelligently.
    """
    # (1.2) Special handling for numerical type tuples with string input
    if isinstance(tgt_type, tuple) and isinstance(src, str):
      # Check if this is a numerical type tuple
      is_number_tuple = any([
        tgt_type == NUMBER_TYPES,
        tgt_type == INT_TYPES,
        tgt_type == FLOAT_TYPES,
      ])
      if is_number_tuple:
        try:
          # For INT_TYPES, convert to int
          if tgt_type == INT_TYPES:
            result = int(float(src))  # float() first to handle '3.0'
            if '.' in src and float(src) != result:
              raise ValueError('!! Failed to convert {} to integer.'.format(src))
            return result
          # For FLOAT_TYPES, convert to float
          elif tgt_type == FLOAT_TYPES:
            return float(src)
          # For NUMBER_TYPES, intelligently choose int or float
          else:  # NUMBER_TYPES
            # Check if it's an integer string (no decimal point)
            if '.' not in src and 'e' not in src.lower():
              return int(src)
            else:
              return float(src)
        except ValueError as e:
          raise ValueError(
            '!! Failed to convert "{}" to {}.'.format(src, tgt_type))

    # (1.3) Regular conversion (tgt_type is a single type)
    if not isinstance(tgt_type, type):
      # Can't convert to a tuple of types, raise error
      raise TypeError('!! Cannot auto-convert to {}'.format(tgt_type))

    tgt = tgt_type(src)
    if all([isinstance(src, float), isinstance(tgt, int), src != tgt]):
      raise ValueError('!! Failed to convert {} to integer.'.format(src))
    return tgt

  def _raise_error(val, name, _type, convert=False):
    msg = '!! {0} (value: {1}) does not match{3} the required type {2}.'.format(
      name, val, _type, ' and can not be converted to' if convert else '')
    raise TypeError(msg)

  def _check_type(inp, _type, name='check_type() arg 1'):
    if not isinstance(inp, _type):
      if nullable and inp is None: return None
      # (1.5) Auto-conversion: works for single types or numerical type tuples
      if auto_conversion:
        # Allow conversion for single types or special numerical type tuples
        if isinstance(_type, type) or _type in (INT_TYPES, FLOAT_TYPES, NUMBER_TYPES):
          try: return _safe_convert(inp, _type)
          except: _raise_error(inp, name, _type, convert=True)
      _raise_error(inp, name, _type, convert=False)
    return inp

  def _apply_constraints(value, name='check_type() arg 1'):
    """Apply numerical constraints to a value."""
    if value is None and nullable: return value
    # (2.1) Positive constraint: value > 0
    _validate_constraint(
      value, 'positive', positive,
      lambda v: v > 0,
      '!! {} does not satisfy constraint "{{}}" (must be > 0)'.format(name))
    # (2.2) Non-negative constraint: value >= 0
    _validate_constraint(
      value, 'non_negative', non_negative,
      lambda v: v >= 0,
      '!! {} does not satisfy constraint "{{}}" (must be >= 0)'.format(name))
    # (2.3) Negative constraint: value < 0
    _validate_constraint(
      value, 'negative', negative,
      lambda v: v < 0,
      '!! {} does not satisfy constraint "{{}}" (must be < 0)'.format(name))
    # (2.4) Non-positive constraint: value <= 0
    _validate_constraint(
      value, 'non_positive', non_positive,
      lambda v: v <= 0,
      '!! {} does not satisfy constraint "{{}}" (must be <= 0)'.format(name))
    return value

  # Check input_
  if type_ is None: type_ = group_types

  # (1.4) Special case: if type_ is a group type and inner_type is specified,
  # and input is not already a group, convert single value to the group type
  if inner_type is not None and type_ in group_types:
    if not isinstance(input_, group_types):
      # Auto-convert single value to group type (e.g., 128 -> (128,))
      if auto_conversion:
        input_ = type_([input_])
      else:
        raise TypeError(
          f'!! check_type() arg 1 must be a group type when inner_type is '
          f'specified, got {type(input_).__name__}')

  input_ = _check_type(input_, type_)
  # Apply constraints to single values
  if inner_type is None:
    return _apply_constraints(input_)
  # Make sure input is a group
  assert inner_type is not None and isinstance(input_, group_types)
  return type(input_)(
    [_apply_constraints(
      _check_type(inp, inner_type, 'The {} element of check_type() arg 1'.format(
        atticus.ordinal(i + 1))),
      'The {} element of check_type() arg 1'.format(atticus.ordinal(i + 1)))
     for i, inp in enumerate(input_)])


if __name__ == '__main__':
  # Original tests
  assert all([
    check_type(31, int) == 31,
    check_type({'adh', 93, 19}, inner_type=str) == {'adh', '93', '19'},
    check_type([None, 12, 19.0], tuple,
               inner_type=int, nullable=True) == (None, 12, 19),
  ])

  # Test numerical type support
  assert check_type(42, INT_TYPES) == 42
  assert check_type(3.14, FLOAT_TYPES) == 3.14
  assert check_type(100, NUMBER_TYPES) == 100
  assert check_type(2.71, NUMBER_TYPES) == 2.71

  # Test with numpy types (if available)
  if _NUMPY_AVAILABLE:
    assert check_type(np.int32(10), INT_TYPES) == 10
    assert check_type(np.float64(3.14), FLOAT_TYPES) == 3.14
    assert check_type(np.int64(42), NUMBER_TYPES) == 42

  # Test positive constraint
  assert check_type(5, int, positive=True) == 5
  assert check_type(3.14, float, positive=True) == 3.14
  try:
    check_type(0, int, positive=True)
    assert False, "Should raise ValueError for 0 with positive=True"
  except ValueError:
    pass
  try:
    check_type(-5, int, positive=True)
    assert False, "Should raise ValueError for -5 with positive=True"
  except ValueError:
    pass

  # Test non_negative constraint
  assert check_type(0, int, non_negative=True) == 0
  assert check_type(10, int, non_negative=True) == 10
  try:
    check_type(-1, int, non_negative=True)
    assert False, "Should raise ValueError for -1 with non_negative=True"
  except ValueError:
    pass

  # Test negative constraint
  assert check_type(-5, int, negative=True) == -5
  try:
    check_type(0, int, negative=True)
    assert False, "Should raise ValueError for 0 with negative=True"
  except ValueError:
    pass

  # Test non_positive constraint
  assert check_type(0, int, non_positive=True) == 0
  assert check_type(-10, int, non_positive=True) == -10
  try:
    check_type(1, int, non_positive=True)
    assert False, "Should raise ValueError for 1 with non_positive=True"
  except ValueError:
    pass

  # Test constraints with inner_type
  assert check_type(
    [1, 2, 3], list, inner_type=int, positive=True) == [1, 2, 3]
  try:
    check_type([1, 0, 3], list, inner_type=int, positive=True)
    assert False, "Should raise ValueError for list with 0"
  except ValueError:
    pass

  # Test conflicting constraints
  try:
    check_type(5, int, positive=True, negative=True)
    assert False, "Should raise ValueError for conflicting constraints"
  except ValueError:
    pass

  # Test auto-conversion: string to number with NUMBER_TYPES
  print('Testing auto-conversion features...')
  result = check_type(
    ['3.0', 16], tuple, inner_type=NUMBER_TYPES, nullable=False)
  assert result == (3.0, 16), f"Expected (3.0, 16), got {result}"
  print(f'  String to number conversion: {result}')

  result = check_type(
    [None, '3.0', 16], tuple, inner_type=NUMBER_TYPES, nullable=True)
  assert result == (None, 3.0, 16), f"Expected (None, 3.0, 16), got {result}"
  print(f'  With nullable: {result}')

  # Test auto-conversion: integer string without decimal
  result = check_type(['5', '10'], list, inner_type=NUMBER_TYPES)
  assert result == [5, 10], f"Expected [5, 10], got {result}"
  print(f'  Integer strings: {result}')

  # Test auto-conversion: single value to tuple
  result = check_type(128, tuple, inner_type=int, positive=True)
  assert result == (128,), f"Expected (128,), got {result}"
  print(f'  Single value to tuple: {result}')

  # Test auto-conversion: single value to list
  result = check_type(42, list, inner_type=int)
  assert result == [42], f"Expected [42], got {result}"
  print(f'  Single value to list: {result}')

  # Test auto-conversion: value with constraints
  result = check_type(5.5, tuple, inner_type=FLOAT_TYPES, positive=True)
  assert result == (5.5,), f"Expected (5.5,), got {result}"
  print(f'  Single float to tuple with constraint: {result}')

  print('All tests passed!')
