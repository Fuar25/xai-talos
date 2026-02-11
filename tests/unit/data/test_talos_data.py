# SPDX-License-Identifier: MIT
"""
Unit tests for TalosData class.

Tests cover:
- Data wrapping (X, Y, metadata)
- Sampling
- Splitting (normal and stratified)
- Subset extraction
- Type checking with check_type
- Error handling
"""

import pytest
import numpy as np
from collections import Counter

from talos.data.talos_data import TalosData


# region: Fixtures

@pytest.fixture
def simple_data():
  """Create a simple TalosData instance for testing."""
  X = np.arange(100)
  Y = np.random.RandomState(42).randint(0, 3, 100)
  return TalosData.wrap(X=X, Y=Y, name='simple_data')


@pytest.fixture
def balanced_data():
  """Create TalosData with balanced classes for stratification tests."""
  X = np.arange(90)
  Y = np.array([0]*30 + [1]*30 + [2]*30)  # 30 samples per class
  return TalosData.wrap(X=X, Y=Y, name='balanced_data')


@pytest.fixture
def metadata_data():
  """Create TalosData with metadata for stratification tests."""
  X = np.arange(90)
  Y = np.array([0]*30 + [1]*30 + [2]*30)
  metadata = [{'label': i % 3, 'id': i, 'group': 'A' if i < 45 else 'B'}
              for i in range(90)]
  return TalosData.wrap(X=X, Y=Y, metadata=metadata, name='metadata_data')


# endregion: Fixtures

# region: Test Wrapping

class TestWrapping:
  """Tests for TalosData.wrap() method."""

  def test_wrap_X_only(self):
    """Test wrapping with X only."""
    X = np.arange(10)
    data = TalosData.wrap(X=X, name='test')

    assert data.X is not None
    assert data.Y is None
    assert data.metadata is None
    assert data.size == 10
    assert data.name == 'test'

  def test_wrap_X_and_Y(self):
    """Test wrapping with X and Y."""
    X = np.arange(10)
    Y = np.arange(10) * 2
    data = TalosData.wrap(X=X, Y=Y, name='test')

    assert data.X is not None
    assert data.Y is not None
    assert data.metadata is None
    assert len(data.X) == len(data.Y)

  def test_wrap_with_metadata(self):
    """Test wrapping with metadata."""
    X = np.arange(10)
    metadata = [{'id': i, 'value': i*2} for i in range(10)]
    data = TalosData.wrap(X=X, metadata=metadata, name='test')

    assert data.metadata is not None
    assert len(data.metadata) == 10
    assert data.metadata[5]['id'] == 5
    assert data.metadata[5]['value'] == 10

  def test_wrap_list_data(self):
    """Test wrapping with list instead of numpy array."""
    X = list(range(10))
    Y = list(range(10, 20))
    data = TalosData.wrap(X=X, Y=Y)

    assert isinstance(data.X, list)
    assert isinstance(data.Y, list)
    assert data.size == 10

# endregion: Test Wrapping

# region: Test Sample

class TestSample:
  """Tests for TalosData.sample() method."""

  def test_sample_basic(self, simple_data):
    """Test basic sampling."""
    sample = simple_data.sample(10)

    assert sample.size == 10
    assert sample.X is not None
    assert len(sample.X) == 10

  def test_sample_larger_than_data(self, simple_data):
    """Test sampling more than available data."""
    sample = simple_data.sample(200)

    # Should return all data (capped at total size)
    assert sample.size == simple_data.size

  def test_sample_with_Y(self, simple_data):
    """Test that sampling preserves Y."""
    sample = simple_data.sample(10)

    assert sample.Y is not None
    assert len(sample.Y) == 10

  def test_sample_negative_batch_size(self, simple_data):
    """Test that negative batch size raises error."""
    with pytest.raises(ValueError):
      simple_data.sample(-5)

  def test_sample_zero_batch_size(self, simple_data):
    """Test that zero batch size raises error."""
    with pytest.raises(ValueError):
      simple_data.sample(0)

  def test_sample_float_batch_size(self, simple_data):
    """Test that float batch size raises error."""
    with pytest.raises(TypeError):
      simple_data.sample(10.5)

# endregion: Test Sample

# region: Test Split (No Stratification)

class TestSplitBasic:
  """Tests for TalosData.split() without stratification."""

  def test_split_absolute_sizes(self, simple_data):
    """Test split with absolute sizes."""
    train, test = simple_data.split(80, 20, names=['train', 'test'])

    assert train.size == 80
    assert test.size == 20
    assert train.name == 'train'
    assert test.name == 'test'

  def test_split_ratios(self, simple_data):
    """Test split with ratios."""
    train, val, test = simple_data.split(7, 2, 1, names=['train', 'val', 'test'])

    # 100 samples split 7:2:1 = 70:20:10
    assert train.size == 70
    assert val.size == 20
    assert test.size == 10

  def test_split_float_ratios(self, simple_data):
    """Test split with float ratios."""
    train, test = simple_data.split(0.8, 0.2)

    assert train.size == 80
    assert test.size == 20

  def test_split_without_names(self, simple_data):
    """Test split without providing names."""
    train, test = simple_data.split(80, 20)

    # Should auto-generate names
    assert 'simple_data' in train.name
    assert 'simple_data' in test.name

  def test_split_preserves_Y(self, simple_data):
    """Test that split preserves Y."""
    train, test = simple_data.split(80, 20)

    assert train.Y is not None
    assert test.Y is not None
    assert len(train.Y) == 80
    assert len(test.Y) == 20

  def test_split_no_shuffle(self, simple_data):
    """Test split without shuffling."""
    train, test = simple_data.split(80, 20, shuffle=False)

    # Without shuffle, should get first 80 and last 20
    assert np.array_equal(train.X, np.arange(80))
    assert np.array_equal(test.X, np.arange(80, 100))

  def test_split_zero_sizes(self):
    """Test that all-zero sizes raise error."""
    data = TalosData.wrap(X=np.arange(10))

    with pytest.raises(ValueError, match='cannot all be zero'):
      data.split(0, 0, 0)

  def test_split_empty_sizes(self):
    """Test that empty sizes raise error."""
    data = TalosData.wrap(X=np.arange(10))

    with pytest.raises(ValueError, match='not specified'):
      data.split()

  def test_split_negative_size(self):
    """Test that negative sizes raise error."""
    data = TalosData.wrap(X=np.arange(10))

    with pytest.raises(ValueError):
      data.split(5, -5)

# endregion: Test Split (No Stratification)

# region: Test Stratified Split

class TestSplitStratified:
  """Tests for stratified splitting."""

  def test_stratify_by_Y_perfect_balance(self, balanced_data):
    """Test stratified split by Y with perfectly balanced classes."""
    train, test = balanced_data.split(0.8, 0.2, stratify=True, shuffle=False)

    # 90 samples, 3 classes (30 each)
    # 80% = 72 total, 20% = 18 total
    # Each class: 24 in train, 6 in test
    assert train.size == 72
    assert test.size == 18

    train_dist = Counter(train.Y.tolist())
    test_dist = Counter(test.Y.tolist())

    # Perfect stratification
    assert all(count == 24 for count in train_dist.values())
    assert all(count == 6 for count in test_dist.values())

  def test_stratify_by_Y_three_way_split(self, balanced_data):
    """Test three-way stratified split."""
    train, val, test = balanced_data.split(
      0.7, 0.2, 0.1, stratify=True, shuffle=False)

    # 90 samples: 63/18/9
    # Each class (30 samples): 21/6/3
    assert train.size == 63
    assert val.size == 18
    assert test.size == 9

    train_dist = Counter(train.Y.tolist())
    val_dist = Counter(val.Y.tolist())
    test_dist = Counter(test.Y.tolist())

    assert all(count == 21 for count in train_dist.values())
    assert all(count == 6 for count in val_dist.values())
    assert all(count == 3 for count in test_dist.values())

  def test_stratify_by_metadata_field(self, metadata_data):
    """Test stratified split by metadata field."""
    train, test = metadata_data.split(
      0.8, 0.2, stratify='label', shuffle=False)

    train_labels = [m['label'] for m in train.metadata]
    test_labels = [m['label'] for m in test.metadata]

    train_dist = Counter(train_labels)
    test_dist = Counter(test_labels)

    # Perfect stratification by label
    assert all(count == 24 for count in train_dist.values())
    assert all(count == 6 for count in test_dist.values())

  def test_stratify_by_different_metadata_field(self, metadata_data):
    """Test stratified split by different metadata field (group)."""
    train, test = metadata_data.split(
      0.8, 0.2, stratify='group', shuffle=False)

    train_groups = [m['group'] for m in train.metadata]
    test_groups = [m['group'] for m in test.metadata]

    train_dist = Counter(train_groups)
    test_dist = Counter(test_groups)

    # 45 samples in each group (A and B)
    # 80% = 36 each, 20% = 9 each
    assert train_dist['A'] == 36
    assert train_dist['B'] == 36
    assert test_dist['A'] == 9
    assert test_dist['B'] == 9

  def test_stratify_with_shuffle(self, balanced_data):
    """Test that stratified split with shuffle works."""
    np.random.seed(42)
    train, test = balanced_data.split(0.8, 0.2, stratify=True, shuffle=True)

    # Distribution should still be preserved
    train_dist = Counter(train.Y.tolist())
    test_dist = Counter(test.Y.tolist())

    assert all(count == 24 for count in train_dist.values())
    assert all(count == 6 for count in test_dist.values())

  def test_stratify_without_Y_raises_error(self):
    """Test that stratify=True without Y raises error."""
    data = TalosData.wrap(X=np.arange(10), name='no_Y')

    with pytest.raises(ValueError, match='Y must not be None'):
      data.split(0.8, 0.2, stratify=True)

  def test_stratify_without_metadata_raises_error(self):
    """Test that stratify with field name but no metadata raises error."""
    data = TalosData.wrap(X=np.arange(10), Y=np.arange(10))

    with pytest.raises(ValueError, match='metadata must not be None'):
      data.split(0.8, 0.2, stratify='label')

  def test_stratify_missing_field_raises_error(self, metadata_data):
    """Test that stratifying by missing field raises error."""
    with pytest.raises(ValueError, match='not found'):
      metadata_data.split(0.8, 0.2, stratify='nonexistent_field')

  def test_stratify_invalid_type_raises_error(self, balanced_data):
    """Test that invalid stratify type raises error."""
    with pytest.raises(TypeError, match='must be None, True, or a string'):
      balanced_data.split(0.8, 0.2, stratify=123)

# endregion: Test Stratified Split

# region: Test Get Subset

class TestGetSubset:
  """Tests for TalosData.get_subset() method."""

  def test_get_subset_basic(self, simple_data):
    """Test basic subset extraction."""
    indices = [0, 10, 20, 30, 40]
    subset = simple_data.get_subset(indices)

    assert subset.size == 5
    assert np.array_equal(subset.X, np.array([0, 10, 20, 30, 40]))

  def test_get_subset_with_Y(self, simple_data):
    """Test that get_subset preserves Y."""
    indices = [0, 1, 2]
    subset = simple_data.get_subset(indices)

    assert subset.Y is not None
    assert len(subset.Y) == 3

  def test_get_subset_with_metadata(self, metadata_data):
    """Test that get_subset preserves metadata."""
    indices = [0, 1, 2]
    subset = metadata_data.get_subset(indices)

    assert subset.metadata is not None
    assert len(subset.metadata) == 3
    assert subset.metadata[0]['id'] == 0
    assert subset.metadata[1]['id'] == 1

  def test_get_subset_numpy_indices(self, simple_data):
    """Test get_subset with numpy array indices."""
    indices = np.array([5, 10, 15])
    subset = simple_data.get_subset(indices)

    assert subset.size == 3
    assert np.array_equal(subset.X, np.array([5, 10, 15]))

  def test_get_subset_tuple_indices(self, simple_data):
    """Test get_subset with tuple indices."""
    indices = (1, 2, 3)
    subset = simple_data.get_subset(indices)

    assert subset.size == 3

  def test_get_subset_out_of_bounds_raises_error(self, simple_data):
    """Test that out of bounds index raises error."""
    with pytest.raises(IndexError, match='out of range'):
      simple_data.get_subset([0, 1, 200])

  def test_get_subset_negative_index_raises_error(self, simple_data):
    """Test that negative index raises error."""
    with pytest.raises(ValueError):
      simple_data.get_subset([0, -1, 2])

  def test_get_subset_none_raises_error(self, simple_data):
    """Test that None indices raise error."""
    with pytest.raises(ValueError, match='must not be None'):
      simple_data.get_subset(None)

  def test_get_subset_empty_list(self, simple_data):
    """Test get_subset with empty list."""
    subset = simple_data.get_subset([])

    assert subset.size == 0

# endregion: Test Get Subset

# region: Test Report

class TestReport:
  """Tests for TalosData.report() method."""

  def test_report_basic(self, simple_data, capsys):
    """Test basic report output."""
    simple_data.report()

    captured = capsys.readouterr()
    assert 'simple_data' in captured.out
    assert 'size: 100' in captured.out

  def test_report_with_metadata(self, metadata_data, capsys):
    """Test report with metadata."""
    metadata_data.report()

    captured = capsys.readouterr()
    assert 'metadata_data' in captured.out
    assert 'size: 90' in captured.out

# endregion: Test Report

# region: Integration Tests

class TestIntegration:
  """Integration tests combining multiple operations."""

  def test_sample_then_split(self, simple_data):
    """Test sampling followed by splitting."""
    sample = simple_data.sample(50)
    train, test = sample.split(0.8, 0.2)

    assert train.size == 40
    assert test.size == 10

  def test_split_then_sample(self, simple_data):
    """Test splitting followed by sampling."""
    train, test = simple_data.split(80, 20)
    train_sample = train.sample(10)

    assert train_sample.size == 10

  def test_multiple_splits(self, balanced_data):
    """Test multiple consecutive splits."""
    # First split
    train, rest = balanced_data.split(0.7, 0.3, stratify=True)

    # Split the rest
    val, test = rest.split(0.5, 0.5, stratify=True)

    # Check total
    assert train.size + val.size + test.size == 90

    # Check stratification maintained
    train_dist = Counter(train.Y.tolist())
    val_dist = Counter(val.Y.tolist())
    test_dist = Counter(test.Y.tolist())

    assert len(train_dist) == 3
    assert len(val_dist) == 3
    assert len(test_dist) == 3

# endregion: Integration Tests
