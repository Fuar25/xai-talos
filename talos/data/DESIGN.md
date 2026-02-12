# talos/data

Flexible data container for all talos workflows. Decouples data manipulation from model and optimization concerns.

## 1. TalosData

A `Nomear`-based container holding `X`, `Y` (optional), and `metadata` (optional, list of dicts per datum).

**`X` flexibility**: `X` can be `np.ndarray` (model-ready tabular/image data) or `list` (raw objects/callables for lazy loading). Generic APIs (split, sample, get_subset) work on all forms, so data processing pipelines can use TalosData at every stage — from raw dataset through intermediate transforms to final model-ready arrays.

### 1.1. API

- `wrap(cls, X, Y=None, name=None, metadata=None, **kwargs)` — Factory classmethod. Wraps raw arrays into a TalosData instance.
- `sample(batch_size)` — Random subset without replacement. `None`/`-1` returns self.
- `split(*sizes, names=None, shuffle=True, stratify=None)` — Split into multiple subsets.
  - `sizes`: absolute counts or ratios (auto-normalized if sum ≠ total).
  - `stratify=True`: stratify by Y. `stratify='field'`: stratify by metadata field.
- `get_subset(indices, name=None)` — Create subset by index selection. Handles ndarray, list, tuple.
- `report()` — Print dataset summary (name, size, X/Y type and shape).
- `size` — Property, returns `len(X)`.

### 1.2. Metadata

Optional `list[dict]` aligned with `X`. Fields can be incomplete across entries. Used for stratified splitting (`stratify='field_name'`).

## 2. TODO

### 2.1. Data Processing Pipeline

Design a layered processing system where raw datasets are transformed stage by stage into model-ready data (X as `np.ndarray`). Each stage produces a TalosData, enabling split/sample at any point.

Envisioned API:
```python
raw = TalosData.wrap(file_paths, name='raw')          # X = list of paths
processed = pipeline.transform(raw)                     # X = np.ndarray, model-ready
train, val = processed.split(0.8, 0.2, stratify=True)
```

### 2.2. Metadata ↔ Pandas/CSV Interop

`metadata` (`list[dict]`) maps naturally to pandas DataFrame (rows = data points, columns = fields) and therefore to CSV. This enables:
- Inspecting/filtering metadata with pandas tooling before splitting or training
- Loading metadata from existing CSV/spreadsheet datasets
- Exporting metadata for external analysis or logging

Envisioned API:
```python
# Export
df = data.metadata_to_dataframe()         # list[dict] → pd.DataFrame
data.metadata_to_csv('meta.csv')          # shorthand for df.to_csv()

# Import
data.metadata_from_dataframe(df)          # pd.DataFrame → list[dict]
data.metadata_from_csv('meta.csv')        # shorthand for pd.read_csv() → list[dict]
```
