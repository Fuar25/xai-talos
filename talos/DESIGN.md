# talos

Decoupled, modular deep learning framework. Isolates data, evaluation, model, and optimization (D-E-M-O) into independent sub-packages with backend-agnostic base classes and backend-specific implementations.

## 1. D-E-M-O Pillars

### 1.1. `data` — Universal Data Protocol

A unified protocol that organizes capabilities found across SOTA data modules in the AI ecosystem: wrapping any data source into a consistent interface, dataset manipulations (splitting, sampling, augmentation), data generation for large-scale pipelines, and distribution estimation.

### 1.2. `eval` — Model Ranking Framework

A framework for systematic model comparison and ranking across diverse tasks. At the fine-grained level, provides callable metric definitions (numpy base + backend overrides); at scale, enables multi-metric evaluation protocols for principled model selection.

### 1.3. `model` — Full-Flexibility Model Construction

All the flexibility needed to build, manage, and deploy AI models — deep neural networks with a rich architecture zoo, traditional ML models, and everything in between. Unified save/load, configuration, and automatic backend detection across frameworks.

### 1.4. `optim` — Systematic Optimization Engine

Transforms optimization from ad-hoc engineering into a reproducible, systematic process. Gradient-based training via `trainer/` delivers battle-tested workflows — validation, early stopping, checkpointing, logging — without sacrificing fine-grained control. Hyperparameter optimization via `alchemy/` automates configuration space exploration with SOTA search strategies, making well-tuned models the default rather than the exception.

### 1.5. `utils` — Shared Utilities

Cross-cutting tools used by all pillars: `check_type`, `Nomear`, `Config`, `set_seed`, `Console`.

## 2. Decided

- Four-pillar separation: data, eval, model, optim — each independently usable.
- PyTorch as first-class backend; TensorFlow planned but not yet implemented.
- `talos.work_dir` as global artifact root (models, checkpoints, logs).
- Lazy loading to keep `import talos` lightweight.

## 3. TODO

### 3.1. Root-Level DESIGN.md for optim

`talos/optim/` has no package-level df yet (only `trainer/DESIGN.md` exists). Create when alchemy sub-package matures.
