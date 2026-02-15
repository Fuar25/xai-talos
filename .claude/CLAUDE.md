# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 1. Project Overview

**xai-talos** is a decoupled, modular deep learning framework that isolates three core concerns:
- Data (data manipulation and loading)
- Evaluation (metrics)
- Model (neural network architecture, algorithm workflow, and model deployment)
- Optimization (gradient-based optimization and hyperparameter tuning)

Built for reproducibility and clean abstractions. Currently v0.1.0 (alpha), PyTorch-focused. Requires Python >=3.11.

## 2. Design File Convention

Each package and module can have an associated `*DESIGN.md` file:
- **Package**: `DESIGN.md` at the package root, sibling to `__init__.py` (e.g., `talos/optim/DESIGN.md`)
- **Module**: `MODULE_NAME_DESIGN.md` sibling to the module (e.g., `talos/optim/trainer/TALOS_TRAINER_DESIGN.md` for `talos_trainer.py`)

**Before planning or coding**, read design files in hierarchical order from root to target. For example, before working on `talos/optim/trainer/talos_trainer.py`, read (if they exist):
1. `talos/DESIGN.md`
2. `talos/optim/DESIGN.md`
3. `talos/optim/trainer/DESIGN.md`
4. `talos/optim/trainer/TALOS_TRAINER_DESIGN.md`

**After updating a module or package**, update its corresponding design file(s) if necessary. Each design file should be as compact as possible — accurate with theoretically minimal text. Upper-level design files describe *what* sub-packages can do, not *how* they do it. This means changes to a sub-package rarely require updating ancestor design files.

**Detail level by depth**: Leaf-level design files (module) should include necessary usage examples and rules to make the API clear. Parent-level design files (package) may include usage examples, but only the most important ones.

**Title**: Use the package/module path as the H1 title (e.g., `# talos/optim/trainer`).

**Numbering**: Use numbered sections (1., 1.1, etc.) for all headings in design files.

**Standard sections** in a design file:
- Main body — current design: purpose, API, usage, decisions
- **Decided** — key architectural decisions and resolution rules, placed right before TODO
- **TODO** — planned features with numbered subsections. Code examples in TODO must be labeled "Envisioned API:" to distinguish from decided/implemented features.

## 3. Code Conventions

- **2-space indentation** for all Python files — this is non-negotiable and applies everywhere (`talos/`, `setup.py`, scripts, tests, experiments)
- Add necessary comments for code blocks in a compact and precise way. Use '(1)'/'(1.2)' style for hierarchical numbering.
- Commit messages use prefix convention: `[A]` added, `[M]` modified, `[AM]` added & modified
- **Type checking**: Use `talos.utils.check_type()` for runtime type validation instead of manual `isinstance()` checks or assertions. Benefits include auto-conversion, numerical type support (int/float/numpy), and value constraints (positive, non_negative, etc.)
- **Design docs**: Keep `.md` design/discussion files lean and decision-focused. Only document what's decided or needs immediate discussion. Avoid speculative content (future examples, rationale for unimplemented features, roadmaps for phases not yet started). Add detail incrementally as implementation progresses.

## 4. Abbreviations

- **df** — design file (e.g., "the df of talos/optim" → `talos/optim/DESIGN.md`)
- **ccmd** — `.claude/CLAUDE.md`
- **s`<N>`** — section or subsection in a `.md` file (e.g., s4.1 → section 4.1). Combine with file abbreviations: `ccmd/s4` → section 4 of CLAUDE.md.
- **tt** — `tutorials/`
- **pint** — `tutorials/PINNs`

## 5. Commands

Commands must be typed exactly. On typos (e.g., "syyc df"), do nothing and suggest "Did you mean: `sync df ...`?". Always discuss discrepancies before editing.

- **`sync df <pkg>`** — Review the df of `<pkg>` against its implementation. Read the dfs of direct sub-packages (not code files) to check consistency. Create the df if it doesn't exist.
- **`sync df <pkg> all`** — Recursive sync from leaf sub-packages upward to `<pkg>`. Only auto-create package-level dfs, never module-level ones.
- **`sync df <module>`** — Review the module-level df against the corresponding module code. Create the df if it doesn't exist.
- **`align df <pkg/module>`** — Check that the df format conforms to ccmd/s2 conventions (title, numbering, standard sections). Discuss issues before editing.
- **`align df *`** — Check all existing df files under `talos/` for convention conformance.

For architecture and module details, read `talos/DESIGN.md` and its sub-package design files per the Design File Convention above.
