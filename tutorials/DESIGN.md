# tutorials

## 1. Purpose

Educational notebooks demonstrating Talos features and workflows. Each tutorial focuses on a specific topic (e.g., PINNs, optimization, data handling) with clear, intuitive explanations that emphasize mental models over implementation details.

## 2. Educational Philosophy

### 2.1 Core Principles

- **Mental model focused**: Explain *what* and *why*, not *how*. Implementation details belong in Talos itself or in utility modules.
- **Clear and intuitive**: Use simple language, progressive complexity, and concrete examples.
- **Show Talos advantages**: Demonstrate ease of use, intuitive APIs, and flexibility. Avoid verbose boilerplate.
- **Hide technical details**: Move plotting code, data generation helpers, and other machinery into `utils/` packages.

### 2.2 Teaching Approach

Each tutorial should:
1. Start with the problem and motivation
2. Show the Talos solution (minimal, clean code)
3. Explain key concepts with examples
4. Provide takeaways that reinforce the mental model

Avoid:
- Lengthy explanations of standard ML concepts (assume basic knowledge)
- Detailed matplotlib/plotting code in notebooks
- Repetitive boilerplate across tutorials

## 3. Structure Conventions

### 3.1 Required Sections

Every tutorial notebook must include:

1. **Title and intro** — Problem statement, motivation, key concepts
2. **Table of Contents** — Clickable navigation with HTML anchors
3. **Step 0: Setups** — Imports and environment setup
4. **Step N: ...** — Main tutorial steps (numbered sequentially)
5. **Takeaway** — Summary of key concepts and patterns
6. **References** (optional) — Citations for external sources

### 3.2 Table of Contents Format

Use explicit HTML anchors for navigation:

```markdown
---

### Table of Contents

- [Step 0: Setups](#step0)
- [Step 1: Prepare data](#step1)
- [Step 2: Build model](#step2)
- [Takeaway](#takeaway)

---
```

Each section header must include the anchor tag:

```markdown
<a id='step1'></a>
### Step 1: Prepare data
---
```

### 3.3 Code Organization

- **One code cell per logical operation** — Don't combine unrelated steps
- **Clear comments** — Use `(1)`, `(1.1)` style for hierarchical numbering
- **Minimal output** — Only show results that illustrate key concepts

## 4. Utility Organization

### 4.1 Structure

Each tutorial sub-package (e.g., `PINNs/`, `optimization/`) follows this pattern:

```
<topic>/
  01_...ipynb          → imports utils.u01
  02_...ipynb          → imports utils.u02
  utils/
    __init__.py
    common.py          ← shared utilities across all tutorials in this topic
    u01.py             ← tutorial 01 specific utilities
    u02.py             ← tutorial 02 specific utilities
```

### 4.2 Sharing Mechanism

- **`common.py`**: Utilities used by 2+ tutorials (plotting templates, trainer builders, data helpers)
- **`uXX.py`**: Tutorial-specific functions only (problem-specific data generation, custom plotting)
- **Rule**: If a utility is used in multiple tutorials, refactor it into `common.py`

### 4.3 Import Convention

Each notebook imports **only one utility module**:

```python
import utils.u01 as u  # Tutorial 01
import utils.u02 as u  # Tutorial 02
```

Within `uXX.py`, import from `common.py` as needed:

```python
# In utils/u03.py
from .common import plot_contour, get_trainer
import numpy as np

def generate_icbc_data():
  # Tutorial 03 specific data generation
  ...
```

### 4.4 Utility Design Guidelines

- **Single responsibility**: Each function does one thing well
- **Clear names**: `generate_icbc_data()`, not `make_data()`
- **Minimal dependencies**: Avoid importing heavy libraries unless necessary
- **Documented**: Include docstrings for non-obvious functions

## 5. Implementation Details

### 5.1 Formatting

- **2-space indentation** — Inherited from project convention
- **Markdown for explanations** — Use bold, italics, lists, and tables
- **LaTeX for math** — Use `$...$` for inline, `$$...$$` for display equations
- **Code syntax highlighting** — Python cells automatically highlighted

### 5.2 Cell Metadata

Use meaningful cell IDs for anchor links:
- `intro` — Introduction cell
- `step0-header`, `step0-imports` — Step 0 sections
- `step1-header`, `step1-data` — Step 1 sections
- `takeaway`, `references` — Final sections

### 5.3 Git Conventions

- Commit message prefix: `[A]` for new tutorials, `[M]` for updates, `[AM]` for both
- Commit tutorials and their utilities together
- Include a clear description of what the tutorial demonstrates

## Decided

### D1. Single Import Rule

Each notebook imports exactly one utility module (`utils.uXX`). This keeps the namespace clean and makes it immediately clear where helper functions come from.

**Rationale**: Multiple imports create confusion about where functions are defined. A single import with a short alias (`as u`) is concise and scannable.

### D2. Common Utilities Module

Shared utilities go in `common.py`, not duplicated across `uXX` files. If a function is used in 2+ tutorials, it belongs in `common.py`.

**Rationale**: Reduces duplication, makes updates easier, and naturally documents reusable patterns.

### D3. Mental Model over Implementation

Tutorials focus on high-level concepts and Talos APIs, not low-level details. Plotting, data generation helpers, and boilerplate go in utility modules.

**Rationale**: Readers should learn "how to think about PINNs with Talos", not "how to write matplotlib code". Clean notebooks are more educational and maintainable.

## TODO

### T1. Template Notebook

Create a template notebook (`tutorials/TEMPLATE.ipynb`) with:
- Standard structure (TOC, Step 0, Takeaway, References)
- Example utility imports and usage
- Placeholder sections for easy copy-paste

### T2. Automated TOC Generation

Consider a script or notebook extension to auto-generate TOCs from section headers, reducing manual maintenance.
