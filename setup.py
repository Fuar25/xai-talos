from setuptools import setup, find_packages
from pathlib import Path

import re


ROOT = Path(__file__).resolve().parent

def _read_version():
  # Avoid importing the package here; dependencies may not be installed during setup.
  init_path = ROOT / "talos" / "__init__.py"
  with init_path.open("r", encoding="utf-8") as fh:
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', fh.read(), re.M)
    if not match:
      raise RuntimeError("Unable to find __version__ in talos/__init__.py")
    return match.group(1)

with (ROOT / "README.md").open("r", encoding="utf-8") as fh:
  long_description = fh.read()

with (ROOT / "requirements.txt").open("r", encoding="utf-8") as fh:
  requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
  name="xai-talos",
  version=_read_version(),
  author="WilliamRo",
  description="A decoupled, modular deep learning framework designed for research",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/WilliamRo/xai-talos",
  packages=find_packages(),
  classifiers=[
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
  ],
  python_requires=">=3.8",
  install_requires=requirements,
  extras_require={
    "dev": [
      "pytest>=7.0.0",
      "pytest-cov>=4.0.0",
      "black>=23.0.0",
      "flake8>=6.0.0",
    ],
  },
)
