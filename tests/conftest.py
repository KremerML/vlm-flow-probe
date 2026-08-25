"""Test configuration.

The suite is CPU-only and downloads nothing. Tests marked ``gpu`` need a CUDA
device plus model weights, and ``gate`` additionally needs the archive repo
checkout; both are deselected by default via pyproject's ``addopts``.
"""
