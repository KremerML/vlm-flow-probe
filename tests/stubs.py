"""Shared CPU test doubles.

Import stubs from here, not from test modules. For now this re-exports the
historical stub classes defined in ``tests.test_ablation``; Phase 2 of the port
moves the definitions here and builds the ``StubAdapter`` on top of them.
"""

from tests.test_ablation import CountingModel, DatasetStub  # noqa: F401

__all__ = ["CountingModel", "DatasetStub"]
