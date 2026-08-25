"""Process-level environment defaults for CLI entry points.

Import this module BEFORE torch (every CLI entry does it first thing):

* ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` — several fp32 SAEs
  coexist beside a 7B fp16 model in 24 GB; without expandable segments the
  allocator fragments and multi-layer runs OOM.
* ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` — required for
  ``torch.use_deterministic_algorithms`` to actually constrain cuBLAS.

``setdefault`` only: anything the user exported wins.
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
