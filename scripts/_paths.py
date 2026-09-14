"""Where a model tag's run artifacts live.

Two layouts are in play and both are correct. Committed artifacts are grouped
per model -- ``output/experiments/<tag>/<tag>_<run>/`` -- because the repo holds
two replications' results side by side. A live run tree stays flat:
``output_base`` is still ``output/experiments``, and ``scripts/snellius_sync.sh``
excludes ``output/`` entirely, so the cluster never sees the grouping.

Every analysis script globs one directory level from ``--root``, so a script
pointed at the wrong one of those two matches nothing and reports "no results"
when it should report "wrong layout". ``resolve_root`` picks the grouping when
it is there, and callers print what it chose.

Stdlib only, and imported as a sibling (the scripts run as
``python scripts/<name>.py``, which puts ``scripts/`` on ``sys.path``), so the
login-node scripts still work without the model venv.
"""

import os


def resolve_root(root, tag):
    """The experiment root holding ``tag``'s runs: grouped if present, else flat."""
    grouped = os.path.join(root, tag)
    return grouped if os.path.isdir(grouped) else root
