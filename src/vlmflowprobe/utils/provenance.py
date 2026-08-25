"""Provenance stamping: every stage records what produced its outputs.

The archive repo recorded nothing (no git SHA, no versions, no GPU); numbers
had to be trusted from directory names. Every CLI entry point here writes
``provenance.json`` next to its outputs instead.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _git(*args: str) -> Optional[str]:
    try:
        repo_dir = Path(__file__).resolve().parents[3]
        out = subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def provenance_dict(
    argv=None,
    config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "argv": list(argv if argv is not None else sys.argv),
        "python": sys.version.split()[0],
        "hostname": os.uname().nodename,
    }
    try:
        import vlmflowprobe

        info["vlmflowprobe"] = vlmflowprobe.__version__
    except Exception:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:
        pass
    if seed is not None:
        info["seed"] = seed
    if config is not None:
        info["resolved_config"] = config
    return info


def write_provenance(
    experiment_dir: str,
    config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    argv=None,
) -> str:
    os.makedirs(experiment_dir, exist_ok=True)
    path = os.path.join(experiment_dir, "provenance.json")
    with open(path, "w") as handle:
        json.dump(provenance_dict(argv=argv, config=config, seed=seed), handle, indent=1)
    return path
