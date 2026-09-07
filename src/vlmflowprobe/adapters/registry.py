"""Adapter registry.

Adapters are looked up by the ``model.adapter`` config key and imported
lazily so that CPU-only work (tests, distillation, analysis) never pays for
torch-heavy adapter imports.
"""

from importlib import import_module
from typing import Any, Dict

from vlmflowprobe.adapters.base import ModelAdapter

# name -> "module:ClassName"
_LAZY: Dict[str, str] = {
    "hf-llava": "vlmflowprobe.adapters.hf_llava:HFLlavaAdapter",
    "hf-gemma3": "vlmflowprobe.adapters.hf_gemma3:HFGemma3Adapter",
    "stub": "vlmflowprobe.adapters.stub:StubAdapter",
}

_REGISTRY: Dict[str, type] = {}


def register(name: str):
    """Class decorator registering an adapter under ``name``."""

    def decorator(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def adapter_class(name: str) -> type:
    if name in _REGISTRY:
        return _REGISTRY[name]
    if name in _LAZY:
        module_name, _, class_name = _LAZY[name].partition(":")
        cls = getattr(import_module(module_name), class_name)
        _REGISTRY[name] = cls
        return cls
    known = sorted(set(_REGISTRY) | set(_LAZY))
    raise KeyError(f"unknown adapter {name!r}; known adapters: {known}")


def create_adapter(model_cfg: Dict[str, Any]) -> ModelAdapter:
    """Instantiate (but do not load) the adapter named by ``model_cfg['adapter']``."""
    name = model_cfg.get("adapter")
    if not name:
        raise KeyError(
            "model config has no 'adapter' key; set model.adapter (e.g. 'hf-llava')"
        )
    return adapter_class(name)(model_cfg)
