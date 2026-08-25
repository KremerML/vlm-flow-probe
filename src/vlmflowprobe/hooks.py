"""Utilities for registering model hooks."""

from typing import Callable, Dict

import torch


class HookManager:
    """Simple forward-hook manager with context support."""

    def __init__(self, model):
        self.model = model
        self.hooks = []

    def register_forward_hook(self, layer, hook_fn: Callable):
        handle = layer.register_forward_hook(hook_fn)
        self.hooks.append(handle)
        return handle

    def register_forward_pre_hook(self, layer, hook_fn: Callable):
        handle = layer.register_forward_pre_hook(hook_fn)
        self.hooks.append(handle)
        return handle

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove_hooks()


def create_activation_capture_hook(storage_dict: Dict, key: str):
    def hook(module, inputs, output):
        storage_dict[key] = output
    return hook
