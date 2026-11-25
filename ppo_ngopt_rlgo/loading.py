"""Helpers for resolving user-provided integration hooks."""

from importlib import import_module
from typing import Any, Callable


def load_callable(reference: str) -> Callable[..., Any]:
    module_name, separator, attribute_name = reference.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError(f"Expected 'module:callable', received {reference!r}")
    value = getattr(import_module(module_name), attribute_name)
    if not callable(value):
        raise TypeError(f"Configured object {reference!r} is not callable")
    return value
