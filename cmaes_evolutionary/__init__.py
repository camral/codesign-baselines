"""Nevergrad CMA-ES co-design baseline."""

from typing import Any


__all__ = ["CMAESConfig", "CMAESResult", "build_optimizer", "run_cmaes"]


def __getattr__(name: str) -> Any:
    """Load the public API lazily so ``python -m ...main`` stays warning-free."""

    if name in __all__:
        from . import main

        return getattr(main, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
