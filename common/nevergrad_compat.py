"""Compatibility helpers for released Nevergrad versions."""

from functools import wraps


_NUMPY_SCALAR_ERROR = "only 0-dimensional arrays can be converted to Python scalars"


def install_numpy_scalar_compat() -> None:
    """Route Nevergrad's NumPy 2 metamodel failure through its CMA fallback."""
    from nevergrad.optimization import optimizerlib

    original = optimizerlib.learn_on_k_best
    if getattr(original, "_codesign_numpy_scalar_compat", False):
        return

    @wraps(original)
    def compatible_learn_on_k_best(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except TypeError as error:
            if str(error) != _NUMPY_SCALAR_ERROR:
                raise
            raise optimizerlib.MetaModelFailure(
                "Nevergrad metamodel is incompatible with NumPy 2 scalar conversion"
            ) from error

    compatible_learn_on_k_best._codesign_numpy_scalar_compat = True
    optimizerlib.learn_on_k_best = compatible_learn_on_k_best
