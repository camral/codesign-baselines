"""Nevergrad remains usable with the NumPy 2 scalar-conversion rules."""

from types import SimpleNamespace

import nevergrad as ng
import numpy as np
import pytest

from common.nevergrad_compat import install_numpy_scalar_compat


def test_metamodel_ask_falls_back_instead_of_crashing():
    install_numpy_scalar_compat()
    parameter = ng.p.Array(init=np.zeros(5)).set_bounds(-np.ones(5), np.ones(5))
    optimizer = ng.optimizers.MetaModel(parametrization=parameter, budget=1000, num_workers=1)
    for _ in range(37):
        candidate = optimizer.ask()
        optimizer.tell(candidate, float(np.square(candidate.value).sum()))
    assert optimizer.num_ask == optimizer.num_tell == 37


def test_install_is_idempotent():
    from nevergrad.optimization import optimizerlib

    install_numpy_scalar_compat()
    installed = optimizerlib.learn_on_k_best
    install_numpy_scalar_compat()
    assert optimizerlib.learn_on_k_best is installed


def test_unrelated_type_error_is_not_suppressed(monkeypatch):
    from nevergrad.optimization import optimizerlib

    def broken(*args, **kwargs):
        raise TypeError("unrelated")

    monkeypatch.setattr(optimizerlib, "learn_on_k_best", broken)
    install_numpy_scalar_compat()
    with pytest.raises(TypeError, match="unrelated"):
        optimizerlib.learn_on_k_best()


def test_numpy_scalar_error_becomes_metamodel_failure(monkeypatch):
    from nevergrad.optimization import optimizerlib

    def broken(*args, **kwargs):
        raise TypeError("only 0-dimensional arrays can be converted to Python scalars")

    monkeypatch.setattr(optimizerlib, "learn_on_k_best", broken)
    install_numpy_scalar_compat()
    with pytest.raises(optimizerlib.MetaModelFailure, match="NumPy 2"):
        optimizerlib.learn_on_k_best(SimpleNamespace())
