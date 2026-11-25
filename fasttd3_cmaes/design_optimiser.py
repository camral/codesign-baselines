"""CMA-ES over the embodiment alone.

`cmaes_evolutionary.main.build_optimizer` searches a joint design-and-policy
genome. Here FastTD3 learns the controller, so CMA-ES searches only the design.
The design block uses the same parametrisation as the joint search: bounded
continuous values with sigma = range / 6, or, for binary layouts, unbounded
logits thresholded at zero by `decode_design`.
"""

import nevergrad as ng
import numpy as np

from cmaes_evolutionary.main import decode_design

__all__ = ["build_design_optimiser", "decode_design"]


def _vector(value, dimension, name):
    vector = np.asarray(value, dtype=float)
    if vector.ndim == 0:
        vector = np.full(dimension, float(vector))
    if vector.shape != (dimension,):
        raise ValueError(f"{name} must be a scalar or have shape ({dimension},)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def build_design_optimiser(lower, upper, initial, *, population_size, seed=0,
                           binary=False, diagonal=True, scale=1.0, elitist=False,
                           min_std=0.0, budget=1_000_000_000):
    """Return a Nevergrad CMA-ES whose candidate `.value` is one design array.

    Pass each candidate's value through `decode_design(value, binary)` before
    installing it in the environment.
    """
    initial = np.asarray(initial, dtype=float).reshape(-1)
    dimension = initial.shape[0]
    if dimension < 1:
        raise ValueError("the design must have at least one dimension")
    if population_size < 1 or budget < 1 or scale <= 0:
        raise ValueError("population_size, budget and scale must be positive")
    if not np.isfinite(min_std) or min_std < 0:
        raise ValueError("min_std must be finite and non-negative")
    lower = _vector(lower, dimension, "lower")
    upper = _vector(upper, dimension, "upper")
    initial = _vector(initial, dimension, "initial")
    if np.any(lower >= upper):
        raise ValueError("every design lower bound must be below its upper bound")
    if np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("initial must be within the design bounds")
    if binary:
        if not (np.all(lower == 0) and np.all(upper == 1) and np.all(np.isin(initial, [0, 1]))):
            raise ValueError("binary designs require binary initialization and bounds [0, 1]")
        # As in the joint search: centre the unit-Gaussian mutation on the
        # reference occupancy, so dense designs do not drift towards 50%.
        from scipy.special import ndtri
        probabilities = 0.9 * initial + 0.1 * initial.mean()
        design = ng.p.Array(init=ndtri(np.clip(probabilities, 1e-6, 1 - 1e-6))).set_mutation(sigma=1.0)
    else:
        design = ng.p.Array(init=initial)
        design.set_mutation(sigma=(upper - lower) / 6.0)
        design.set_bounds(lower, upper)
    design.random_state.seed(seed)
    inopts = {"minstd": float(min_std)} if min_std > 0.0 else None
    configured = ng.optimizers.ParametrizedCMA(
        scale=scale, popsize=population_size, diagonal=diagonal, elitist=elitist, inopts=inopts)
    return configured(parametrization=design, budget=budget, num_workers=population_size)
