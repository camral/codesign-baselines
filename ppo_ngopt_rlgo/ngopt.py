"""Nevergrad NGOpt runner with no Stable-Baselines3 dependency."""

from __future__ import annotations

import json
import inspect
from typing import Callable, Optional, Union

import torch

from common.nevergrad_compat import install_numpy_scalar_compat
from .config import NGOptConfig
from .exchange import DesignCandidate, EvaluationResult
from .loading import load_callable


Evaluator = Callable[[DesignCandidate], Union[EvaluationResult, float]]


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device {requested!r} was requested but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return str(device)


def _evaluate(objective, candidate, device: str, kwargs):
    signature = inspect.signature(objective)
    accepts_device = "device" in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    return objective(candidate, device=device, **kwargs) if accepts_device else objective(
        candidate, **kwargs
    )


def optimize(
    config: NGOptConfig, evaluator: Optional[Evaluator] = None
) -> EvaluationResult:
    """Minimize a general design evaluator over configured continuous variables."""
    try:
        import nevergrad as ng
    except ImportError as exc:
        raise RuntimeError(
            "NGOpt dependencies are missing; install the root requirements.txt"
        ) from exc

    objective = evaluator or load_callable(config.objective)
    device = _resolve_device(config.device)
    install_numpy_scalar_compat()
    parameters = {
        variable.name: ng.p.Scalar(lower=variable.lower, upper=variable.upper)
        for variable in config.variables
    }
    parametrization = ng.p.Instrumentation(**parameters)
    parametrization.random_state.seed(config.seed)
    optimizer = ng.optimizers.NGOpt(
        parametrization=parametrization,
        budget=config.budget,
        num_workers=config.num_workers,
    )
    evaluations = 0

    def score(**values: float) -> float:
        nonlocal evaluations
        candidate = DesignCandidate(
            candidate_id=f"ngopt-{config.seed}-{evaluations}",
            parameters={name: float(value) for name, value in values.items()},
        )
        evaluations += 1
        outcome = _evaluate(objective, candidate, device, dict(config.objective_kwargs))
        value = float(outcome.score if isinstance(outcome, EvaluationResult) else outcome)
        return -value if config.maximize else value

    recommendation = optimizer.minimize(score)
    recommended_values = {
        name: float(value) for name, value in recommendation.kwargs.items()
    }
    candidate = DesignCandidate(
        candidate_id=f"ngopt-{config.seed}-recommendation",
        parameters=recommended_values,
        metadata={"optimizer": "NGOpt", "budget": config.budget, "device": device},
    )
    outcome = _evaluate(objective, candidate, device, dict(config.objective_kwargs))
    result = outcome if isinstance(outcome, EvaluationResult) else EvaluationResult(
        candidate_id=candidate.candidate_id,
        score=float(outcome),
        metrics={"objective": float(outcome)},
    )
    config.result_path.parent.mkdir(parents=True, exist_ok=True)
    config.result_path.write_text(
        json.dumps(
            {"candidate": candidate.to_dict(), "result": result.to_dict()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result
