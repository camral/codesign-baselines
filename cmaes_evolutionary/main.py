"""Nevergrad CMA-ES baseline for general continuous co-design problems."""

from __future__ import annotations

import json
import inspect
import math
from concurrent.futures import Executor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Union

import nevergrad as ng
import numpy as np
import torch

try:
    from .args import parse_args
except ImportError:  # Support ``python main.py`` from this directory.
    from args import parse_args


Bounds = Union[float, Sequence[float], np.ndarray]
Evaluator = Callable[..., float]


def _vector(value: Bounds, dimension: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.ndim == 0:
        return np.full(dimension, float(vector))
    if vector.shape != (dimension,):
        raise ValueError(f"{name} must be a scalar or contain {dimension} values")
    return vector


@dataclass(frozen=True)
class CMAESConfig:
    """Search-space and optimizer settings for joint design-policy search."""

    design_dimension: int
    policy_dimension: int
    budget: int
    num_workers: int = 1
    evaluation_repeats: int = 1
    design_lower_bound: Bounds = -1.0
    design_upper_bound: Bounds = 1.0
    design_initial: Optional[Bounds] = None
    policy_lower_bound: Bounds = -1.0
    policy_upper_bound: Bounds = 1.0
    policy_initial: Optional[Bounds] = None
    seed: int = 0
    scale: float = 1.0
    population_size: Optional[int] = None
    binary_design: bool = False
    diagonal: bool = False
    elitist: bool = False
    maximize: bool = True

    def validated_arrays(self) -> Dict[str, np.ndarray]:
        if self.design_dimension <= 0 or self.policy_dimension <= 0:
            raise ValueError("design_dimension and policy_dimension must be positive")
        if self.budget <= 0 or self.num_workers <= 0:
            raise ValueError("budget and num_workers must be positive")
        if self.evaluation_repeats <= 0:
            raise ValueError("evaluation_repeats must be positive")
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        if self.population_size is not None and self.population_size <= 0:
            raise ValueError("population_size must be positive when provided")

        arrays = {
            "design_lower": _vector(
                self.design_lower_bound, self.design_dimension, "design_lower_bound"
            ),
            "design_upper": _vector(
                self.design_upper_bound, self.design_dimension, "design_upper_bound"
            ),
            "policy_lower": _vector(
                self.policy_lower_bound, self.policy_dimension, "policy_lower_bound"
            ),
            "policy_upper": _vector(
                self.policy_upper_bound, self.policy_dimension, "policy_upper_bound"
            ),
        }
        if np.any(arrays["design_lower"] >= arrays["design_upper"]):
            raise ValueError("every design lower bound must be below its upper bound")
        if np.any(arrays["policy_lower"] >= arrays["policy_upper"]):
            raise ValueError("every policy lower bound must be below its upper bound")
        arrays["design_initial"] = (
            (arrays["design_lower"] + arrays["design_upper"]) / 2
            if self.design_initial is None
            else _vector(self.design_initial, self.design_dimension, "design_initial")
        )
        arrays["policy_initial"] = (
            (arrays["policy_lower"] + arrays["policy_upper"]) / 2
            if self.policy_initial is None
            else _vector(self.policy_initial, self.policy_dimension, "policy_initial")
        )
        if np.any(arrays["design_initial"] < arrays["design_lower"]) or np.any(
            arrays["design_initial"] > arrays["design_upper"]
        ):
            raise ValueError("design_initial must be within the design bounds")
        if np.any(arrays["policy_initial"] < arrays["policy_lower"]) or np.any(
            arrays["policy_initial"] > arrays["policy_upper"]
        ):
            raise ValueError("policy_initial must be within the policy bounds")
        return arrays


@dataclass(frozen=True)
class CMAESResult:
    design: np.ndarray
    policy: np.ndarray
    score: float
    loss: float
    candidates_evaluated: int
    rollouts_evaluated: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "design": self.design.tolist(),
            "policy": self.policy.tolist(),
            "score": self.score,
            "loss": self.loss,
            "candidates_evaluated": self.candidates_evaluated,
            "rollouts_evaluated": self.rollouts_evaluated,
        }


def decode_design(value, binary):
    """Decode a search design into the values passed to the environment."""
    value = np.asarray(value)
    return (value >= 0).astype(np.float32) if binary else value.copy()


def build_optimizer(config: CMAESConfig) -> ng.optimization.base.Optimizer:
    """Build CMA-ES over a single genome with named design and policy blocks."""

    bounds = config.validated_arrays()
    design_range = bounds["design_upper"] - bounds["design_lower"]
    if config.binary_design:
        if not (np.all(bounds["design_lower"] == 0) and np.all(bounds["design_upper"] == 1)
                and np.all(np.isin(bounds["design_initial"], [0, 1]))):
            raise ValueError("binary designs require binary initialization and bounds [0, 1]")
        # Unbounded logits are thresholded at zero. Initialize the unit-Gaussian
        # mutation distribution around the reference's occupancy rate, so dense
        # designs do not immediately drift towards an infeasible 50% occupancy.
        from scipy.special import ndtri
        reference = bounds["design_initial"]
        probabilities = 0.9 * reference + 0.1 * reference.mean()
        logits = ndtri(np.clip(probabilities, 1e-6, 1 - 1e-6))
        design = ng.p.Array(init=logits).set_mutation(sigma=1.0)
    else:
        design = ng.p.Array(init=bounds["design_initial"])
        design.set_mutation(sigma=design_range / 6.0)
        design.set_bounds(bounds["design_lower"], bounds["design_upper"])
    policy_range = bounds["policy_upper"] - bounds["policy_lower"]
    policy = ng.p.Array(init=bounds["policy_initial"])
    policy.set_mutation(sigma=policy_range / 6.0)
    policy.set_bounds(bounds["policy_lower"], bounds["policy_upper"])
    parametrization = ng.p.Dict(design=design, policy=policy)
    parametrization.random_state.seed(config.seed)

    configured_cma = ng.optimizers.ParametrizedCMA(
        scale=config.scale,
        popsize=config.population_size,
        diagonal=config.diagonal,
        elitist=config.elitist,
    )
    return configured_cma(
        parametrization=parametrization,
        budget=config.budget,
        num_workers=config.num_workers,
    )


def _call_evaluator(
    evaluator: Evaluator,
    design: np.ndarray,
    policy: np.ndarray,
    device: str | torch.device | None,
) -> float:
    signature = inspect.signature(evaluator)
    accepts_device = "device" in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_device and device is not None:
        return float(evaluator(design, policy, device=str(device)))
    return float(evaluator(design, policy))


def _mean_score(
    evaluator: Evaluator,
    value: Dict[str, np.ndarray],
    repeats: int,
    device: str | torch.device | None = None,
) -> float:
    scores = [
        _call_evaluator(
            evaluator,
            np.asarray(value["design"]),
            np.asarray(value["policy"]),
            device,
        )
        for _ in range(repeats)
    ]
    score = float(np.mean(scores))
    if not np.isfinite(score):
        raise ValueError("the evaluator must return finite scalar scores")
    return score


def run_cmaes(
    evaluator: Evaluator,
    config: CMAESConfig,
    executor: Optional[Executor] = None,
    device: str | torch.device | None = None,
    progress_callback: Optional[Callable[[int, float, float], None]] = None,
) -> CMAESResult:
    """Jointly optimize morphology and policy parameters.

    ``budget`` counts candidate solutions. Each candidate is evaluated
    ``evaluation_repeats`` times and its mean score is sent to CMA-ES. The
    evaluator should return reward when ``maximize=True`` and cost otherwise.
    When an executor is supplied, candidate batches are evaluated concurrently.
    Process lifecycle remains under the environment integration's control.

    If given, ``progress_callback(candidates_evaluated, score, best_score)`` is
    called once per evaluated candidate so callers can stream a performance
    curve (e.g. to Weights & Biases) as the search progresses. ``best_score``
    is the best candidate score seen so far, in the ``maximize`` sense of
    ``config``.
    """

    optimizer = build_optimizer(config)
    candidates_evaluated = 0
    best_score = -math.inf if config.maximize else math.inf
    while candidates_evaluated < config.budget:
        batch_size = min(config.num_workers, config.budget - candidates_evaluated)
        candidates = [optimizer.ask() for _ in range(batch_size)]
        values = [{**candidate.value,
                   "design": decode_design(candidate.value["design"], config.binary_design)}
                  for candidate in candidates]
        if executor is None:
            scores = [
                _mean_score(evaluator, value, config.evaluation_repeats, device)
                for value in values
            ]
        else:
            futures = [
                executor.submit(
                    _mean_score,
                    evaluator,
                    value,
                    config.evaluation_repeats,
                    device,
                )
                for value in values
            ]
            scores = [future.result() for future in futures]
        for candidate, score in zip(candidates, scores):
            optimizer.tell(candidate, -score if config.maximize else score)
            candidates_evaluated += 1
            best_score = max(best_score, score) if config.maximize else min(best_score, score)
            if progress_callback is not None:
                progress_callback(candidates_evaluated, score, best_score)

    recommendation = optimizer.provide_recommendation()
    recommended = {**recommendation.value,
                   "design": decode_design(recommendation.value["design"], config.binary_design)}
    score = _mean_score(evaluator, recommended, config.evaluation_repeats, device)
    return CMAESResult(
        design=np.asarray(recommended["design"], dtype=float),
        policy=np.asarray(recommendation.value["policy"], dtype=float),
        score=score,
        loss=-score if config.maximize else score,
        candidates_evaluated=candidates_evaluated,
        rollouts_evaluated=(candidates_evaluated + 1) * config.evaluation_repeats,
    )


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device {requested!r} was requested but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


def codesign_smoke_evaluator(
    design: np.ndarray, policy: np.ndarray, device: str | torch.device = "cpu"
) -> float:
    """Synthetic reward used only to verify the general co-design interface."""

    device = torch.device(device)
    dtype = torch.float32 if device.type == "mps" else torch.float64
    design_tensor = torch.as_tensor(design, device=device, dtype=dtype)
    policy_tensor = torch.as_tensor(policy, device=device, dtype=dtype)
    design_target = torch.linspace(-0.5, 0.5, design.size, device=device, dtype=dtype)
    repeats = (policy.size + design.size - 1) // design.size
    policy_target = design_tensor.repeat(repeats)[: policy.size]
    return -float(
        ((design_tensor - design_target) ** 2).sum().item()
        + ((policy_tensor - policy_target) ** 2).sum().item()
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    device = _resolve_device(args.device)
    config = CMAESConfig(
        design_dimension=args.design_dimension,
        policy_dimension=args.policy_dimension,
        budget=args.budget,
        num_workers=args.num_workers,
        evaluation_repeats=args.evaluation_repeats,
        design_lower_bound=args.design_lower_bound,
        design_upper_bound=args.design_upper_bound,
        design_initial=args.design_initial,
        policy_lower_bound=args.policy_lower_bound,
        policy_upper_bound=args.policy_upper_bound,
        policy_initial=args.policy_initial,
        seed=args.seed,
        scale=args.scale,
        population_size=args.population_size,
        diagonal=args.diagonal,
        elitist=args.elitist,
        maximize=args.maximize,
    )
    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            config={**asdict(config), "device": str(device)},
        )

    result = run_cmaes(codesign_smoke_evaluator, config, device=device)
    payload = {"config": asdict(config), "device": str(device), **result.to_dict()}
    if wandb_run is not None:
        wandb_run.log({"score": result.score, "loss": result.loss})
        wandb_run.finish()
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        output: Path = args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
