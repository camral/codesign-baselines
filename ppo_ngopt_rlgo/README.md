# PPO + NGOpt RL/GO baseline

This directory provides two independently runnable components for a general
co-design problem:

- `ppo.py` trains and evaluates a controller with Stable-Baselines3 PPO.
- `ngopt.py` searches named, bounded design variables with Nevergrad NGOpt.

They deliberately do not import each other. The serializable records in
`exchange.py` (`DesignCandidate`, `PolicyArtifact`, and `EvaluationResult`)
form the boundary a future coordinator can use to specify synchronization,
policy reuse, and design-policy evaluation.

## Installation

Install the shared Python 3.12 environment from the repository root. It
contains PPO, NGOpt, and the test dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run each component

Run from the repository root. The checked-in configurations are smoke-test
examples, not assumptions about a particular robot or task.

```bash
python -m ppo_ngopt_rlgo ppo train --device cuda:0 --wandb-project my-project
python -m ppo_ngopt_rlgo ppo evaluate --device cuda:0 --wandb-project my-project
python -m ppo_ngopt_rlgo ngopt optimize --device cuda:0 --wandb-project my-project
```

Each command accepts `--config PATH` and `--seed`. PPO also accepts
`--model-path`; NGOpt accepts `--budget`. PPO writes its SB3 checkpoint under
`artifacts/ppo/`, while NGOpt owns `artifacts/ngopt/`.
`--device` selects `auto`, `cpu`, `cuda`, `cuda:N`, or `mps`; NGOpt forwards
the resolved value to objectives that accept a `device` keyword. Supplying
`--wandb-project` enables result logging.

## Configure a co-design problem

`configs/ppo.json` identifies a Gymnasium-compatible environment factory as
`module:callable`, plus arbitrary factory arguments and standard PPO
hyperparameters. The factory must return an environment accepted by SB3. A
coordinator may instead construct a design-specific environment and pass it
directly to `ppo.train()` or `ppo.evaluate()`.

`configs/ngopt.json` declares named continuous design variables, their physical
bounds, an objective hook in `module:callable` form, its keyword arguments, and
the search budget. The objective receives a `DesignCandidate` and returns
either a scalar score or an `EvaluationResult`. Set `maximize` to `true` for
controller reward and leave it `false` for a cost. `budget` counts objective
calls made by NGOpt; the recommended candidate is evaluated once more for the
persisted result.

A future co-design loop can therefore:

1. Receive a `DesignCandidate` from NGOpt and build the matching simulator.
2. Train or load PPO for that simulator, producing a `PolicyArtifact`.
3. Evaluate the design-policy pair and return an `EvaluationResult` to NGOpt.
4. Decide externally when policies and designs synchronize, how checkpoints
   are reused, and whether candidate evaluations run in parallel.

The included CartPole factory and quadratic cost only verify the independent
installations. Replace both hooks for the actual co-design task.
