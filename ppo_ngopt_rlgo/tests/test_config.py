import json

import pytest

from ppo_ngopt_rlgo.config import (
    DesignVariable,
    NGOptConfig,
    PPOConfig,
    load_ngopt_config,
)
from ppo_ngopt_rlgo.args import build_parser
from ppo_ngopt_rlgo.exchange import DesignCandidate


def test_components_construct_without_each_others_dependencies():
    ppo = PPOConfig(environment_factory="package:environment")
    ngopt = NGOptConfig(
        variables=(DesignVariable("length", 0.1, 1.0),),
        objective="package:objective",
    )
    assert ppo.environment_factory == "package:environment"
    assert ngopt.variables[0].name == "length"
    record = DesignCandidate("one", {"length": 0.5}).to_dict()
    assert record["parameters"] == {"length": 0.5}


def test_ngopt_config_loads_named_variables_and_paths(tmp_path):
    path = tmp_path / "ngopt.json"
    path.write_text(
        json.dumps(
            {
                "variables": [{"name": "length", "lower": 0.1, "upper": 1.0}],
                "objective": "package:objective",
                "budget": 8,
                "result_path": "output/result.json",
            }
        ),
        encoding="utf-8",
    )
    config = load_ngopt_config(path)
    assert config.budget == 8
    assert config.variables[0].upper == 1.0
    assert str(config.result_path) == "output/result.json"


def test_invalid_ppo_batch_layout_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        PPOConfig(environment_factory="package:environment", n_steps=10, batch_size=6)


def test_duplicate_design_variable_is_rejected():
    variable = DesignVariable("length", 0.1, 1.0)
    with pytest.raises(ValueError, match="unique"):
        NGOptConfig(variables=(variable, variable), objective="package:objective")


def test_runtime_flags_are_available_on_both_components():
    parser = build_parser()
    ppo = parser.parse_args(
        ["ppo", "train", "--device", "cuda:1", "--wandb_project", "project"]
    )
    ngopt = parser.parse_args(
        ["ngopt", "optimize", "--device", "cpu", "--wandb-project", "project"]
    )
    assert (ppo.device, ppo.wandb_project) == ("cuda:1", "project")
    assert (ngopt.device, ngopt.wandb_project) == ("cpu", "project")
