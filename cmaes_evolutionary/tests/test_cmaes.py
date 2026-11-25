"""Unit and smoke tests for the general co-design CMA-ES interface."""

import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from cmaes_evolutionary.args import parse_args
from cmaes_evolutionary.main import CMAESConfig, run_cmaes


class CMAESTest(unittest.TestCase):
    def test_default_config_accepts_command_line_overrides(self) -> None:
        args = parse_args(
            [
                "--budget",
                "12",
                "--no-maximize",
                "--device",
                "cpu",
                "--wandb_project",
                "test-project",
            ]
        )
        self.assertEqual(args.budget, 12)
        self.assertFalse(args.maximize)
        self.assertEqual(args.design_dimension, 4)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.wandb_project, "test-project")

    def test_general_codesign_evaluator(self) -> None:
        config = CMAESConfig(
            design_dimension=2,
            policy_dimension=3,
            budget=12,
            evaluation_repeats=2,
            design_lower_bound=[-2.0, -1.0],
            design_upper_bound=[2.0, 1.0],
            policy_lower_bound=-0.5,
            policy_upper_bound=0.5,
            seed=3,
            num_workers=2,
            maximize=False,
        )

        def cost(design: np.ndarray, policy: np.ndarray) -> float:
            return float(np.sum(design**2) + np.sum(policy**2))

        with ThreadPoolExecutor(max_workers=2) as executor:
            result = run_cmaes(cost, config, executor=executor)
        self.assertEqual(result.design.shape, (2,))
        self.assertEqual(result.policy.shape, (3,))
        self.assertEqual(result.candidates_evaluated, 12)
        self.assertEqual(result.rollouts_evaluated, 26)
        self.assertGreaterEqual(result.loss, 0.0)
        np.testing.assert_array_less(result.design, [2.0, 1.0])
        np.testing.assert_array_less([-2.0, -1.0], result.design)

    def test_rejects_mismatched_per_parameter_bounds(self) -> None:
        config = CMAESConfig(
            design_dimension=2,
            policy_dimension=1,
            budget=1,
            design_lower_bound=[-1.0],
        )
        with self.assertRaisesRegex(ValueError, "design_lower_bound"):
            config.validated_arrays()


if __name__ == "__main__":
    unittest.main()
