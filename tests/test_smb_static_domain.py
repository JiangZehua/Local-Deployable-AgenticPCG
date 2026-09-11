from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from experiments.r098_smb_static_baseline import STATIC_CASES, build_cases
from experiments.r108_smb_functional_harness import build_resolver
from toolusepcg.config import get_problem_type, load_config
from toolusepcg.evaluator import StatsEvaluator
from toolusepcg.level import Level, SMBTiles
from toolusepcg.optimizer.greedy import GreedyOptimizer
from toolusepcg.optimizer.scoring import Scorer
from toolusepcg.optimizer.termination import TerminationChecker
from toolusepcg.functional_resolver import FunctionalSpec, MetricConstraint
from toolusepcg.smb_contract import COLLECT_EVENT, SMB_COIN_EVENT_PARAM
from toolusepcg.smb_static_metrics import (
    build_static_support_scaffold,
    evaluate_static_grid,
)
from run_experiment import ExperimentLauncher


class _RaisingEnv:
    def __init__(self):
        self.called = False

    def info(self, data):
        self.called = True
        raise AssertionError("static evaluation must not call env.info")


class _PlayableEnv:
    def __init__(self):
        self.called = False
        self._problem = SimpleNamespace(
            _empty=0.5,
            _fenemies=0.1,
            _agent="astar",
            _timer=4,
            _solver=100,
        )

    def info(self, data):
        self.called = True
        coin_count = int(np.sum(data == int(SMBTiles.COIN)))
        return {
            "tube": 0,
            "empty": float(np.mean(data == int(SMBTiles.EMPTY))),
            "fenemies": 0.0,
            "complete": 1.0,
            "enemies": 0,
            "coins": coin_count,
            "jumps": 1,
            "actions": [1, 1],
            "locations": [[48.0, 224.0], [64.0, 224.0]],
            "game_events": [
                {
                    "type": COLLECT_EVENT,
                    "param": SMB_COIN_EVENT_PARAM,
                    "mario_x": 48.0 + 16.0 * index,
                    "mario_y": 208.0,
                    "time": index,
                }
                for index in range(coin_count)
            ],
        }


def _exact_grid() -> np.ndarray:
    grid = build_static_support_scaffold()
    grid[12:15, 3:5] = int(SMBTiles.TUBE)
    grid[11:15, 9:11] = int(SMBTiles.TUBE)
    grid[14, 16:19] = int(SMBTiles.GOOMBA)
    grid[13, 22:25] = int(SMBTiles.COIN)
    grid[15, 0:12] = int(SMBTiles.SOLID)
    return grid


class SMBStaticDomainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config("configs/smb_static.yaml")

    def test_registered_edit_tool_matches_smb_grammar_protocol(self) -> None:
        self.assertEqual(self.config.optimizer.edit_tools, ["place_tile"])
        optimizer = GreedyOptimizer(self.config, verbose=False)
        optimizer.initialize(seed=42)
        self.assertIn("place_tile", optimizer.registry.list_tools())

    def test_domain_identity_and_level_mappings_are_separate_but_compatible(self) -> None:
        self.assertEqual(get_problem_type("smb-static-v0"), "smb_static")
        self.assertEqual(get_problem_type("smb-v0"), "smb")
        level = Level(_exact_grid(), problem_type="smb_static")
        restored = Level.from_string(level.to_string(), problem_type="smb_static")
        np.testing.assert_array_equal(restored.data, level.data)

    def test_baseline_cases_cover_zero_and_boundary_targets(self) -> None:
        cases = build_cases()
        self.assertEqual(len(cases), len(STATIC_CASES))
        self.assertEqual(len({row["case_id"] for row in cases}), len(cases))
        targets = [row["targets"] for row in cases]
        self.assertIn(
            {
                "pipe_count": 0,
                "enemy_tile_count": 0,
                "coin_tile_count": 0,
                "block_tile_count": 0,
            },
            targets,
        )
        self.assertTrue(any(row["pipe_count"] == 4 for row in targets))
        self.assertTrue(any(row["enemy_tile_count"] == 6 for row in targets))
        self.assertTrue(any(row["coin_tile_count"] == 6 for row in targets))
        self.assertTrue(any(row["block_tile_count"] == 32 for row in targets))

    def test_exact_counts_require_and_record_gameplay_witness(self) -> None:
        env = _PlayableEnv()
        result = StatsEvaluator(self.config, env).evaluate(
            Level(_exact_grid(), problem_type="smb_static")
        )
        self.assertTrue(env.called)
        self.assertTrue(result.valid)
        self.assertEqual(result.problem_type, "smb_static")
        self.assertEqual(result.info["evaluation_mode"], "static_grid_plus_gameplay")
        self.assertTrue(result.info["simulation_called"])
        self.assertTrue(result.info["engine_executed"])
        self.assertEqual(result.metrics["complete"], 1.0)
        self.assertEqual(result.metrics["coins_collected"], 3)
        self.assertEqual(result.metrics["coin_tiles_collected"], 3)
        self.assertEqual(result.metrics["pipe_count"], 2)
        self.assertEqual(result.metrics["enemy_tile_count"], 3)
        self.assertEqual(result.metrics["coin_tile_count"], 3)
        self.assertEqual(result.metrics["block_tile_count"], 12)

    def test_malformed_tube_fails_hard_validity(self) -> None:
        grid = np.zeros((16, 32), dtype=np.int32)
        grid[5, 4:7] = int(SMBTiles.TUBE)
        metrics = evaluate_static_grid(grid)
        self.assertEqual(metrics["pipe_count"], 0)
        self.assertEqual(metrics["tube_issues"], 1)
        env = _RaisingEnv()
        result = StatsEvaluator(self.config, env).evaluate(
            Level(grid, problem_type="smb_static")
        )
        self.assertFalse(env.called)
        self.assertFalse(result.valid)
        self.assertIn("malformed tube", result.errors[0])

    def test_floating_enemy_fails_before_simulation(self) -> None:
        grid = build_static_support_scaffold()
        grid[8, 4] = int(SMBTiles.GOOMBA)
        env = _RaisingEnv()
        result = StatsEvaluator(self.config, env).evaluate(
            Level(grid, problem_type="smb_static")
        )
        self.assertFalse(env.called)
        self.assertFalse(result.valid)
        self.assertEqual(result.metrics["unsupported_enemy_count"], 1)
        self.assertIn("without solid support", result.errors[0])

    def test_exact_score_is_monotonic_and_terminates(self) -> None:
        evaluator = StatsEvaluator(self.config, _PlayableEnv())
        exact = evaluator.evaluate(Level(_exact_grid(), problem_type="smb_static"))
        off_grid = _exact_grid()
        off_grid[13, 24] = int(SMBTiles.EMPTY)
        off = evaluator.evaluate(Level(off_grid, problem_type="smb_static"))
        scorer = Scorer(self.config)
        self.assertGreater(scorer.score(exact), scorer.score(off))
        self.assertTrue(TerminationChecker(self.config).should_terminate(exact))
        self.assertFalse(TerminationChecker(self.config).should_terminate(off))

    def test_prompt_contains_static_targets_and_playability_contract(self) -> None:
        optimizer = GreedyOptimizer(self.config)
        optimizer.initialize(
            seed=42,
            initial_level=Level(build_static_support_scaffold(), "smb_static"),
        )
        prompt = optimizer.build_prompt().lower()
        for required in (
            "pipe_count", "enemy_tile_count", "coin_tile_count", "block_tile_count",
            "simulation", "complete", "support", "collected",
        ):
            self.assertIn(required, prompt)

    def test_dynamic_smb_still_uses_environment_info(self) -> None:
        dynamic = load_config("configs/smb.yaml")
        env = _RaisingEnv()
        result = StatsEvaluator(dynamic, env).evaluate(
            Level(np.zeros((16, 32), dtype=np.int32), problem_type="smb")
        )
        self.assertTrue(env.called)
        self.assertFalse(result.valid)
        self.assertEqual(result.problem_type, "smb")

    def test_experiment_aggregation_uses_static_metrics(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "trial_0"
            trial.mkdir()
            target = {
                "pipe_count": 2,
                "enemy_tile_count": 3,
                "coin_tile_count": 3,
                "block_tile_count": 12,
            }
            resolution = build_resolver().resolve(
                FunctionalSpec(
                    "smb_static",
                    {
                        name: MetricConstraint("exact", value)
                        for name, value in target.items()
                    },
                ),
                max_evaluator_calls=8,
            )
            self.assertEqual(resolution.status, "exact")
            (trial / "final_level.txt").write_text(
                resolution.level.to_string()
            )
            launcher = ExperimentLauncher(
                config_path="configs/smb_static.yaml",
                experiment_type="controllability",
                num_trials=1,
                max_workers=1,
                max_steps=1,
                base_seed=42,
                sampling_seed=42,
                run_dir=root / "experiment",
                verbose=False,
            )
            summary = launcher._aggregate_smb_static_controllability(
                [
                    {
                        "run_dir": str(trial),
                        "target_overrides": target,
                    }
                ]
            )
            self.assertEqual(summary["exact_matches"], 1)
            self.assertEqual(summary["exact_match_rate"], 1.0)
            self.assertEqual(summary["solvable_count"], 1)

    def test_r109_gate_covers_every_registered_static_case(self) -> None:
        from experiments.r109_smb_static_playability import run

        with TemporaryDirectory() as tmp:
            receipt = run(Path(tmp) / "receipt.json")
        self.assertTrue(receipt["gate_pass"])
        self.assertEqual(receipt["case_count"], 15)
        self.assertEqual(receipt["passed"], 15)


if __name__ == "__main__":
    unittest.main()
