from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.r061_qwen38_m1 import _exact_config
from toolusepcg.level import Level
from toolusepcg.optimizer.greedy import GreedyOptimizer
from toolusepcg.optimizer.termination import TerminationChecker


class PaperHarnessCommitStopTests(unittest.TestCase):
    def _prompt(self, *, mode: str, exact_stop: bool) -> str:
        config = _exact_config(
            20,
            30,
            model="test-model",
            base_url="http://127.0.0.1:1/v1",
            seed=42,
        )
        optimizer = GreedyOptimizer.__new__(GreedyOptimizer)
        optimizer.config = config
        optimizer.ablation_mode = mode
        optimizer.evaluator_exact_stop = exact_stop
        optimizer.edit_manager = SimpleNamespace(
            level=Level.from_string(
                "################\n"
                + ("#..............#\n" * 14)
                + "################",
                problem_type="zelda",
            )
        )
        optimizer.current_result = object()
        optimizer.evaluator = SimpleNamespace(
            format_metrics_for_prompt=lambda _: "METRIC_SENTINEL"
        )
        optimizer.scorer = SimpleNamespace(
            format_score=lambda _: "SCORE_SENTINEL"
        )
        optimizer.registry = SimpleNamespace(
            format_tools_for_prompt=lambda: "TOOL_SENTINEL"
        )
        optimizer.termination = SimpleNamespace(
            format_status=lambda: "BUDGET_SENTINEL"
        )
        return optimizer._build_zelda_prompt()

    def test_same_stopping_policy_has_same_model_visible_prompt(self) -> None:
        neither = self._prompt(mode="no_evaluation_loop", exact_stop=False)
        rollback_only = self._prompt(
            mode="no_metric_observation", exact_stop=False
        )
        exact_stop_only = self._prompt(
            mode="no_metric_observation", exact_stop=True
        )
        both = self._prompt(mode="no_metric_observation", exact_stop=True)

        self.assertEqual(neither, rollback_only)
        self.assertEqual(exact_stop_only, both)
        self.assertNotEqual(neither, exact_stop_only)
        self.assertNotIn("METRIC_SENTINEL", rollback_only)
        self.assertNotIn("SCORE_SENTINEL", rollback_only)

    def test_exact_stop_switch_is_independent_of_ablation_mode(self) -> None:
        config = _exact_config(
            20,
            30,
            model="test-model",
            base_url="http://127.0.0.1:1/v1",
            seed=42,
        )
        config.optimizer.ablation_mode = "no_metric_observation"
        exact = SimpleNamespace(
            problem_type="zelda",
            valid=True,
            metrics={"player_key": 20, "key_door": 30},
        )

        config.optimizer.evaluator_exact_stop = "on"
        self.assertTrue(TerminationChecker(config).should_terminate(exact))
        config.optimizer.evaluator_exact_stop = "off"
        self.assertFalse(TerminationChecker(config).should_terminate(exact))

    def test_commit_policy_and_stop_policy_resolve_independently(self) -> None:
        config = _exact_config(
            20,
            30,
            model="test-model",
            base_url="http://127.0.0.1:1/v1",
            seed=42,
        )
        config.optimizer.ablation_mode = "no_metric_observation"
        config.optimizer.commit_policy = "force_apply"
        config.optimizer.evaluator_exact_stop = "on"
        optimizer = GreedyOptimizer(config, verbose=False)
        self.assertEqual(optimizer.commit_policy, "force_apply")
        self.assertTrue(optimizer.evaluator_exact_stop)

        config.optimizer.commit_policy = "verified"
        config.optimizer.evaluator_exact_stop = "off"
        optimizer = GreedyOptimizer(config, verbose=False)
        self.assertEqual(optimizer.commit_policy, "verified")
        self.assertFalse(optimizer.evaluator_exact_stop)



if __name__ == "__main__":
    unittest.main()
