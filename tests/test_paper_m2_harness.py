from __future__ import annotations

import unittest
from types import SimpleNamespace

from experiments.r061_qwen38_m1 import _exact_config
from toolusepcg.level import Level
from toolusepcg.optimizer.greedy import GreedyOptimizer
from toolusepcg.optimizer.termination import TerminationChecker


class PaperM2HarnessTests(unittest.TestCase):
    def _prompt(self, mode: str) -> str:
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

    def test_h0_hides_metrics_but_retains_state_contract_tools_and_budget(self) -> None:
        full = self._prompt("full")
        h0 = self._prompt("no_evaluation_loop")
        self.assertIn("METRIC_SENTINEL", full)
        self.assertIn("SCORE_SENTINEL", full)
        self.assertNotIn("METRIC_SENTINEL", h0)
        self.assertNotIn("SCORE_SENTINEL", h0)
        self.assertIn("player-to-key path length of exactly 20", h0)
        self.assertIn("key-to-door path length of exactly 30", h0)
        self.assertIn("TOOL_SENTINEL", h0)
        self.assertIn("BUDGET_SENTINEL", h0)
        self.assertIn("Current Level", h0)
        self.assertNotIn("stops automatically at an exact", h0)

    def test_h0_shadow_metrics_do_not_trigger_exact_termination(self) -> None:
        config = _exact_config(
            20,
            30,
            model="test-model",
            base_url="http://127.0.0.1:1/v1",
            seed=42,
        )
        exact = SimpleNamespace(
            problem_type="zelda",
            valid=True,
            metrics={"player_key": 20, "key_door": 30},
        )
        self.assertTrue(TerminationChecker(config).should_terminate(exact))
        config.optimizer.ablation_mode = "no_evaluation_loop"
        self.assertFalse(TerminationChecker(config).should_terminate(exact))


if __name__ == "__main__":
    unittest.main()
