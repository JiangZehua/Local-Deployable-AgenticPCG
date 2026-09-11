from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments import high_level_freeform_eval as freeform
from experiments.paper_m4_freeform_routing import (
    _route_status,
    load_manifest as load_paper_m4_manifest,
)
from experiments.zelda_freeform_ablation_eval import paired_against_full
from experiments.zelda_pcg_tool_usage_eval import _sequence_status, load_manifest
from toolusepcg.config import load_config
from toolusepcg.level import Level
from toolusepcg.optimizer.greedy import GreedyOptimizer


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "configs/high_level_freeform_zelda_t1_t4_32x32.yaml"
MANIFEST = ROOT / "configs/zelda_pcg_tool_routing_32x32.yaml"
PAPER_M4_MANIFEST = ROOT / "configs/paper_m4_freeform_routing18.yaml"


class ZeldaFreeformExpandedEvalTests(unittest.TestCase):
    def test_expanded_suite_has_three_prompts_per_tier(self) -> None:
        suite = freeform.load_suite(SUITE)
        self.assertEqual(len(suite["cases"]), 12)
        self.assertEqual(
            {tier: sum(case["tier"] == tier for case in suite["cases"]) for tier in range(1, 5)},
            {1: 3, 2: 3, 3: 3, 4: 3},
        )
        self.assertTrue(all(case.get("evaluator") for case in suite["cases"]))

    def _prompt(self, mode: str) -> str:
        config = load_config(ROOT / "configs/zelda_freeform_32x32_full.yaml")
        optimizer = GreedyOptimizer.__new__(GreedyOptimizer)
        optimizer.config = config
        optimizer.ablation_mode = mode
        level = Level.from_string(
            (ROOT / "configs/high_level_freeform_zelda_initial_32x32.txt").read_text(),
            problem_type="zelda",
        )
        optimizer.edit_manager = SimpleNamespace(level=level)
        optimizer.current_result = object()
        optimizer.evaluator = SimpleNamespace(
            format_metrics_for_prompt=lambda _: "METRIC_SENTINEL"
        )
        optimizer.scorer = SimpleNamespace(format_score=lambda _: "SCORE_SENTINEL")
        optimizer.registry = SimpleNamespace(format_tools_for_prompt=lambda: "TOOL_SENTINEL")
        optimizer.termination = SimpleNamespace(format_status=lambda: "BUDGET_SENTINEL")
        return optimizer._build_zelda_prompt()

    def test_no_metric_modes_hide_only_metric_and_score_sections(self) -> None:
        full = self._prompt("full")
        hidden = self._prompt("no_metric_observation")
        no_eval = self._prompt("no_evaluation_loop")
        self.assertIn("METRIC_SENTINEL", full)
        self.assertIn("SCORE_SENTINEL", full)
        for prompt in (hidden, no_eval):
            self.assertNotIn("METRIC_SENTINEL", prompt)
            self.assertNotIn("SCORE_SENTINEL", prompt)
            self.assertIn("TOOL_SENTINEL", prompt)
            self.assertIn("BUDGET_SENTINEL", prompt)
            self.assertIn("Current Level", prompt)

    def test_official_ablation_configs_are_a_cumulative_ladder(self) -> None:
        expected = {
            "zelda_freeform_32x32_full.yaml": ("full", "default"),
            "zelda_freeform_32x32_no_history.yaml": ("no_historical_feedback", "default"),
            "zelda_freeform_32x32_no_metrics.yaml": ("no_metric_observation", "default"),
            "zelda_freeform_32x32_no_eval.yaml": ("no_evaluation_loop", "default"),
            "zelda_freeform_32x32_always_accept.yaml": ("full", "always_accept"),
        }
        for name, (mode, acceptance) in expected.items():
            config = load_config(ROOT / "configs" / name)
            self.assertEqual(config.optimizer.ablation_mode, mode)
            self.assertEqual(config.optimizer.acceptance_policy, acceptance)
            self.assertEqual(config.optimizer.edit_tools, ["place_tile"])
            self.assertEqual(config.optimizer.conversation_length, 2)
            self.assertTrue(config.optimizer.stable_change_accounting)

    def test_paired_ablation_counts_losses_and_gains(self) -> None:
        def row(case: str, score: float, strict: bool, functional: bool) -> dict:
            return {
                "case_id": case,
                "seed": 42,
                "tier": 1,
                "programmatic_score": score,
                "programmatic_strict_success": strict,
                "functional": {"success": functional},
            }

        result = paired_against_full(
            [row("a", 1.0, True, True), row("b", 0.0, False, False)],
            [row("a", 0.5, False, True), row("b", 1.0, True, True)],
        )
        self.assertEqual(result["strict_losses"], 1)
        self.assertEqual(result["strict_gains"], 1)
        self.assertEqual(result["functional_gains"], 1)
        self.assertEqual(result["mean_score_delta"], 0.25)


    def test_expected_sequence_requires_order_but_coverage_does_not(self) -> None:
        result = _sequence_status(["a", "b", "c"], ["a", "x", "c", "b"])
        self.assertTrue(result["all_expected_tools_used"])
        self.assertFalse(result["expected_sequence_observed"])
        self.assertEqual(result["expected_tool_coverage"], 1.0)

    def test_initial_maps_are_exactly_32_square(self) -> None:
        for name in (
            "high_level_freeform_zelda_initial_32x32.txt",
            "zelda_pcg_disconnected_initial_32x32.txt",
        ):
            rows = (ROOT / "configs" / name).read_text().splitlines()
            self.assertEqual(len(rows), 32)
            self.assertTrue(all(len(row) == 32 for row in rows))


    def test_paper_m4_route_separates_called_successful_and_accepted(self) -> None:
        calls = [
            {
                "tool_name": "generate_zelda_random",
                "tool_success": True,
                "accepted_successful_call": False,
            },
            {
                "tool_name": "generate_zelda_ca",
                "tool_success": True,
                "accepted_successful_call": True,
            },
        ]
        route = _route_status(
            ["generate_zelda_random", "generate_zelda_ca"], calls
        )
        self.assertTrue(route["called_sequence_observed"])
        self.assertTrue(route["successful_sequence_observed"])
        self.assertFalse(route["accepted_sequence_observed"])
        self.assertEqual(route["called_pcg_count"], 2)
        self.assertEqual(route["accepted_pcg_count"], 1)


if __name__ == "__main__":
    unittest.main()
