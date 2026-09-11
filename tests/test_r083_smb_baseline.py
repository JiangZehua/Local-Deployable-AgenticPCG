from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.r083_smb_baseline import (
    DEFAULT_MANIFEST,
    EXPECTED_CASES,
    SMOKE_CASE_IDS,
    _run_smb_optimizer,
    aggregate_records,
    build_cases,
)


ROOT = Path(__file__).resolve().parent.parent


def _step(
    number: int,
    message_type: str,
    *,
    truncated: bool = False,
    reasoning: str = "",
):
    return SimpleNamespace(
        step_num=number,
        message_type=message_type,
        truncated=truncated,
        reasoning_content=reasoning,
    )


class _FakeOptimizer:
    def __init__(self, steps):
        self._steps = list(steps)
        self.step_records = []
        self.current_result = object()
        self.termination = SimpleNamespace(should_terminate=lambda result: False)
        self.config = SimpleNamespace(
            llm=SimpleNamespace(
                max_tokens=1,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            ),
            optimizer=SimpleNamespace(extra_instruction="base"),
        )

    def step(self):
        record = self._steps.pop(0)
        self.step_records.append(record)
        return record


class TestR083SmbBaseline(unittest.TestCase):
    def test_second_thinking_off_truncation_recovers_without_spending_steps(self) -> None:
        optimizer = _FakeOptimizer(
            [
                _step(1, "PARSE_ERROR", truncated=True, reasoning="draft"),
                _step(2, "STEP"),
                _step(3, "PARSE_ERROR", truncated=True),
                _step(4, "STEP"),
                _step(5, "STOP"),
            ]
        )
        recovery = _run_smb_optimizer(
            optimizer, max_steps=4, max_protocol_retries=3
        )
        self.assertEqual(recovery["semantic_steps"], 3)
        self.assertEqual(recovery["protocol_retries"], 2)
        self.assertEqual(recovery["recovered_truncations"], 2)
        self.assertEqual(recovery["unrecovered_truncations"], 0)
        self.assertEqual(recovery["trigger_steps"], [1, 3])
        self.assertFalse(recovery["budget_exhausted"])
        self.assertEqual(optimizer.config.optimizer.extra_instruction, "base")

    def test_consecutive_truncations_recover_with_one_semantic_action(self) -> None:
        optimizer = _FakeOptimizer(
            [
                _step(1, "PARSE_ERROR", truncated=True, reasoning="draft"),
                _step(2, "PARSE_ERROR", truncated=True),
                _step(3, "STEP"),
                _step(4, "STOP"),
            ]
        )
        recovery = _run_smb_optimizer(
            optimizer, max_steps=2, max_protocol_retries=2
        )
        self.assertEqual(recovery["semantic_steps"], 2)
        self.assertEqual(recovery["recovered_truncations"], 2)
        self.assertEqual(len(optimizer.step_records), 4)

    def test_retry_budget_exhaustion_is_reported_fail_closed(self) -> None:
        optimizer = _FakeOptimizer(
            [
                _step(1, "PARSE_ERROR", truncated=True, reasoning="draft"),
                _step(2, "PARSE_ERROR", truncated=True),
            ]
        )
        recovery = _run_smb_optimizer(
            optimizer, max_steps=4, max_protocol_retries=1
        )
        self.assertEqual(recovery["semantic_steps"], 0)
        self.assertEqual(recovery["recovered_truncations"], 0)
        self.assertEqual(recovery["unrecovered_truncations"], 2)
        self.assertTrue(recovery["budget_exhausted"])

    def test_manifest_is_frozen_exact_and_oracle_free(self) -> None:
        manifest, cases = build_cases(ROOT / DEFAULT_MANIFEST)
        self.assertEqual(len(cases), EXPECTED_CASES)
        self.assertFalse(manifest["provisional_only"])
        self.assertFalse(manifest["oracle_witnesses_exposed_to_model"])
        self.assertEqual(manifest["settings"]["temperature"], 0.0)
        self.assertEqual(manifest["settings"]["initialization"], "empty")
        self.assertEqual(manifest["settings"]["prompt_contract"], "exact")

    def test_smoke_covers_all_axis_values_and_is_deterministic(self) -> None:
        _, cases = build_cases(
            ROOT / DEFAULT_MANIFEST, case_ids=SMOKE_CASE_IDS
        )
        self.assertEqual([row["case_id"] for row in cases], list(SMOKE_CASE_IDS))
        self.assertEqual({row["targets"]["enemies"] for row in cases}, {0, 1, 2, 3})
        self.assertEqual({row["targets"]["coins"] for row in cases}, set(range(6)))
        self.assertEqual({row["targets"]["jumps"] for row in cases}, set(range(6)))
        self.assertEqual({row["seed"] for row in cases}, {83083})

    def test_unknown_or_duplicate_case_ids_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            build_cases(ROOT / DEFAULT_MANIFEST, case_ids=["missing"])
        with self.assertRaises(ValueError):
            build_cases(ROOT / DEFAULT_MANIFEST, case_ids=[SMOKE_CASE_IDS[0]] * 2)

    def test_aggregate_reports_joint_axis_and_worst_case_metrics(self) -> None:
        base = {
            "playable": True,
            "hard_valid": True,
            "parser_errors": 0,
            "unrecovered_parser_errors": 0,
            "transport_errors": 0,
            "tool_errors": 0,
        }
        records = [
            {
                **base,
                "exact": True,
                "e_inf": 0,
                "e_sum": 0,
                "metric_gaps": {"complete": 0, "enemies": 0, "coins": 0, "jumps": 0},
            },
            {
                **base,
                "exact": False,
                "e_inf": 2,
                "e_sum": 3,
                "metric_gaps": {"complete": 0, "enemies": 2, "coins": 0, "jumps": 1},
            },
        ]
        aggregate = aggregate_records(records)
        self.assertEqual(aggregate["exact"], 1)
        self.assertEqual(aggregate["per_metric_exact"]["coins"], 2)
        self.assertEqual(aggregate["per_metric_exact"]["enemies"], 1)
        self.assertEqual(aggregate["e_inf_mean"], 1.0)
        self.assertEqual(aggregate["e_inf_p95"], 2.0)
        self.assertEqual(aggregate["e_inf_worst"], 2.0)
        self.assertEqual(aggregate["failure_taxonomy"]["supported_target_miss"], 1)


if __name__ == "__main__":
    unittest.main()
