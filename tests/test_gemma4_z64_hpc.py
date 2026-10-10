from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.gemma4_z64_hpc import (
    exact_mcnemar,
    full_trace,
    placeholder_episode,
    planned_cells,
    read_ids,
    summary_for,
    validate_dev_run,
)


class Gemma4Z64ContractTests(unittest.TestCase):
    def test_development_ids_are_disjoint_from_formal_z64(self) -> None:
        formal = set(read_ids(Path("configs/paper_m2_z64_target_ids.txt")))
        dev = read_ids(Path("configs/gemma4_z64_dev_ids.txt"))
        self.assertEqual(len(dev), 8)
        self.assertFalse(formal.intersection(dev))
        self.assertEqual(len(planned_cells("dev", dev)), 8)
        self.assertEqual(
            len(planned_cells("formal", sorted(formal))),
            128,
        )

    def test_exact_mcnemar_handles_discordant_and_tied_pairs(self) -> None:
        self.assertIsNone(exact_mcnemar(0, 0))
        self.assertEqual(exact_mcnemar(1, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar(4, 0), 0.125)
        self.assertEqual(exact_mcnemar(2, 2), 1.0)

    def test_summary_uses_full_denominator_and_paired_difference(self) -> None:
        rows = []
        targets = [f"target-{index}" for index in range(4)]
        outcomes = {
            "target-0": (False, True),
            "target-1": (True, False),
            "target-2": (True, True),
            "target-3": (False, False),
        }
        for target in targets:
            for arm, exact in zip(("open_loop", "verified"), outcomes[target]):
                row = placeholder_episode("run", arm, target, status="completed")
                row["exact"] = exact
                row["hard_valid"] = True
                row["absolute_error"] = 0 if exact else 1
                rows.append(row)
        protocol = {
            "analysis": {
                "bootstrap_resamples": 2000,
                "bootstrap_seed": 20260831,
            }
        }
        summary, paired = summary_for(rows, protocol)
        self.assertFalse(summary["temporary"])
        self.assertEqual(len(paired), 4)
        self.assertEqual(summary["paired_exact"]["verified_only"], 1)
        self.assertEqual(summary["paired_exact"]["open_loop_only"], 1)
        self.assertEqual(summary["paired_exact"]["common_success"], 1)
        self.assertEqual(summary["paired_exact"]["common_failure"], 1)
        self.assertEqual(summary["paired_exact"]["difference_verified_minus_open"], 0.0)
        self.assertEqual(summary["arms"]["open_loop"]["exact_rate_full_denominator"], 0.5)

    def test_full_trace_preserves_request_response_tools_maps_and_decision(self) -> None:
        step = SimpleNamespace(
            step_num=1,
            request_messages=[{"role": "user", "content": "prompt"}],
            request_params={
                "model": "gemma4-31b-pcg",
                "temperature": 0.0,
                "max_tokens": 6144,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            },
            raw_response='{"type":"STEP"}',
            reasoning_content="draft",
            truncated=False,
            input_tokens=12,
            output_tokens=5,
            message_type="STEP",
            errors=[],
            parsed_message={"type": "STEP"},
            tool_results=[{"tool_name": "place_tile", "result": {"success": True}}],
            level_before="before",
            level_after="after",
            metrics_before={"player_key": 1},
            metrics_after={"player_key": np.int64(2)},
            score_before=0.1,
            score_after=0.2,
            accepted=False,
            accept_reason="rejected",
            acceptance_triage={},
            elapsed_time=1.5,
            model_seconds=1.0,
            tool_seconds=0.2,
        )
        optimizer = SimpleNamespace(
            step_records=[step],
            current_level=SimpleNamespace(to_string=lambda: "before"),
        )
        trace = full_trace(
            optimizer,
            {"target_id": "pk10-kd14", "player_key": 10, "key_door": 14},
        )
        event = trace["steps"][0]
        self.assertEqual(event["request"]["messages"][0]["content"], "prompt")
        self.assertEqual(event["response"]["reasoning_content"], "draft")
        self.assertEqual(event["tool_execution"][0]["tool_name"], "place_tile")
        self.assertEqual(event["maps"]["candidate_after"], "after")
        self.assertEqual(event["maps"]["saved_after"], "before")
        self.assertEqual(event["decision"]["outcome"], "reject")
        self.assertEqual(event["metrics"]["candidate_after"]["player_key"], 2)
        json.dumps(trace)

    def test_placeholder_has_required_handoff_fields(self) -> None:
        row = placeholder_episode(
            "run",
            "open_loop",
            "pk10-kd14",
            status="not_started",
        )
        required = {
            "run_id",
            "arm",
            "target_id",
            "seed",
            "attempt_id",
            "status",
            "exact",
            "hard_valid",
            "target_player_key",
            "target_key_door",
            "final_player_key",
            "final_key_door",
            "absolute_error",
            "failure_category",
            "steps",
            "calls",
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "handoff_calls",
            "wall_seconds",
            "model_seconds",
            "tool_seconds",
            "started_at",
            "ended_at",
            "trace_path",
            "final_level_path",
        }
        self.assertTrue(required.issubset(row))
        self.assertIsNone(row["exact"])
        self.assertEqual(row["status"], "not_started")

    def test_protocol_json_is_valid_and_has_two_arms(self) -> None:
        protocol = json.loads(Path("configs/gemma4_z64_protocol.json").read_text())
        self.assertEqual(protocol["schema_version"], "gemma4-z64-protocol-v2")
        self.assertEqual(set(protocol["arms"]), {"open_loop", "verified"})
        self.assertEqual(protocol["shared"]["formal_target_count_per_arm"], 64)
        self.assertEqual(protocol["analysis"]["bootstrap_resamples"], 200000)
        self.assertEqual(protocol["shared"]["runtime_dependencies"]["stdlib_imports"], [])
        self.assertEqual(len(protocol["shared"]["runtime_dependencies"]["pythonpath_submodules"]), 2)
        self.assertIn("nvidia-smi", protocol["shared"]["runtime_dependencies"]["host_cli_tools"])
        self.assertIn("git", protocol["shared"]["runtime_dependencies"]["host_cli_tools"])


    def test_dev_gate_rejects_incomplete_evidence(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "interface_probe.json").write_text(
                json.dumps({"gate_pass": True}), encoding="utf-8"
            )
            (root / "analysis_validation.json").write_text(
                json.dumps({"episodes_rows": 8, "expected_rows": 8, "duplicates": 0}),
                encoding="utf-8",
            )
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "expected_episodes": 8,
                        "arms": {
                            arm: {
                                "expected": 4,
                                "completed": 0,
                                "task_failed": 0,
                                "timeout": 0,
                                "not_started": 4,
                                "infrastructure_interrupted": 0,
                            }
                            for arm in ("open_loop", "verified")
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "Development gate failed"):
                validate_dev_run(root)

    def test_dev_gate_accepts_complete_summary_contract(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "interface_probe.json").write_text(
                json.dumps({"gate_pass": True}), encoding="utf-8"
            )
            (root / "analysis_validation.json").write_text(
                json.dumps({"episodes_rows": 8, "expected_rows": 8, "duplicates": 0}),
                encoding="utf-8",
            )
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "expected_episodes": 8,
                        "arms": {
                            arm: {
                                "expected": 4,
                                "completed": 4,
                                "task_failed": 0,
                                "timeout": 0,
                                "not_started": 0,
                                "infrastructure_interrupted": 0,
                            }
                            for arm in ("open_loop", "verified")
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = validate_dev_run(root)
            self.assertTrue(result["gate_pass"])
            self.assertEqual(result["errors"], [])

    def test_dev_gate_rejects_missing_or_wrong_expected(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "interface_probe.json").write_text(
                json.dumps({"gate_pass": True}), encoding="utf-8"
            )
            (root / "analysis_validation.json").write_text(
                json.dumps({"episodes_rows": 8, "expected_rows": 8, "duplicates": 0}),
                encoding="utf-8",
            )
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "expected_episodes": 8,
                        "arms": {
                            "open_loop": {
                                "expected_denominator": 4,
                                "completed": 4,
                                "task_failed": 0,
                                "timeout": 0,
                                "not_started": 0,
                                "infrastructure_interrupted": 0,
                            },
                            "verified": {
                                "expected": 3,
                                "completed": 4,
                                "task_failed": 0,
                                "timeout": 0,
                                "not_started": 0,
                                "infrastructure_interrupted": 0,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "open_loop expected is not 4; verified expected is not 4",
            ):
                validate_dev_run(root)


if __name__ == "__main__":
    unittest.main()
