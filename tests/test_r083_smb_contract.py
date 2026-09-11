from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.r080_smb_astar_probe import _config, build_cases, build_optimizer
from experiments.r083_smb_contract import (
    _negative_assertions,
    _positive_assertions,
    build_evaluator,
    run_audit,
    synthetic_levels,
)
from toolusepcg.config import get_control_ranges, load_config
from toolusepcg.evaluator import EvalResult
from toolusepcg.smb_contract import (
    FALL_KILL_EVENT,
    SMB_TARGET_ENVELOPES,
    STOMP_KILL_EVENT,
    iter_smb_targets,
    smb_target_error,
)


class TestSmbRangeAndExactContract(unittest.TestCase):
    def test_named_envelopes_have_140_and_210_targets(self) -> None:
        self.assertEqual(len(list(iter_smb_targets("native"))), 140)
        self.assertEqual(len(list(iter_smb_targets("expanded"))), 210)
        self.assertEqual(SMB_TARGET_ENVELOPES["native"]["coins"], (0, 3))
        self.assertEqual(SMB_TARGET_ENVELOPES["expanded"]["coins"], (0, 5))

        native = get_control_ranges("smb", 16, 32, smb_envelope="native")
        expanded = get_control_ranges("smb", 16, 32, smb_envelope="expanded")
        self.assertEqual(native["enemies"].max_value, 4)
        self.assertEqual(native["coins"].max_value, 3)
        self.assertEqual(native["jumps"].max_value, 6)
        self.assertEqual(expanded["coins"].max_value, 5)

    def test_expanded_boundary_is_valid_and_outside_is_rejected(self) -> None:
        config = load_config(
            "configs/smb.yaml",
            {"targets": {"coins": {"mode": "target", "value": 5}}},
        )
        self.assertEqual(config.targets.coins.value, 5)
        with self.assertRaisesRegex(ValueError, "out_of_contract"):
            load_config(
                "configs/smb.yaml",
                {"targets": {"coins": {"mode": "target", "value": 6}}},
            )
        with self.assertRaisesRegex(ValueError, "finite integer"):
            load_config(
                "configs/smb.yaml",
                {"targets": {"jumps": {"mode": "target", "value": 2.5}}},
            )

    def test_exact_error_has_no_tolerance(self) -> None:
        exact = smb_target_error(
            {"enemies": 1, "coins": 2, "jumps": 3},
            {"enemies": 1, "coins": 2, "jumps": 3},
        )
        miss = smb_target_error(
            {"enemies": 1, "coins": 2, "jumps": 2},
            {"enemies": 1, "coins": 2, "jumps": 3},
        )
        self.assertTrue(exact.exact)
        self.assertEqual(exact.e_inf, 0)
        self.assertFalse(miss.exact)
        self.assertEqual(miss.e_inf, 1)


class TestSmbSyntheticWitnesses(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evaluator, _ = build_evaluator()
        cls.levels = synthetic_levels()
        cls.results = {
            name: cls.evaluator.evaluate(level)
            for name, level in cls.levels.items()
        }

    def test_coin_ground_jump_stomp_and_fall_witnesses(self) -> None:
        for name in (
            "coin_pickup",
            "ground_jump",
            "stomp_kill",
            "fall_kill_excluded",
        ):
            checks = _positive_assertions(name, self.results[name])
            self.assertTrue(all(checks.values()), (name, checks))

        stomp_events = [
            event
            for event in self.results["stomp_kill"].info["game_events"]
            if event["event_type"] == STOMP_KILL_EVENT
        ]
        self.assertEqual(stomp_events[0]["mario_event_position"]["y"], 15)
        fall_events = [
            event
            for event in self.results["fall_kill_excluded"].info["game_events"]
            if event["event_type"] == FALL_KILL_EVENT
        ]
        self.assertEqual(len(fall_events), 1)
        self.assertEqual(self.results["fall_kill_excluded"].metrics["enemies"], 0)

    def test_tube_empty_and_floating_enemy_fail_closed(self) -> None:
        for name in ("invalid_tube", "empty_gate_skip", "floating_enemy_skip"):
            checks = _negative_assertions(name, self.results[name])
            self.assertTrue(all(checks.values()), (name, checks))
        self.assertGreater(
            self.results["invalid_tube"].metrics["tube_issues"], 0
        )
        self.assertFalse(self.results["invalid_tube"].valid)

    def test_normalized_trace_is_repeatable(self) -> None:
        first = self.evaluator.evaluate(self.levels["stomp_kill"])
        second = self.evaluator.evaluate(self.levels["stomp_kill"])
        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(
            first.info["normalized_event_trace"],
            second.info["normalized_event_trace"],
        )


class TestSmbPromptAndAudit(unittest.TestCase):
    def test_prompt_and_formatter_use_exact_language(self) -> None:
        optimizer = build_optimizer(
            _config(
                build_cases()[0],
                model_id="test",
                base_url="http://127.0.0.1:1/v1",
            )
        )
        level = synthetic_levels()["coin_pickup"]
        optimizer.initialize(seed=1001, initial_level=level)
        prompt = optimizer._build_smb_prompt()
        metrics = optimizer.evaluator.format_metrics_for_prompt(
            optimizer.current_result
        )
        self.assertIn("exactly", prompt)
        self.assertNotRegex(prompt, r"~\d+ (?:enemies|jumps|coins)")
        self.assertIn("exact goal", metrics)
        self.assertNotRegex(metrics, r"goal: ~")

    def test_termination_requires_joint_exact_and_hard_valid(self) -> None:
        case = build_cases()[0]
        optimizer = build_optimizer(
            _config(case, model_id="test", base_url="http://127.0.0.1:1/v1")
        )
        exact_metrics = dict(case["targets"])
        exact = EvalResult(
            metrics=exact_metrics,
            valid=True,
            problem_type="smb",
        )
        one_off = EvalResult(
            metrics={**exact_metrics, "jumps": exact_metrics["jumps"] - 1},
            valid=True,
            problem_type="smb",
        )
        invalid_exact = EvalResult(
            metrics=exact_metrics,
            valid=False,
            problem_type="smb",
        )
        self.assertTrue(optimizer.termination.should_terminate(exact))
        self.assertFalse(optimizer.termination.should_terminate(one_off))
        self.assertFalse(optimizer.termination.should_terminate(invalid_exact))

    def test_audit_writes_machine_readable_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = run_audit(Path(tmp), repeats=2)
            self.assertTrue(payload["gate_pass"])
            self.assertTrue((Path(tmp) / "audit.json").exists())
            self.assertEqual(payload["range_contract"]["native"]["count"], 140)
            self.assertEqual(payload["range_contract"]["expanded"]["count"], 210)
            self.assertTrue(
                payload["determinism"]["stomp_kill"]["byte_identical"]
            )


if __name__ == "__main__":
    unittest.main()
