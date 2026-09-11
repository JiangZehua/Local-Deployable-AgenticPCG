from __future__ import annotations

import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from experiments.zelda_mastery_m0 import (
    DOMAIN_SIZE,
    FAILURE_TAXONOMY,
    STRATA_MINIMUMS,
    build_target_manifest,
    construct_zelda_corridor,
    episode_schema,
    evaluate_oracle_domain,
    run_contract_gate,
    run_schema_gate,
    sha256_json,
    synthetic_failure_records,
    validate_episode,
    validate_target_manifest,
    write_m0_artifacts,
)
from toolusepcg.config import config_from_dict, config_to_dict, load_config
from toolusepcg.evaluator import EvalResult, StatsEvaluator
from toolusepcg.factory import LevelFactory
from toolusepcg.optimizer.greedy import _zelda_goal_contract
from toolusepcg.optimizer.scoring import Scorer
from toolusepcg.tools.eval_stats import CalculateStatsTool


class R060AOracleTests(unittest.TestCase):
    def test_corridor_oracle_is_961_of_961_hard_valid_exact(self) -> None:
        records, summary = evaluate_oracle_domain()
        self.assertEqual(len(records), DOMAIN_SIZE)
        self.assertEqual(summary["hard_valid"], DOMAIN_SIZE)
        self.assertEqual(summary["exact"], DOMAIN_SIZE)
        self.assertTrue(summary["gate_pass"])
        self.assertTrue(all(record["exact"] for record in records))


class R060BContractTests(unittest.TestCase):
    def test_production_full_prompt_feedback_stop_acceptance_and_rollback(self) -> None:
        gate = run_contract_gate()
        self.assertTrue(gate["gate_pass"], gate)
        self.assertEqual(gate["passed"], gate["total"])

    def test_exact_override_replaces_maximize_family(self) -> None:
        config = load_config(
            "configs/zelda.yaml",
            {"targets": {
                "player_key": {"mode": "target", "value": 17},
                "key_door": {"mode": "target", "value": 31},
            }},
        )
        self.assertEqual(config.targets.player_key.value, 17)
        self.assertIsNone(config.targets.player_key.min_threshold)
        self.assertEqual(config.targets.key_door.value, 31)
        self.assertIsNone(config.targets.key_door.min_threshold)
        self.assertIsNone(config.targets.solution_length)
        goal, instruction, _ = _zelda_goal_contract(config)
        self.assertIn("exactly 17", goal)
        self.assertIn("exactly 31", goal)
        self.assertNotIn("~", goal + instruction)
        self.assertNotIn("maximize", (goal + instruction).lower())

    def test_exact_and_maximize_components_are_rejected(self) -> None:
        raw = config_to_dict(load_config("configs/zelda.yaml"))
        raw["targets"]["player_key"] = {"mode": "target", "value": 20}
        raw["targets"]["key_door"] = {
            "mode": "maximize", "min_threshold": 16, "value": None,
        }
        with self.assertRaisesRegex(ValueError, "Exact and maximize"):
            config_from_dict(raw)

    def test_exact_target_cannot_retain_maximize_threshold(self) -> None:
        raw = config_to_dict(load_config("configs/zelda.yaml"))
        raw["targets"] = {
            "solution_length": {
                "mode": "target", "value": 40, "min_threshold": 32,
            }
        }
        with self.assertRaisesRegex(ValueError, "min_threshold"):
            config_from_dict(raw)

    def test_solution_length_only_drives_scorer_feedback_and_prompt(self) -> None:
        config = load_config(
            "configs/zelda.yaml",
            {"targets": {"solution_length": {"mode": "target", "value": 40}}},
        )
        self.assertIsNone(config.targets.player_key)
        self.assertIsNone(config.targets.key_door)
        self.assertEqual(config.targets.solution_length.value, 40)

        exact = EvalResult(
            problem_type="zelda", valid=True,
            metrics={
                "players": 1, "keys": 1, "doors": 1, "enemies": 3,
                "player_key": 10, "key_door": 30, "solution_length": 40,
                "regions": 1,
            },
        )
        miss = deepcopy(exact)
        miss.metrics["key_door"] = 29
        miss.metrics["solution_length"] = 39
        scorer = Scorer(config)
        self.assertGreater(scorer.score(exact), scorer.score(miss))

        evaluator_stub = type("EvaluatorStub", (), {"config": config})()
        gaps, summary = CalculateStatsTool(evaluator_stub, None)._compute_zelda_gaps(exact)
        self.assertEqual(gaps["solution_length"]["target"], 40)
        self.assertEqual(gaps["solution_length"]["status"], "ok")
        self.assertTrue(any("solution_length: ok" in line for line in summary))

        goal, instruction, _ = _zelda_goal_contract(config)
        self.assertIn("exactly 40", goal)
        self.assertNotIn("maximize", (goal + instruction).lower())

    def test_enemies_zero_is_exact_in_evaluator_scorer_feedback_and_prompt(self) -> None:
        config = load_config(
            "configs/zelda.yaml",
            {"targets": {"enemies": {"mode": "target", "value": 0}}},
        )
        factory = LevelFactory(config)
        evaluator = StatsEvaluator(config, factory.env)
        zero_result = evaluator.evaluate(
            construct_zelda_corridor(20, 20, enemies=0)
        )
        one_result = evaluator.evaluate(
            construct_zelda_corridor(20, 20, enemies=1)
        )
        self.assertTrue(zero_result.valid)
        self.assertFalse(one_result.valid)
        self.assertGreater(Scorer(config).score(zero_result), Scorer(config).score(one_result))

        gaps, _ = CalculateStatsTool(evaluator, None)._compute_zelda_gaps(zero_result)
        self.assertEqual(gaps["enemies"]["target"], 0)
        self.assertEqual(gaps["enemies"]["status"], "ok")
        _, _, enemy_goal = _zelda_goal_contract(config)
        self.assertEqual(enemy_goal, "5. Have exactly 0 enemies (E)")

    def test_regions_not_one_is_hard_invalid(self) -> None:
        config = load_config("configs/zelda.yaml")
        factory = LevelFactory(config)
        evaluator = StatsEvaluator(config, factory.env)
        level = construct_zelda_corridor(20, 20)
        level.data[15, 15] = 1
        result = evaluator.evaluate(level)
        self.assertEqual(result.metrics["regions"], 2)
        self.assertFalse(result.valid)
        self.assertTrue(any("need exactly 1" in error for error in result.errors))

    def test_v0_target_entry_rejects_partial_noninteger_out_of_range_and_negative(self) -> None:
        invalid_overrides = [
            {"targets": {"key_door": {"mode": "target", "value": 20}}},
            {"targets": {
                "player_key": {"mode": "target", "value": 10.5},
                "key_door": {"mode": "target", "value": 20},
            }},
            {"targets": {
                "player_key": {"mode": "target", "value": 9},
                "key_door": {"mode": "target", "value": 20},
            }},
            {"targets": {
                "player_key": {"mode": "target", "value": 20},
                "key_door": {"mode": "target", "value": 41},
            }},
            {"targets": {"enemies": {"mode": "target", "value": -1}}},
        ]
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                load_config("configs/zelda.yaml", overrides)


class R060CManifestTests(unittest.TestCase):
    def test_manifest_is_frozen_complete_disjoint_and_stratified(self) -> None:
        manifest = build_target_manifest()
        self.assertEqual(manifest, build_target_manifest())
        payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        self.assertEqual(manifest["manifest_sha256"], sha256_json(payload))
        self.assertEqual(manifest["split_sizes"], {
            "dev": 120, "validation": 120, "test": 721,
        })
        self.assertTrue(validate_target_manifest(manifest)["gate_pass"])
        pairs = [
            (entry["player_key"], entry["key_door"])
            for entry in manifest["targets"]
        ]
        self.assertEqual(len(pairs), DOMAIN_SIZE)
        self.assertEqual(len(set(pairs)), DOMAIN_SIZE)

        for split in ("dev", "validation", "test"):
            coverage = manifest["coverage"][split]
            self.assertEqual(set(coverage["corner_quadrants"]), {
                "low_low", "low_high", "high_low", "high_high",
            })
            self.assertEqual(set(coverage["boundary_sides"]), {
                "pk_min", "pk_max", "kd_min", "kd_max",
            })
            self.assertIn("pk_much_longer", coverage["imbalance"])
            self.assertIn("kd_much_longer", coverage["imbalance"])
            self.assertEqual(set(coverage["parity"]), {"ee", "eo", "oe", "oo"})
            minimums = STRATA_MINIMUMS[split]
            self.assertTrue(all(
                count >= minimums["corner"]
                for count in coverage["corner_quadrants"].values()
            ))
            self.assertTrue(all(
                count >= minimums["boundary"]
                for count in coverage["boundary_sides"].values()
            ))
            self.assertGreaterEqual(
                coverage["imbalance"]["pk_much_longer"],
                minimums["extreme_imbalance"],
            )
            self.assertGreaterEqual(
                coverage["imbalance"]["kd_much_longer"],
                minimums["extreme_imbalance"],
            )
            self.assertTrue(all(
                count >= minimums["sum_band"]
                for count in coverage["sum_bands"].values()
            ))
            self.assertTrue(all(
                count >= minimums["parity"]
                for count in coverage["parity"].values()
            ))


class R060DSchemaTests(unittest.TestCase):
    def test_schema_requires_all_plan_fields_and_synthetic_taxonomy_coverage(self) -> None:
        schema = episode_schema()
        required = set(schema["required"])
        self.assertTrue({
            "target", "split", "manifest_sha256", "model", "harness",
            "initialization", "actions", "final", "errors", "costs",
        } <= required)
        records = synthetic_failure_records()
        gate = run_schema_gate(records)
        self.assertTrue(gate["gate_pass"], gate)
        self.assertEqual(
            {record["errors"][0]["taxonomy"] for record in records},
            set(FAILURE_TAXONOMY),
        )

    def test_schema_validator_fails_closed(self) -> None:
        record = synthetic_failure_records()[0]
        del record["costs"]["verifier_calls"]
        with self.assertRaises(ValueError):
            validate_episode(record)

    def test_aggregate_gate_is_derived_and_fails_closed(self) -> None:
        with TemporaryDirectory() as temp_dir, mock.patch(
            "experiments.zelda_mastery_m0.evaluate_oracle_domain",
            return_value=([], {"gate_pass": True, "records_sha256": "0" * 64}),
        ), mock.patch(
            "experiments.zelda_mastery_m0.run_contract_gate",
            return_value={"gate_pass": False, "cases": {"forced": False}},
        ):
            with self.assertRaisesRegex(RuntimeError, "failed closed"):
                write_m0_artifacts(temp_dir)
            summary = __import__("json").loads(
                (Path(temp_dir) / "r060_m0_summary.json").read_text()
            )
            self.assertFalse(summary["component_gates"]["R060-B"])
            self.assertFalse(summary["gate_pass"])


if __name__ == "__main__":
    unittest.main()
