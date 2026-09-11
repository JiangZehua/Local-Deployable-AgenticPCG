from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.r074_binarydoor_harness_probe import (
    TARGETS,
    LEGACY_ACCEPTANCE,
    TARGET_FIRST_ACCEPTANCE,
    TargetFirstBinaryDoorOptimizer,
    build_optimizer,
    construct_witness,
    exact_config,
    pin_registry_schema,
    preflight,
)
from toolusepcg.factory import LevelFactory
from toolusepcg.evaluator import StatsEvaluator
from toolusepcg.optimizer.greedy import GreedyOptimizer


class TestR074BinaryDoorContract(unittest.TestCase):
    def test_all_targets_have_real_exact_witnesses(self) -> None:
        for target in TARGETS:
            config = exact_config(target)
            evaluator = StatsEvaluator(config, LevelFactory(config).env)
            level, endpoints = construct_witness(target)
            metrics = evaluator.evaluate(level).metrics
            self.assertEqual(metrics["door_path"], target)
            self.assertEqual(metrics["num_connected_regions"], 1)
            self.assertEqual(endpoints[0], 0)
            self.assertEqual(endpoints[-1], 9)

    def test_config_is_exact_empty_and_place_tile_only(self) -> None:
        config = exact_config(64)
        self.assertEqual(config.targets.door_path.mode, "target")
        self.assertEqual(config.targets.door_path.value, 64)
        self.assertEqual(config.targets.num_connected_regions.value, 1)
        self.assertEqual(config.optimizer.init_strategy, "empty")
        self.assertEqual(config.optimizer.edit_tools, ["place_tile"])
        self.assertEqual(config.optimizer.max_tool_calls_per_step, 20)
        self.assertEqual(config.optimizer.conversation_length, 2)

    def test_schema_has_only_advertised_tool(self) -> None:
        old = os.environ.get("RPE_PCG_COORD_GRAMMAR")
        os.environ["RPE_PCG_COORD_GRAMMAR"] = "on"
        try:
            optimizer = GreedyOptimizer(exact_config(64), verbose=False)
            optimizer.initialize(seed=42)
            schema = pin_registry_schema(optimizer)
        finally:
            if old is None:
                os.environ.pop("RPE_PCG_COORD_GRAMMAR", None)
            else:
                os.environ["RPE_PCG_COORD_GRAMMAR"] = old
        rendered = str(schema)
        self.assertIn("place_tile", rendered)
        self.assertNotIn("generate_bsp", rendered)
        self.assertNotIn("generate_ca", rendered)

    def test_empty_initialization_is_identical_across_targets(self) -> None:
        levels = set()
        for target in TARGETS:
            optimizer = GreedyOptimizer(exact_config(target), verbose=False)
            optimizer.initialize(seed=42)
            levels.add(optimizer.current_level.to_string())
            self.assertEqual(optimizer.current_result.metrics["door_path"], 22)
            self.assertEqual(
                optimizer.current_result.metrics["num_connected_regions"], 1
            )
        self.assertEqual(len(levels), 1)

    def test_preflight_materializes_auditable_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "preflight.json"
            result = preflight(output)
            self.assertTrue(result["gate_pass"])
            self.assertTrue(result["all_targets_oracle_reachable"])
            self.assertEqual(result["target_count"], 8)
            self.assertTrue(output.is_file())

    def test_legacy_default_preserves_scalar_rejection(self) -> None:
        config = exact_config(32)
        optimizer = build_optimizer(config)
        self.assertIs(type(optimizer), GreedyOptimizer)
        self.assertEqual(LEGACY_ACCEPTANCE, "legacy_score")
        optimizer.initialize(seed=42)
        candidate, _ = construct_witness(32)
        result = optimizer.evaluator.evaluate(candidate)
        result.metrics["change_ratio"] = float(
            np.mean(candidate.data != optimizer.current_level.data)
        )
        accepted, reason = optimizer.decide_accept(result, num_changes=1)
        self.assertFalse(accepted)
        self.assertIn("rejected", reason)

    def test_target_first_accepts_exact_candidate_despite_rewrite(self) -> None:
        config = exact_config(32)
        optimizer = build_optimizer(
            config, acceptance_policy=TARGET_FIRST_ACCEPTANCE
        )
        self.assertIsInstance(optimizer, TargetFirstBinaryDoorOptimizer)
        optimizer.initialize(seed=42)
        candidate, _ = construct_witness(32)
        result = optimizer.evaluator.evaluate(candidate)
        result.metrics["change_ratio"] = float(
            np.mean(candidate.data != optimizer.current_level.data)
        )
        self.assertLess(optimizer.scorer.score(result), optimizer.current_score)
        accepted, reason = optimizer.decide_accept(result, num_changes=1)
        self.assertTrue(accepted)
        self.assertIn("accepted_target_primary", reason)

    def test_target_first_stops_on_exact_binarydoor_result(self) -> None:
        config = exact_config(40)
        optimizer = build_optimizer(
            config, acceptance_policy=TARGET_FIRST_ACCEPTANCE
        )
        optimizer.initialize(seed=42)
        candidate, _ = construct_witness(40)
        result = optimizer.evaluator.evaluate(candidate)
        self.assertTrue(optimizer.termination.should_terminate(result))
        self.assertFalse(
            GreedyOptimizer(config, verbose=False).termination.should_terminate(result)
        )
