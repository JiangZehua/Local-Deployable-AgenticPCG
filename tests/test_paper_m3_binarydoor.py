from __future__ import annotations

import os
import unittest
from pathlib import Path

from experiments.paper_m3_binarydoor import (
    GENERAL_SCAFFOLD,
    NEW_CASES,
    exact_config,
    oracle_witness,
    validate_manifest,
)
from experiments.r074_binarydoor_harness_probe import (
    TARGET_FIRST_ACCEPTANCE,
    build_optimizer,
)


class PaperM3BinaryDoorTests(unittest.TestCase):
    def test_frozen_manifest_replays_all_oracles(self) -> None:
        manifest = validate_manifest(Path("configs/paper_m3_binarydoor_manifest.json"))
        heldout = [row for row in manifest["cases"] if row["block"] == "heldout32"]
        regression = [row for row in manifest["cases"] if row["block"] == "regression8"]
        self.assertEqual(len(heldout), NEW_CASES)
        self.assertEqual(len(regression), 8)
        self.assertEqual(len({row["door_seed"] for row in heldout}), NEW_CASES)
        self.assertEqual({row["scaffold"] for row in heldout}, {"general_geometry_v2"})

    def test_oracle_is_exact_and_one_region_on_nonlegacy_geometry(self) -> None:
        level, oracle = oracle_witness(6499025, 8161489857950186643)
        config = exact_config(oracle["target"], oracle["door_seed"])
        optimizer = build_optimizer(config, acceptance_policy=TARGET_FIRST_ACCEPTANCE)
        optimizer.initialize(seed=oracle["door_seed"])
        result = optimizer.evaluator.evaluate(level)
        self.assertEqual(result.metrics["door_path"], oracle["target"])
        self.assertEqual(result.metrics["num_connected_regions"], 1)
        self.assertNotEqual(oracle["door_seed"], 42)

    def test_h0_binarydoor_prompt_hides_observations(self) -> None:
        old = os.environ.get("RPE_PCG_COORD_GRAMMAR")
        os.environ["RPE_PCG_COORD_GRAMMAR"] = "on"
        try:
            config = exact_config(43, 6499024, ablation_mode="no_evaluation_loop")
            optimizer = build_optimizer(config, acceptance_policy=TARGET_FIRST_ACCEPTANCE)
            optimizer.initialize(seed=6499024)
            optimizer.evaluator.format_metrics_for_prompt = lambda _: "METRIC_SENTINEL"
            optimizer.scorer.format_score = lambda _: "SCORE_SENTINEL"
            prompt = optimizer._build_binarydoor_prompt()
        finally:
            if old is None:
                os.environ.pop("RPE_PCG_COORD_GRAMMAR", None)
            else:
                os.environ["RPE_PCG_COORD_GRAMMAR"] = old
        self.assertNotIn("METRIC_SENTINEL", prompt)
        self.assertNotIn("SCORE_SENTINEL", prompt)
        self.assertIn("door-to-door path length of ~43", prompt)
        self.assertIn("Use the available action budget", prompt)

    def test_general_scaffold_has_no_seed42_coordinates(self) -> None:
        text = GENERAL_SCAFFOLD.read_text()
        self.assertIn("never assume the seed-42 door positions", text)
        self.assertNotIn("y=4,x=0", text)
        self.assertNotIn("y=15,x=9", text)


if __name__ == "__main__":
    unittest.main()
