from __future__ import annotations

import unittest

from experiments.r080_smb_astar_probe import _config
from experiments.r083_smb_baseline import DEFAULT_MANIFEST, build_cases
from experiments.r107_smb_mechanic_library import (
    BACKBONES,
    BACKBONE_IDS,
    LIBRARY_SHA256,
    TOOL_NAME,
    BuildSmbFromMechanicLibraryTool,
    _actual_behavior,
    _audit_records,
    _hard_valid,
    _level_sha256,
    build_library_optimizer,
    intended_parameters,
    load_backbone,
)


class R107MechanicLibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.cases = build_cases(DEFAULT_MANIFEST, seed=107100)
        cls.by_id = {case["case_id"]: case for case in cls.cases}

    def _optimizer(self, case_id: str):
        return build_library_optimizer(
            _config(
                self.by_id[case_id],
                model_id="test",
                base_url="http://127.0.0.1:1/v1",
            )
        )

    def test_catalog_is_one_dev_source_per_joint_profile(self) -> None:
        self.assertEqual(len(BACKBONES), 16)
        self.assertEqual(len(BACKBONE_IDS), 16)
        self.assertEqual(len(LIBRARY_SHA256), 64)
        self.assertEqual(
            {tuple(profile["behavior"]) for profile in BACKBONES.values()},
            {
                (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
                (2, 0), (2, 1), (2, 2), (2, 3),
                (3, 0), (3, 1),
            },
        )
        for backbone_id, profile in BACKBONES.items():
            source = self.by_id[profile["source_case_id"]]
            self.assertEqual(
                (source["targets"]["enemies"], source["targets"]["jumps"]),
                tuple(profile["behavior"]),
                backbone_id,
            )
            level = load_backbone(backbone_id)
            self.assertEqual(_level_sha256(level), profile["backbone_sha256"])
            self.assertEqual(len(profile["slots"]), 5)

    def test_every_dev_target_has_a_library_action(self) -> None:
        for case in self.cases:
            parameters = intended_parameters(case["targets"])
            self.assertIn(parameters["backbone_id"], BACKBONES)
            self.assertEqual(
                parameters["coin_slot_ids"],
                list(range(int(case["targets"]["coins"]))),
            )

    def test_representative_k3_action_is_exact_under_real_astar(self) -> None:
        case = self.by_id["smb-dev-28-k3-c4-j1"]
        optimizer = self._optimizer(case["case_id"])
        optimizer.initialize(seed=107100)
        tool = optimizer.registry.get(TOOL_NAME)
        self.assertIsInstance(tool, BuildSmbFromMechanicLibraryTool)
        result = tool.execute(**intended_parameters(case["targets"]))
        self.assertTrue(result.success, result.errors)
        evaluated = optimizer.evaluator.evaluate(optimizer.edit_manager.level)
        self.assertTrue(_hard_valid(evaluated))
        self.assertEqual(_actual_behavior(evaluated), (3, 4, 1))
        # R107 inherits SmbTargetTermination through SpawnFateSmbOptimizer, so
        # the bounded protocol loop stops after this first exact candidate.
        self.assertTrue(optimizer.termination.should_terminate(evaluated))

    def test_tool_rejects_non_prefix_coin_slots(self) -> None:
        optimizer = self._optimizer("smb-dev-00-k0-c0-j1")
        optimizer.initialize(seed=107100)
        tool = optimizer.registry.get(TOOL_NAME)
        assert isinstance(tool, BuildSmbFromMechanicLibraryTool)
        self.assertFalse(
            tool.execute(backbone_id="k0_j1", coin_slot_ids=[1]).success
        )
        self.assertFalse(
            tool.execute(backbone_id="k0_j1", coin_slot_ids=[0, 0]).success
        )
        self.assertFalse(
            tool.execute(backbone_id="not-a-profile", coin_slot_ids=[]).success
        )

    def test_runtime_schema_registers_only_the_library_tool(self) -> None:
        optimizer = self._optimizer("smb-dev-20-k2-c1-j3")
        optimizer.initialize(seed=107100)
        schema = optimizer._build_openai_extra_body()["structured_outputs"]["json"]
        decoded = {
            branch["properties"]["tool_name"]["const"]
            for branch in schema["properties"]["tool_calls"]["items"]["oneOf"]
        }
        self.assertEqual(decoded, {TOOL_NAME})
        self.assertEqual(decoded, set(optimizer.registry.list_tools()))

    def test_audit_attributes_first_candidate_to_the_library_call(self) -> None:
        target = {"complete": 1.0, "enemies": 2, "coins": 1, "jumps": 3}
        expected = intended_parameters(target)
        audit = _audit_records(
            [{
                "case_id": "synthetic",
                "targets": target,
                "exact": True,
                "steps": [{
                    "final_json": {
                        "tool_calls": [{
                            "tool_name": TOOL_NAME,
                            "parameters": expected,
                        }]
                    },
                    "metrics_after": target,
                }],
            }]
        )
        self.assertEqual(audit["library_calls"], 1)
        self.assertEqual(audit["first_choice_correct"], 1)
        self.assertEqual(audit["first_candidate_exact"], 1)


if __name__ == "__main__":
    unittest.main()
