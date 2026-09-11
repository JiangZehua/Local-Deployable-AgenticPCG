from __future__ import annotations

import json
import unittest

from experiments.r108_smb_functional_harness import (
    ALL_LANGUAGE_METRICS,
    DEFAULT_LANGUAGE_MANIFEST,
    LANGUAGE_SCHEMA,
    build_resolver,
    derive_parser_disposition,
    is_fail_closed,
    parser_payload_to_spec,
)
from toolusepcg.functional_resolver import (
    FunctionalObjective,
    FunctionalSpec,
    MetricConstraint,
    select_satisfying_target,
)


def _dynamic(enemies: int, coins: int, jumps: int) -> FunctionalSpec:
    return FunctionalSpec(
        "smb_dynamic",
        {
            "enemies": MetricConstraint("exact", enemies),
            "coins": MetricConstraint("exact", coins),
            "jumps": MetricConstraint("exact", jumps),
        },
    )


class FunctionalResolverCoreTests(unittest.TestCase):
    def test_constraint_selection_is_objective_then_stable(self) -> None:
        spec = FunctionalSpec(
            "synthetic",
            {"kills": MetricConstraint("min", 2)},
            (FunctionalObjective("jumps", "minimize"),),
        )
        selected = select_satisfying_target(
            spec,
            [
                {"kills": 3, "jumps": 1},
                {"kills": 2, "jumps": 1},
                {"kills": 2, "jumps": 2},
            ],
            stable_metric_order=("kills", "jumps"),
        )
        self.assertEqual(selected, {"kills": 2, "jumps": 1})

    def test_unregistered_domain_is_unsupported_without_evaluation(self) -> None:
        result = build_resolver().resolve(
            FunctionalSpec("zelda", {"enemies": MetricConstraint("exact", 1)})
        )
        self.assertEqual(result.status, "unsupported")
        self.assertEqual(result.evaluator_calls, 0)


class SMBFunctionalAdapterTests(unittest.TestCase):
    def test_dynamic_archive_hit_is_one_verified_evaluation(self) -> None:
        result = build_resolver().resolve(_dynamic(3, 4, 1), max_evaluator_calls=4)
        self.assertEqual(result.status, "exact")
        self.assertEqual(result.source, "archive_or_compiler_hit")
        self.assertEqual(result.evaluator_calls, 1)
        self.assertEqual(result.selected_target, {"enemies": 3, "coins": 4, "jumps": 1})

    def test_jump_archive_miss_routes_to_bounded_search(self) -> None:
        result = build_resolver(disabled_backbones={"k0_j5"}).resolve(
            _dynamic(0, 5, 5), max_evaluator_calls=8
        )
        self.assertEqual(result.status, "exact")
        self.assertEqual(result.source, "bounded_search")
        self.assertEqual(result.strategy, "jump_episode_search")
        self.assertEqual(result.evaluator_calls, 2)
        self.assertEqual(result.search["route"], "jump_episode")

    def test_stomp_archive_miss_routes_to_bounded_search(self) -> None:
        result = build_resolver(disabled_backbones={"k3_j0"}).resolve(
            _dynamic(3, 3, 0), max_evaluator_calls=8
        )
        self.assertEqual(result.status, "exact")
        self.assertEqual(result.source, "bounded_search")
        self.assertEqual(result.strategy, "stomp_corridor_search")
        self.assertEqual(result.evaluator_calls, 2)
        self.assertEqual(result.search["route"], "stomp_corridor")

    def test_range_and_objective_select_certified_target(self) -> None:
        spec = FunctionalSpec(
            "smb_dynamic",
            {
                "enemies": MetricConstraint("min", 2),
                "coins": MetricConstraint("exact", 3),
            },
            (FunctionalObjective("jumps", "minimize"),),
        )
        result = build_resolver().resolve(spec, max_evaluator_calls=4)
        self.assertEqual(result.status, "exact")
        self.assertEqual(result.selected_target, {"enemies": 2, "coins": 3, "jumps": 0})

    def test_static_target_uses_playable_compiler_and_real_simulation(self) -> None:
        spec = FunctionalSpec(
            "smb_static",
            {
                "pipe_count": MetricConstraint("exact", 4),
                "enemy_tile_count": MetricConstraint("exact", 6),
                "coin_tile_count": MetricConstraint("exact", 5),
                "block_tile_count": MetricConstraint("exact", 32),
            },
        )
        result = build_resolver().resolve(spec, max_evaluator_calls=4)
        self.assertEqual(result.status, "exact")
        self.assertEqual(result.strategy, "static_playable_compiler")
        self.assertTrue(result.search["attempted"])
        self.assertTrue(result.provenance["simulation_called"])
        self.assertGreaterEqual(result.evaluator_calls, 2)
        self.assertEqual(result.observed_metrics["complete"], 1.0)
        self.assertEqual(result.observed_metrics["coin_tiles_collected"], 5)
        self.assertEqual(result.observed_metrics["unsupported_enemy_count"], 0)
        self.assertEqual(result.observed_metrics["unsupported_pipe_count"], 0)
        serialized = json.dumps(result.as_dict(include_level=True))
        self.assertTrue(json.loads(serialized)["evaluation_info"]["simulation_called"])

    def test_zero_budget_miss_is_unknown_not_infeasible(self) -> None:
        result = build_resolver().resolve(_dynamic(4, 0, 6), max_evaluator_calls=0)
        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.evaluator_calls, 0)
        payload = result.as_dict(include_level=False)
        self.assertIn("not monetary cost", payload["evaluator_accounting"])


class LanguageContractTests(unittest.TestCase):
    def test_frozen_manifest_is_complete_and_covers_both_domains(self) -> None:
        payload = json.loads(DEFAULT_LANGUAGE_MANIFEST.read_text())
        self.assertTrue(payload["frozen"])
        self.assertEqual(len(payload["cases"]), 24)
        expected_domains = {row["expected"]["domain"] for row in payload["cases"]}
        self.assertEqual(expected_domains, {"smb_dynamic", "smb_static", "unknown"})
        self.assertTrue(any(row.get("disabled_backbones") for row in payload["cases"]))

    def test_language_schema_separates_typed_parse_from_execution_status(self) -> None:
        constraints = LANGUAGE_SCHEMA["properties"]["constraints"]
        self.assertEqual(set(constraints["required"]), set(ALL_LANGUAGE_METRICS))
        self.assertNotIn("status", LANGUAGE_SCHEMA["properties"])
        self.assertIn("ambiguities", LANGUAGE_SCHEMA["required"])
        self.assertIn(
            "unspecified",
            constraints["properties"]["jumps"]["properties"]["relation"]["enum"],
        )

    def test_language_prompt_makes_cross_domain_fields_mutually_exclusive(self) -> None:
        from experiments.r108_smb_functional_harness import LANGUAGE_SYSTEM_PROMPT

        self.assertIn(
            "when domain=smb_static, enemies, coins, and jumps MUST all be unspecified",
            LANGUAGE_SYSTEM_PROMPT,
        )
        self.assertIn(
            "Never copy a tile count into its similarly named gameplay-event field",
            LANGUAGE_SYSTEM_PROMPT,
        )

    def test_parser_payload_converts_relations_and_objectives(self) -> None:
        constraints = {
            name: {"relation": "unspecified", "value": -1}
            for name in ALL_LANGUAGE_METRICS
        }
        constraints["enemies"] = {"relation": "min", "value": 2}
        constraints["coins"] = {"relation": "exact", "value": 3}
        spec = parser_payload_to_spec(
            {
                "domain": "smb_dynamic",
                "constraints": constraints,
                "objectives": [{"metric": "jumps", "direction": "minimize"}],
                "ambiguities": [],
            }
        )
        self.assertEqual(spec.constraints["enemies"].relation, "min")
        self.assertNotIn("jumps", spec.constraints)
        self.assertEqual(spec.objectives[0].metric, "jumps")

    def test_validator_clean_spec_is_resolved_deterministically(self) -> None:
        constraints = {
            name: {"relation": "unspecified", "value": -1}
            for name in ALL_LANGUAGE_METRICS
        }
        constraints["enemies"] = {"relation": "exact", "value": 0}
        constraints["coins"] = {"relation": "exact", "value": 0}
        constraints["jumps"] = {"relation": "exact", "value": 5}
        decision = derive_parser_disposition({
            "domain": "smb_dynamic",
            "constraints": constraints,
            "objectives": [],
            "ambiguities": [],
        })
        self.assertEqual(decision["status"], "resolved")

    def test_ambiguity_unknown_domain_and_out_of_range_fail_closed(self) -> None:
        base = {
            "domain": "smb_dynamic",
            "constraints": {
                name: {"relation": "unspecified", "value": -1}
                for name in ALL_LANGUAGE_METRICS
            },
            "objectives": [],
            "ambiguities": [],
        }
        vague = json.loads(json.dumps(base))
        vague["ambiguities"] = [{"kind": "vague_numeric", "evidence": "a few"}]
        self.assertEqual(derive_parser_disposition(vague)["status"], "needs_clarification")

        unknown = json.loads(json.dumps(base))
        unknown["domain"] = "unknown"
        self.assertEqual(derive_parser_disposition(unknown)["status"], "unsupported")

        out_of_range = json.loads(json.dumps(base))
        out_of_range["constraints"]["enemies"] = {"relation": "exact", "value": 10}
        self.assertEqual(derive_parser_disposition(out_of_range)["status"], "unsupported")

    def test_fail_closed_does_not_require_the_same_closed_subtype(self) -> None:
        self.assertTrue(
            is_fail_closed("needs_clarification", "unsupported", None)
        )
        self.assertTrue(
            is_fail_closed("unsupported", "needs_clarification", None)
        )
        self.assertFalse(is_fail_closed("needs_clarification", "resolved", None))
        self.assertFalse(is_fail_closed("unsupported", "unsupported", object()))


if __name__ == "__main__":
    unittest.main()
