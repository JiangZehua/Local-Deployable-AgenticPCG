from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from experiments.r080_smb_astar_probe import _config
from experiments.r102_smb_behavior_operator import (
    OPERATOR_NAME,
    SMOKE_CASE_IDS,
    build_behavior_optimizer,
    is_exact_behavior_witness,
)
from toolusepcg.protocol.json_schema import StepMessage, ToolCall
from toolusepcg.tools.edit_smb_behavior_episode import SUPPORTED_BEHAVIOR_TARGETS


def _case(target: tuple[int, int, int]) -> dict:
    kills, coins, jumps = target
    return {
        "case_id": "test",
        "spec_id": "test",
        "seed": 102100,
        "targets": {
            "complete": 1.0,
            "enemies": kills,
            "coins": coins,
            "jumps": jumps,
        },
    }


class TestR102BehaviorOperator(unittest.TestCase):
    def test_support_matches_six_smoke_cases(self) -> None:
        self.assertEqual(len(SMOKE_CASE_IDS), 6)
        self.assertEqual(
            set(SUPPORTED_BEHAVIOR_TARGETS),
            {(1, 0, 1), (1, 1, 1), (1, 1, 2), (1, 2, 3), (2, 1, 2), (2, 5, 1)},
        )

    def test_every_transaction_is_exact_and_strictly_verified(self) -> None:
        for target, profile in SUPPORTED_BEHAVIOR_TARGETS.items():
            optimizer = build_behavior_optimizer(
                _config(_case(target), model_id="test", base_url="http://127.0.0.1:1/v1")
            )
            optimizer.initialize(seed=102100)
            before = optimizer.current_level.to_string()
            message = StepMessage(
                tool_calls=[
                    ToolCall(
                        OPERATOR_NAME,
                        {
                            "enemy_count": target[0],
                            "coin_count": target[1],
                            "jump_count": target[2],
                        },
                    )
                ]
            )
            candidate, results, changed = optimizer.execute_step(message)
            self.assertEqual(optimizer.current_level.to_string(), before)
            self.assertTrue(results[0]["result"]["success"])
            evaluated = optimizer.evaluator.evaluate(candidate)
            self.assertTrue(
                is_exact_behavior_witness(
                    evaluated, target, tuple(profile["obstacles"])
                )
            )
            accepted, reason = optimizer.decide_accept(
                evaluated,
                changed,
                new_level=candidate,
                tool_results=results,
                message=message,
            )
            self.assertTrue(accepted)
            self.assertIn("accepted_verified_behavior_operator", reason)

    def test_unsupported_triple_is_atomic_failure(self) -> None:
        target = (1, 0, 1)
        optimizer = build_behavior_optimizer(
            _config(_case(target), model_id="test", base_url="http://127.0.0.1:1/v1")
        )
        optimizer.initialize(seed=102100)
        before = optimizer.current_level.to_string()
        result = optimizer.registry.execute(
            OPERATOR_NAME, enemy_count=2, coin_count=2, jump_count=3
        )
        self.assertFalse(result.success)
        self.assertEqual(optimizer.current_level.to_string(), before)

    def test_runtime_registry_matches_coord_bounded_schema(self) -> None:
        optimizer = build_behavior_optimizer(
            _config(_case((1, 2, 3)), model_id="test", base_url="http://127.0.0.1:1/v1")
        )
        optimizer.initialize(seed=102100)
        with patch.dict(os.environ, {"RPE_PCG_COORD_GRAMMAR": "on"}):
            schema = optimizer._build_openai_extra_body()["structured_outputs"]["json"]
        decoded = {
            branch["properties"]["tool_name"]["const"]
            for branch in schema["properties"]["tool_calls"]["items"]["oneOf"]
        }
        self.assertEqual(decoded, set(optimizer.registry.list_tools()))
        self.assertIn(OPERATOR_NAME, decoded)


if __name__ == "__main__":
    unittest.main()
