from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from experiments.r080_smb_astar_probe import _config
from experiments.r101_smb_jump_episode_atlas import is_exact_jump_witness
from experiments.r101_smb_jump_operator import OPERATOR_NAME, SMOKE_CASE_IDS, build_jump_optimizer
from toolusepcg.protocol.json_schema import StepMessage, ToolCall
from toolusepcg.tools.edit_smb_jump_episode import SUPPORTED_JUMP_TARGETS


def _case(coins: int, jumps: int) -> dict:
    return {"case_id": "test", "spec_id": "test", "seed": 101100, "targets": {"complete": 1.0, "enemies": 0, "coins": coins, "jumps": jumps}}


class TestR101JumpOperator(unittest.TestCase):
    def test_support_matches_six_smoke_cases(self) -> None:
        self.assertEqual(len(SMOKE_CASE_IDS), 6)
        self.assertEqual(set(SUPPORTED_JUMP_TARGETS), {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 5)})

    def test_every_transaction_is_exact_and_causally_verified(self) -> None:
        for coins, jumps in SUPPORTED_JUMP_TARGETS:
            optimizer = build_jump_optimizer(_config(_case(coins, jumps), model_id="test", base_url="http://127.0.0.1:1/v1"))
            optimizer.initialize(seed=101100)
            before = optimizer.current_level.to_string()
            message = StepMessage(tool_calls=[ToolCall(OPERATOR_NAME, {"coin_count": coins, "jump_count": jumps})])
            candidate, results, changed = optimizer.execute_step(message)
            self.assertEqual(optimizer.current_level.to_string(), before)
            self.assertTrue(results[0]["result"]["success"])
            obstacles = tuple(results[0]["result"]["data"]["obstacle_x"])
            evaluated = optimizer.evaluator.evaluate(candidate)
            self.assertTrue(is_exact_jump_witness(evaluated, (0, coins, jumps), obstacles))
            accepted, reason = optimizer.decide_accept(evaluated, changed, new_level=candidate, tool_results=results, message=message)
            self.assertTrue(accepted)
            self.assertIn("accepted_verified_jump_operator", reason)

    def test_unsupported_pair_is_atomic_failure(self) -> None:
        optimizer = build_jump_optimizer(_config(_case(0, 1), model_id="test", base_url="http://127.0.0.1:1/v1"))
        optimizer.initialize(seed=101100)
        before = optimizer.current_level.to_string()
        result = optimizer.registry.execute(OPERATOR_NAME, coin_count=5, jump_count=1)
        self.assertFalse(result.success)
        self.assertEqual(optimizer.current_level.to_string(), before)

    def test_runtime_registry_matches_coord_bounded_schema(self) -> None:
        optimizer = build_jump_optimizer(_config(_case(3, 4), model_id="test", base_url="http://127.0.0.1:1/v1"))
        optimizer.initialize(seed=101100)
        with patch.dict(os.environ, {"RPE_PCG_COORD_GRAMMAR": "on"}):
            schema = optimizer._build_openai_extra_body()["structured_outputs"]["json"]
        decoded = {branch["properties"]["tool_name"]["const"] for branch in schema["properties"]["tool_calls"]["items"]["oneOf"]}
        self.assertEqual(decoded, set(optimizer.registry.list_tools()))
        self.assertIn(OPERATOR_NAME, decoded)


if __name__ == "__main__":
    unittest.main()
