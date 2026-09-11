from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from toolusepcg.protocol.grammar import build_grammar_for_domain
from experiments.r080_smb_astar_probe import _config
from experiments.r100_smb_corridor_stomp_atlas import is_exact_stomp_witness
from experiments.r100_smb_stomp_operator import (
    OPERATOR_NAME,
    SMOKE_CASE_IDS,
    build_operator_optimizer,
)
from toolusepcg.optimizer.greedy import GreedyOptimizer
from toolusepcg.protocol.json_schema import StepMessage, ToolCall
from toolusepcg.tools.edit_smb_stomp_corridor import SUPPORTED_STOMP_TARGETS


def _case(kills: int, coins: int) -> dict:
    return {
        "case_id": "test",
        "spec_id": "test",
        "seed": 100100,
        "targets": {
            "complete": 1.0,
            "enemies": kills,
            "coins": coins,
            "jumps": 0,
        },
    }


class TestR100StompOperator(unittest.TestCase):
    def test_operator_support_matches_the_six_frozen_smoke_targets(self) -> None:
        self.assertEqual(len(SMOKE_CASE_IDS), 6)
        self.assertEqual(
            set(SUPPORTED_STOMP_TARGETS),
            {(1, 0), (1, 5), (2, 4), (2, 5), (3, 2), (3, 3)},
        )

    def test_each_transaction_builds_an_actual_exact_stomp_witness(self) -> None:
        for kills, coins in SUPPORTED_STOMP_TARGETS:
            optimizer = build_operator_optimizer(
                _config(
                    _case(kills, coins),
                    model_id="test",
                    base_url="http://127.0.0.1:1/v1",
                )
            )
            optimizer.initialize(seed=100100)
            before = optimizer.current_level.to_string()
            message = StepMessage(
                tool_calls=[
                    ToolCall(
                        OPERATOR_NAME,
                        {"enemy_count": kills, "coin_count": coins},
                    )
                ]
            )
            candidate, results, changed = optimizer.execute_step(message)
            self.assertEqual(optimizer.current_level.to_string(), before)
            self.assertTrue(results[0]["result"]["success"])
            self.assertTrue(results[0]["result"]["data"]["transactional"])
            self.assertGreater(changed, 0)
            evaluated = optimizer.evaluator.evaluate(candidate)
            self.assertTrue(
                is_exact_stomp_witness(evaluated, (kills, coins, 0))
            )
            accepted, reason = optimizer.decide_accept(
                evaluated,
                changed,
                new_level=candidate,
                tool_results=results,
                message=message,
            )
            self.assertTrue(accepted)
            self.assertIn("accepted_verified_stomp_operator", reason)

    def test_unsupported_pair_fails_without_mutation(self) -> None:
        optimizer = build_operator_optimizer(
            _config(
                _case(1, 0),
                model_id="test",
                base_url="http://127.0.0.1:1/v1",
            )
        )
        optimizer.initialize(seed=100100)
        before = optimizer.current_level.to_string()
        result = optimizer.registry.execute(
            OPERATOR_NAME, enemy_count=3, coin_count=5
        )
        self.assertFalse(result.success)
        self.assertEqual(optimizer.current_level.to_string(), before)

    def test_runtime_registry_and_decoder_schema_have_exact_tool_parity(self) -> None:
        optimizer = build_operator_optimizer(
            _config(
                _case(2, 5),
                model_id="test",
                base_url="http://127.0.0.1:1/v1",
            )
        )
        optimizer.initialize(seed=100100)
        with patch.dict(os.environ, {"RPE_PCG_COORD_GRAMMAR": "on"}):
            schema = optimizer._build_openai_extra_body()["structured_outputs"]["json"]
        decoded = {
            branch["properties"]["tool_name"]["const"]
            for branch in schema["properties"]["tool_calls"]["items"]["oneOf"]
        }
        self.assertEqual(decoded, set(optimizer.registry.list_tools()))
        self.assertIn(OPERATOR_NAME, decoded)

    def test_default_smb_schema_is_unchanged_by_runtime_registry_source(self) -> None:
        config = _config(
            _case(1, 0), model_id="test", base_url="http://127.0.0.1:1/v1"
        )
        optimizer = GreedyOptimizer(config, verbose=False)
        optimizer.initialize(seed=100100)
        coord_bounds = (
            (int(config.env.height), int(config.env.width))
            if os.environ.get("RPE_PCG_COORD_GRAMMAR") == "on"
            else None
        )
        expected = build_grammar_for_domain(
            "smb", max_tool_calls=20, coord_bounds=coord_bounds
        ).json_schema
        actual = optimizer._build_openai_extra_body()["structured_outputs"]["json"]
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
