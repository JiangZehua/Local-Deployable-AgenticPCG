from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import experiments.r085_smb_spawn_fate as spawn_fate
from experiments.r085_smb_spawn_fate import (
    attribute_coin_spawn_fates,
    attribute_enemy_spawn_fates,
)
from toolusepcg.smb_contract import COLLECT_EVENT, FALL_KILL_EVENT, STOMP_KILL_EVENT


def _base_info():
    return {
        "width": 32,
        "height": 16,
        "locations": [[80.0, 239.0], [96.0, 239.0], [112.0, 239.0]],
        "spawn_positions": {"enemies": [], "coins": []},
        "enemy_frames": [[], [], []],
        "game_events": [],
    }


class TestR085SmbSpawnFate(unittest.TestCase):
    def test_stomp_event_is_attached_to_observed_spawn(self) -> None:
        info = _base_info()
        info["spawn_positions"]["enemies"] = [{"x": 4, "y": 14, "tile": 7}]
        info["enemy_frames"] = [
            [[2, 120.0, 239.0]],
            [[2, 112.0, 239.0]],
            [],
        ]
        info["game_events"] = [
            {
                "type": STOMP_KILL_EVENT,
                "param": 2,
                "mario_x": 112.0,
                "mario_y": 239.0,
                "time": 2,
                "mario_event_position": {"x": 4, "y": 14},
            }
        ]
        fates = attribute_enemy_spawn_fates(info)
        self.assertEqual(fates[0]["fate"], "stomped")
        self.assertEqual(fates[0]["observed_frames"], 2)
        self.assertIsNotNone(fates[0]["closest_approach_tiles"])

    def test_fall_is_excluded_and_unreached_is_not_called_a_kill(self) -> None:
        info = _base_info()
        info["spawn_positions"]["enemies"] = [
            {"x": 4, "y": 14, "tile": 7},
            {"x": 25, "y": 14, "tile": 7},
        ]
        info["enemy_frames"] = [
            [[2, 120.0, 239.0]],
            [[2, 120.0, 260.0]],
            [],
        ]
        info["game_events"] = [
            {
                "type": FALL_KILL_EVENT,
                "param": 2,
                "mario_x": 112.0,
                "mario_y": 239.0,
                "time": 2,
                "mario_event_position": {"x": 4, "y": 14},
            }
        ]
        fates = attribute_enemy_spawn_fates(info)
        self.assertEqual(fates[0]["fate"], "fell_excluded")
        self.assertEqual(fates[1]["fate"], "unreached")

    def test_observed_enemy_without_kill_is_conservative(self) -> None:
        info = _base_info()
        info["spawn_positions"]["enemies"] = [{"x": 4, "y": 14, "tile": 7}]
        info["enemy_frames"] = [[[2, 120.0, 239.0]], [], []]
        fate = attribute_enemy_spawn_fates(info)[0]
        self.assertEqual(fate["fate"], "avoided_or_survived")

    def test_coin_events_are_one_to_one_and_unmatched_stay_uncollected(self) -> None:
        info = _base_info()
        info["spawn_positions"]["coins"] = [
            {"x": 5, "y": 14},
            {"x": 12, "y": 14},
        ]
        info["game_events"] = [
            {
                "type": COLLECT_EVENT,
                "param": 15,
                "mario_event_position": {"x": 4, "y": 14},
            }
        ]
        fates = attribute_coin_spawn_fates(info)
        self.assertEqual([item["fate"] for item in fates], ["collected", "uncollected"])

    @patch("experiments.r085_smb_spawn_fate.baseline_run")
    def test_runner_isolates_spawn_fate_as_single_changed_factor(self, mocked) -> None:
        mocked.return_value = {"contract_gate_pass": True}
        result = spawn_fate.run(
            Path("manifest.json"),
            "http://127.0.0.1:1234/v1",
            Path("result.json"),
            model_id="model",
            revision="revision",
        )
        self.assertTrue(result["contract_gate_pass"])
        kwargs = mocked.call_args.kwargs
        self.assertIs(kwargs["optimizer_builder"], spawn_fate.build_spawn_fate_optimizer)
        self.assertEqual(
            kwargs["harness_overrides"]["single_changed_factor"],
            "candidate-local spawn fate feedback",
        )
        self.assertEqual(kwargs["experiment_name"], "R085-F2")


if __name__ == "__main__":
    unittest.main()
