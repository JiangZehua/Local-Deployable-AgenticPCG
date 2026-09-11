from __future__ import annotations

import unittest

from experiments.r083_smb_contract import build_evaluator
from experiments.r101_smb_jump_episode_atlas import (
    OBSTACLE_PROFILES,
    TARGETS,
    SearchConfig,
    classify_jump_causes,
    is_exact_jump_witness,
    jump_episode_level,
)


class TestR101JumpEpisodeAtlas(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evaluator, _ = build_evaluator()

    def test_frozen_targets_cover_jump_one_through_five(self) -> None:
        self.assertEqual({target[2] for target in TARGETS}, {1, 2, 3, 4, 5})
        self.assertEqual(len(TARGETS), 6)

    def test_config_requires_real_recertification_and_finite_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "recertify_repeats"):
            SearchConfig(recertify_repeats=1).validate()
        with self.assertRaisesRegex(ValueError, "max_coin_candidates"):
            SearchConfig(max_coin_candidates_per_target=0).validate()

    def test_every_structural_profile_has_one_terminal_plus_obstacles(self) -> None:
        for jumps, obstacles in OBSTACLE_PROFILES.items():
            result = self.evaluator.evaluate(jump_episode_level(jumps))
            self.assertTrue(
                is_exact_jump_witness(result, (0, 0, jumps), obstacles)
            )
            causes = classify_jump_causes(result, obstacles)["counts"]
            self.assertEqual(causes["terminal"], 1)
            self.assertEqual(causes["obstacle"], jumps - 1)
            self.assertEqual(causes["gap"], 0)
            self.assertEqual(causes["unknown"], 0)


if __name__ == "__main__":
    unittest.main()
