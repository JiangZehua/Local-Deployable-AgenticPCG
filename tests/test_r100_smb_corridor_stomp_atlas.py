from __future__ import annotations

import unittest

from experiments.r100_smb_corridor_stomp_atlas import (
    ENEMY_PROFILES,
    TARGETS,
    SearchConfig,
    coin_overlay_candidates,
    is_exact_stomp_witness,
    structural_candidates,
)
from experiments.r083_smb_contract import build_evaluator


class TestR100CorridorStompAtlas(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evaluator, _ = build_evaluator()

    def test_target_set_is_the_frozen_r099_kill_j0_boundary(self) -> None:
        self.assertEqual(
            TARGETS,
            (
                (1, 0, 0),
                (1, 5, 0),
                (2, 4, 0),
                (2, 5, 0),
                (3, 2, 0),
                (3, 3, 0),
            ),
        )

    def test_search_config_rejects_weak_or_unbounded_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "recertify_repeats"):
            SearchConfig(recertify_repeats=1).validate()
        with self.assertRaisesRegex(ValueError, "max_coin_candidates"):
            SearchConfig(max_coin_candidates_per_target=0).validate()

    def test_structural_profiles_are_bounded_and_deterministic(self) -> None:
        self.assertEqual(set(ENEMY_PROFILES), {1, 2, 3})
        for kills in (1, 2, 3):
            first = structural_candidates(kills)
            second = structural_candidates(kills)
            self.assertEqual(
                [candidate.level.to_string() for candidate in first],
                [candidate.level.to_string() for candidate in second],
            )
            self.assertEqual(len(first), 4)
            self.assertTrue(all(len(profile) == kills for profile in ENEMY_PROFILES[kills]))

    def test_first_profiles_are_actual_stomp_only_j0_witnesses(self) -> None:
        for kills in (1, 2, 3):
            candidate = structural_candidates(kills)[0]
            result = self.evaluator.evaluate(candidate.level)
            self.assertTrue(is_exact_stomp_witness(result, (kills, 0, 0)))

    def test_trajectory_coin_overlay_preserves_k3_stomp_semantics(self) -> None:
        base = structural_candidates(3)[0]
        base_result = self.evaluator.evaluate(base.level)
        candidate = next(coin_overlay_candidates(base, base_result, 3))
        result = self.evaluator.evaluate(candidate.level)
        self.assertTrue(is_exact_stomp_witness(result, (3, 3, 0)))


if __name__ == "__main__":
    unittest.main()
