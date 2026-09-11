from __future__ import annotations

import unittest
from pathlib import Path

from experiments.r083_smb_atlas import _level_sha256
from experiments.r083_smb_atlas_expansion import (
    FRONTIER_KJ,
    SearchConfig,
    _coupling_candidates,
    _jump_candidates,
    _kill_candidates,
    _load_structural_seeds,
    target_conditioned_candidates,
)


ROOT = Path(__file__).resolve().parent.parent
ATLAS = ROOT / "benchmarks" / "smb" / "search_seeds" / "atlas.json"


class TestSmbAtlasExpansionGrammar(unittest.TestCase):
    def test_pre_registered_config_rejects_weak_certification(self) -> None:
        with self.assertRaisesRegex(ValueError, "recertify_repeats"):
            SearchConfig(recertify_repeats=1).validate()
        with self.assertRaisesRegex(ValueError, "saturation_patience"):
            SearchConfig(saturation_patience=1).validate()

    def test_initial_structural_seeds_cover_14_kill_jump_states(self) -> None:
        seeds = _load_structural_seeds(ATLAS)
        self.assertEqual(len(seeds), 14)
        self.assertEqual(len({behavior for behavior, _ in seeds}), 14)
        self.assertIn((0, 5), {behavior for behavior, _ in seeds})
        self.assertIn((3, 1), {behavior for behavior, _ in seeds})

    def test_each_search_family_is_nonempty_and_deterministic(self) -> None:
        families = [_kill_candidates(), _jump_candidates(), _coupling_candidates(ATLAS)]
        for candidates in families:
            self.assertTrue(candidates)
            hashes = [_level_sha256(candidate.level) for candidate in candidates]
            self.assertEqual(len(hashes), len(set(hashes)))
        first = [_level_sha256(candidate.level) for candidate in target_conditioned_candidates(ATLAS)]
        second = [_level_sha256(candidate.level) for candidate in target_conditioned_candidates(ATLAS)]
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))

    def test_frontier_contains_all_pre_registered_boundaries(self) -> None:
        for target in ((4, 0), (4, 1), (0, 0), (0, 6), (3, 2), (2, 3), (1, 4)):
            self.assertIn(target, FRONTIER_KJ)


if __name__ == "__main__":
    unittest.main()
