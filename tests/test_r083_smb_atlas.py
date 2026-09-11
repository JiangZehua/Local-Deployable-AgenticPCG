from __future__ import annotations

import unittest

from experiments.r083_smb_atlas import (
    ALLOWED_STATUSES,
    UNKNOWN,
    build_manifest,
    structural_candidates,
)


class TestSmbAtlasContract(unittest.TestCase):
    def test_structural_candidate_generation_is_deterministic_and_unique(self) -> None:
        first = structural_candidates()
        second = structural_candidates()
        first_levels = [candidate.level.to_string() for candidate in first]
        second_levels = [candidate.level.to_string() for candidate in second]
        self.assertEqual(first_levels, second_levels)
        self.assertEqual(len(first_levels), len(set(first_levels)))
        self.assertGreater(len(first_levels), 300)

    def test_missing_witnesses_are_unknown_never_infeasible(self) -> None:
        native = build_manifest("native", {}, candidate_evaluations=123)
        expanded = build_manifest("expanded", {}, candidate_evaluations=123)
        self.assertEqual(len(native), 140)
        self.assertEqual(len(expanded), 210)
        self.assertEqual({row["status"] for row in expanded}, {UNKNOWN})
        self.assertTrue(
            all(row["status"] in ALLOWED_STATUSES for row in native + expanded)
        )
        self.assertNotIn("infeasible", {row["status"] for row in expanded})


if __name__ == "__main__":
    unittest.main()
