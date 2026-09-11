"""Tests for the clean always-accept controller ablation."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from toolusepcg.optimizer.greedy import GreedyOptimizer


class AlwaysAcceptPolicyTests(unittest.TestCase):
    def _optimizer(self) -> GreedyOptimizer:
        optimizer = GreedyOptimizer.__new__(GreedyOptimizer)
        optimizer.acceptance_policy = "always_accept"
        optimizer._last_acceptance_triage = None
        return optimizer

    def test_accepts_invalid_non_noop_before_hard_validity_gate(self) -> None:
        optimizer = self._optimizer()
        accepted, reason = optimizer.decide_accept(
            SimpleNamespace(valid=False, problem_type="zelda"), 12
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, "always_accept_non_noop")

    def test_still_rejects_noop(self) -> None:
        optimizer = self._optimizer()
        accepted, reason = optimizer.decide_accept(
            SimpleNamespace(valid=False, problem_type="zelda"), 0
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "always_accept_no_changes")


if __name__ == "__main__":
    unittest.main()
