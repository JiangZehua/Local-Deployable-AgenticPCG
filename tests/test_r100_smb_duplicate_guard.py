from __future__ import annotations

import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.r100_smb_duplicate_guard import audit
from toolusepcg.evaluator import EvalResult
from toolusepcg.level import Level
from toolusepcg.optimizer.greedy import GreedyOptimizer, _level_digest
from toolusepcg.protocol.json_schema import StepMessage


ROOT = Path(__file__).resolve().parent.parent
R099_ROOT = ROOT / "runs/r099_dynamic_pass4/temperature-0.7/array-16452061"


class _EditManager:
    def __init__(self, level: Level):
        self.level = level

    def set_level(self, level: Level) -> None:
        self.level = level.copy()


class _Scorer:
    def score(self, result: EvalResult) -> float:
        return float(result.metrics.get("value", 0))


class _Termination:
    def __init__(self) -> None:
        self.changes: list[int] = []

    def add_changes(self, value: int) -> None:
        self.changes.append(value)


def _optimizer(*, guard: bool) -> tuple[GreedyOptimizer, dict[str, int]]:
    current = Level(np.zeros((2, 2), dtype=np.int64), "binary")
    candidate = Level(np.ones((2, 2), dtype=np.int64), "binary")
    current_result = EvalResult(metrics={"value": 0}, valid=True)
    candidate_result = EvalResult(metrics={"value": 1}, valid=True)
    calls = {"evaluator": 0, "llm": 0}

    opt = GreedyOptimizer.__new__(GreedyOptimizer)
    opt.edit_manager = _EditManager(current)
    opt.current_result = current_result
    opt.current_score = 0.0
    opt.scorer = _Scorer()

    def evaluate(level: Level) -> EvalResult:
        calls["evaluator"] += 1
        return copy.deepcopy(candidate_result)

    opt.evaluator = SimpleNamespace(evaluate=evaluate)
    opt.termination = _Termination()
    opt.config = SimpleNamespace(optimizer=SimpleNamespace(strategy="greedy"))
    opt.problem_type = "binary"
    opt.stable_change_accounting = True
    opt.ablation_mode = "full"
    opt.epsilon = 0.0
    opt.step_count = 0
    opt.step_records = []
    opt._conversation_history = []
    opt._last_acceptance_triage = None
    opt._last_reasoning_content = ""
    opt.logger = None
    opt.verbose = False
    opt.debug = False
    opt._debug_file = None
    opt.duplicate_level_guard = guard
    opt._evaluated_level_cache = {
        _level_digest(candidate): (4, copy.deepcopy(candidate_result))
    }
    opt.build_prompt = lambda: "prompt"
    opt._capture_grpo_request_messages = lambda record, prompt: None

    def call_llm(prompt: str):
        calls["llm"] += 1
        return "{}", False, 1, 1

    opt.call_llm = call_llm
    opt.parse_response = lambda raw, truncated=False: (StepMessage(), [])
    opt.execute_step = lambda message: (candidate.copy(), [], 4)
    opt._build_environment_step_feedback = lambda level, evaluation, tools: "cached mechanic feedback"
    return opt, calls


class TestR100DuplicateGuard(unittest.TestCase):

    def test_online_guard_skips_evaluator_and_emits_explicit_feedback(self) -> None:
        optimizer, calls = _optimizer(guard=True)
        record = optimizer.step()
        self.assertEqual(calls, {"evaluator": 0, "llm": 1})
        self.assertTrue(record.duplicate_level)
        self.assertTrue(record.evaluator_skipped)
        self.assertEqual(record.duplicate_first_evaluated_step, 4)
        self.assertFalse(record.accepted)
        self.assertIn("duplicate_level_guard_rejected", record.accept_reason)
        self.assertIn("DUPLICATE_LEVEL_GUARD", record.environment_feedback_after)
        self.assertEqual(optimizer.step_count, 1)
        self.assertEqual(optimizer.termination.changes, [])

    def test_default_off_still_executes_evaluator(self) -> None:
        optimizer, calls = _optimizer(guard=False)
        record = optimizer.step()
        self.assertEqual(calls["evaluator"], 1)
        self.assertFalse(record.duplicate_level)
        self.assertFalse(record.evaluator_skipped)


if __name__ == "__main__":
    unittest.main()
