"""CPU regression tests for interaction protocol v2."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import yaml

# greedy.py transitively imports ``turtle``, which may lack tkinter on a
# headless login node. Match the existing rollout tests' import-only shim.
for _name in ("tkinter", "turtle"):
    if _name not in sys.modules:
        try:
            __import__(_name)
        except Exception:
            sys.modules[_name] = types.ModuleType(_name)
if not hasattr(sys.modules.get("turtle"), "width"):
    sys.modules["turtle"].width = None

from toolusepcg.config import Config, config_to_dict
from toolusepcg.edit_manager import EditManager
from toolusepcg.evaluator import EvalResult
from toolusepcg.level import Level
from toolusepcg.logging import TraceLogger
from toolusepcg.optimizer.greedy import GreedyOptimizer
from toolusepcg.protocol import (
    INTERACTION_PROTOCOL_ENV,
    effective_interaction_protocol,
)
from toolusepcg.tools.edit_place_line import PlaceLineTool
from toolusepcg.tools.edit_place_patch import PlacePatchTool
from toolusepcg.tools.edit_place_single import PlaceSingleTileTool
from toolusepcg.tools.edit_place_tile import PlaceTileTool
from toolusepcg.tools.edit_place_wall import PlaceWallSegmentTool


# Minimal vendored representation of all 96 glyph calls in the frozen G1
# base×champion/bin-test-02 trace. Step 1 is unique; steps 2-8 repeat the same
# twelve start/end calls byte-for-byte. The live trace is not read at runtime.
_STEP_1_GLYPH_CALLS = [
    ("d", 1, 1, "right", 14),
    ("d", 1, 14, "down", 14),
    ("d", 15, 14, "left", 14),
    ("d", 15, 1, "up", 13),
    ("d", 2, 2, "right", 12),
    ("d", 2, 14, "down", 12),
    ("d", 14, 14, "left", 12),
    ("d", 14, 2, "up", 11),
    ("d", 3, 3, "right", 10),
    ("d", 3, 13, "down", 10),
    ("d", 13, 13, "left", 10),
    ("d", 13, 3, "up", 91),
]
_REPEATED_GLYPH_CALLS = [
    ("e", y, 1, y, 14)
    for y in range(15, 3, -1)
]
_GOLDEN_GLYPH_CALLS = _STEP_1_GLYPH_CALLS + _REPEATED_GLYPH_CALLS * 7


def _wall_parameters(call: tuple) -> dict:
    mode, start_y, start_x, fourth, fifth = call
    parameters = {
        "start_y": start_y,
        "start_x": start_x,
        "tile_type": "#",
    }
    if mode == "d":
        parameters.update(
            mode="start_dir_len",
            direction=fourth,
            length=fifth,
        )
    else:
        parameters.update(
            mode="start_end",
            end_y=fourth,
            end_x=fifth,
        )
    return parameters


def _manager(problem_type: str = "binary") -> EditManager:
    level = Level(np.zeros((16, 16), dtype=np.int32), problem_type=problem_type)
    return EditManager(None, level, problem_type=problem_type)


class ProtocolSelectionTests(unittest.TestCase):
    def test_only_exact_v2_enables_v2(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(INTERACTION_PROTOCOL_ENV, None)
            self.assertEqual(effective_interaction_protocol(), "v1")
            for value in ("v1", "garbage", "V2", "", " v2"):
                with self.subTest(value=value):
                    os.environ[INTERACTION_PROTOCOL_ENV] = value
                    self.assertEqual(effective_interaction_protocol(), "v1")
            os.environ[INTERACTION_PROTOCOL_ENV] = "v2"
            self.assertEqual(effective_interaction_protocol(), "v2")


class TileVocabularyTests(unittest.TestCase):
    def test_v1_replays_all_96_glyph_noops_byte_identically(self) -> None:
        self.assertEqual(len(_GOLDEN_GLYPH_CALLS), 96)
        manager = _manager("binary")
        tool = PlaceWallSegmentTool(manager)
        original_level = manager.level.to_string()
        expected = {
            "success": True,
            "data": {
                "new_level": original_level,
                "applied": True,
                "num_tiles_changed": 0,
                "diff": [],
            },
            "errors": [],
        }
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(INTERACTION_PROTOCOL_ENV, None)
            for index, call in enumerate(_GOLDEN_GLYPH_CALLS):
                with self.subTest(index=index):
                    self.assertEqual(
                        tool.execute(**_wall_parameters(call)).to_dict(),
                        expected,
                    )
        self.assertEqual(manager.total_tiles_changed, 0)
        self.assertEqual(manager.level.to_string(), original_level)

    def test_v2_rejects_all_96_vendored_glyph_calls(self) -> None:
        manager = _manager("binary")
        tool = PlaceWallSegmentTool(manager)
        original_level = manager.level.to_string()
        expected = {
            "success": False,
            "data": {},
            "errors": ["Unknown tile_type '#'. Legal values: ['wall', 'empty']"],
        }
        with mock.patch.dict(os.environ, {INTERACTION_PROTOCOL_ENV: "v2"}):
            for index, call in enumerate(_GOLDEN_GLYPH_CALLS):
                with self.subTest(index=index):
                    self.assertEqual(
                        tool.execute(**_wall_parameters(call)).to_dict(),
                        expected,
                    )
        self.assertEqual(manager.total_tiles_changed, 0)
        self.assertEqual(manager.level.to_string(), original_level)

    def test_v2_unknown_tile_type_lists_legal_values_for_every_edit_tool(self) -> None:
        cases = [
            (
                PlaceWallSegmentTool,
                "binary",
                {
                    "mode": "start_end",
                    "start_y": 0,
                    "start_x": 0,
                    "end_y": 0,
                    "end_x": 1,
                    "tile_type": "#",
                },
                ["wall", "empty"],
            ),
            (PlaceTileTool, "zelda", {"mode": "single", "tile_type": "#", "y": 0, "x": 0},
             ["wall", "empty", "player", "key", "door", "enemy"]),
            (PlaceSingleTileTool, "zelda", {"tile_type": "#", "y": 0, "x": 0},
             ["wall", "empty", "player", "key", "door", "enemy"]),
            (PlaceLineTool, "zelda", {"tile_type": "#", "y": 0, "x": 0, "end_y": 0, "end_x": 1},
             ["wall", "empty", "player", "key", "door", "enemy"]),
            (PlacePatchTool, "zelda", {"tile_type": "#", "y": 0, "x": 0, "end_y": 1, "end_x": 1},
             ["wall", "empty", "player", "key", "door", "enemy"]),
        ]
        with mock.patch.dict(os.environ, {INTERACTION_PROTOCOL_ENV: "v2"}):
            for tool_class, problem_type, parameters, legal_values in cases:
                with self.subTest(tool=tool_class.__name__):
                    manager = _manager(problem_type)
                    before = manager.level.to_string()
                    result = tool_class(manager).execute(**parameters)
                    self.assertEqual(
                        result.to_dict(),
                        {
                            "success": False,
                            "data": {},
                            "errors": [
                                f"Unknown tile_type '#'. Legal values: {legal_values}"
                            ],
                        },
                    )
                    self.assertEqual(manager.total_tiles_changed, 0)
                    self.assertEqual(manager.level.to_string(), before)


class _MetricScorer:
    def __init__(self) -> None:
        self.calls = 0

    def score(self, result: EvalResult) -> float:
        self.calls += 1
        return float(result.metrics["score"])


def _acceptance_harness(*, incumbent_valid: bool, current_score: float):
    optimizer = GreedyOptimizer.__new__(GreedyOptimizer)
    optimizer.current_result = EvalResult(valid=incumbent_valid)
    optimizer.current_score = current_score
    optimizer.scorer = _MetricScorer()
    optimizer.config = SimpleNamespace(
        optimizer=SimpleNamespace(strategy="greedy")
    )
    optimizer.epsilon = 0.0
    optimizer.step_count = 0
    optimizer._last_acceptance_triage = None
    return optimizer


class ValidityAcceptanceTests(unittest.TestCase):
    def test_v2_validity_regression_rejects_without_scoring(self) -> None:
        optimizer = _acceptance_harness(incumbent_valid=True, current_score=-100.0)
        candidate = EvalResult(metrics={"score": 100.0}, valid=False)
        with mock.patch.dict(os.environ, {INTERACTION_PROTOCOL_ENV: "v2"}):
            accepted, reason = optimizer.decide_accept(candidate, num_changes=1)
        self.assertFalse(accepted)
        self.assertEqual(reason, "rejected_validity_regression")
        self.assertEqual(optimizer.scorer.calls, 0)

    def test_v2_invalid_to_invalid_remains_score_decided(self) -> None:
        optimizer = _acceptance_harness(incumbent_valid=False, current_score=1.0)
        candidate = EvalResult(metrics={"score": 2.0}, valid=False)
        with mock.patch.dict(os.environ, {INTERACTION_PROTOCOL_ENV: "v2"}):
            accepted, reason = optimizer.decide_accept(candidate, num_changes=1)
        self.assertTrue(accepted)
        self.assertEqual(reason, "improved (score: 1.00 -> 2.00)")
        self.assertEqual(optimizer.scorer.calls, 1)

    def test_v2_invalid_to_valid_remains_score_decided(self) -> None:
        optimizer = _acceptance_harness(incumbent_valid=False, current_score=2.0)
        candidate = EvalResult(metrics={"score": 1.0}, valid=True)
        with (
            mock.patch.dict(os.environ, {INTERACTION_PROTOCOL_ENV: "v2"}),
            mock.patch("toolusepcg.optimizer.greedy.random.random", return_value=1.0),
        ):
            accepted, reason = optimizer.decide_accept(candidate, num_changes=1)
        self.assertFalse(accepted)
        self.assertEqual(reason, "rejected (score: 2.00 -> 1.00)")
        self.assertEqual(optimizer.scorer.calls, 1)

    def test_v1_validity_regression_preserves_score_only_acceptance(self) -> None:
        optimizer = _acceptance_harness(incumbent_valid=True, current_score=1.0)
        candidate = EvalResult(metrics={"score": 2.0}, valid=False)
        with mock.patch.dict(os.environ, {INTERACTION_PROTOCOL_ENV: "garbage"}):
            accepted, reason = optimizer.decide_accept(candidate, num_changes=1)
        self.assertTrue(accepted)
        self.assertEqual(reason, "improved (score: 1.00 -> 2.00)")
        self.assertEqual(optimizer.scorer.calls, 1)


class ProtocolRecordingTests(unittest.TestCase):
    def test_effective_protocol_is_written_to_resolved_run_config(self) -> None:
        for raw_value, expected in ((None, "v1"), ("garbage", "v1"), ("v2", "v2")):
            with self.subTest(raw_value=raw_value), tempfile.TemporaryDirectory() as temp_dir:
                with mock.patch.dict(os.environ, {}, clear=False):
                    if raw_value is None:
                        os.environ.pop(INTERACTION_PROTOCOL_ENV, None)
                    else:
                        os.environ[INTERACTION_PROTOCOL_ENV] = raw_value
                    self.assertEqual(
                        config_to_dict(Config())["interaction_protocol"], expected
                    )
                    TraceLogger(Config(), run_dir=temp_dir)
                saved = yaml.safe_load((Path(temp_dir) / "config.yaml").read_text())
                self.assertEqual(saved["interaction_protocol"], expected)



if __name__ == "__main__":
    unittest.main()
