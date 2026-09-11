"""Tests for the env-gated coordinate-bounded decoding grammar (R033).

B8 (R024-R031) localized the dead-sokoban bottleneck to coordinate
out-of-bounds: the 35B designer emits ``place_tile`` at y=10/20/50 on an 8x8
grid, so ~86-91% of placements are rejected at the tool layer. Every
prompt-side lever (bounds text R028, error feedback R029/R030, visible ruler
R031) failed to move the OOB rate. R033 moves the intervention to the
DECODING layer: the designer already decodes under an xgrammar GBNF grammar
(``llm.grammar_domain``), whose tool-call ``parameters`` slot was
intentionally permissive (any JSON object). With
``RPE_PCG_COORD_GRAMMAR=on``, coordinate-bearing tools get an explicit
parameters rule whose y-axis params can only decode to 0..H-1 and x-axis
params to 0..W-1 — out-of-bounds coordinates become unsampleable, which
renormalizes the model's own coordinate distribution onto the grid.

Gate off (default) must be byte-identical to the baseline grammar.

The repo runs its suite with ``unittest`` (pytest is not installed in the
shared ``.venv``); the ``turtle`` stub mirrors the other rpe test files.
"""

from __future__ import annotations

import json
import os
import sys
import types
import unittest
from unittest import mock

if "turtle" not in sys.modules:  # pragma: no cover - environment shim
    _turtle_stub = types.ModuleType("turtle")
    _turtle_stub.width = 0
    sys.modules["turtle"] = _turtle_stub

from toolusepcg.protocol.grammar import (
    build_grammar_for_domain,
    build_step_json_schema,
    build_step_xgrammar,
)

try:  # xgrammar is present in the shared .venv (vLLM dependency)
    from xgrammar import Grammar as _XGrammar
    from xgrammar.testing import _is_grammar_accept_string as _xg_accept

    _HAVE_XGRAMMAR = True
except ImportError:  # pragma: no cover
    _HAVE_XGRAMMAR = False

ALL_DOMAINS = ["binary", "binarydoor", "zelda", "sokoban", "loderunner", "smb"]
DOMAIN_DIMS = {
    "binary": (16, 16),
    "binarydoor": (16, 16),
    "zelda": (16, 16),
    "sokoban": (8, 8),
    "loderunner": (11, 16),
    "smb": (16, 32),
}


def _place_tile_step(y: int, x: int, extra: dict | None = None) -> str:
    """A complete STEP message JSON with one place_tile call."""
    params: dict = {"mode": "single", "tile_type": "crate", "y": y, "x": x}
    if extra:
        params.update(extra)
    return json.dumps(
        {
            "type": "STEP",
            "rationale": "r",
            "plan": "p",
            "tool_calls": [{"tool_name": "place_tile", "parameters": params}],
            "acceptance_hint": "h",
        }
    )


class TestDefaultByteIdentity(unittest.TestCase):
    """coord_bounds=None (and the omitted kwarg) must be the baseline grammar."""

    def test_no_bounds_has_no_coord_rules_and_permissive_params(self) -> None:
        for domain in ALL_DOMAINS:
            with self.subTest(domain=domain):
                pack = build_grammar_for_domain(domain, max_tool_calls=20)
                self.assertIsNone(pack.coord_bounds)
                self.assertNotIn("coord_y", pack.xgrammar_str)
                self.assertNotIn("coord_x", pack.xgrammar_str)
                self.assertNotIn("params_", pack.xgrammar_str)
                self.assertNotIn("R033", pack.xgrammar_str)
                # parameters slot stays the permissive generic object
                self.assertIn(
                    '"\\"parameters\\"" ws ":" ws object ws "}"', pack.xgrammar_str
                )
                # JSON-schema side carries no injected bounds
                self.assertNotIn('"minimum"', json.dumps(pack.json_schema))

    def test_none_kwarg_identical_to_omitted(self) -> None:
        pack = build_grammar_for_domain("sokoban", max_tool_calls=20)
        specs = pack.tool_specs
        self.assertEqual(
            build_step_xgrammar(specs, max_tool_calls=20),
            build_step_xgrammar(specs, max_tool_calls=20, coord_bounds=None),
        )
        self.assertEqual(
            build_step_json_schema(specs, max_tool_calls=20),
            build_step_json_schema(specs, max_tool_calls=20, coord_bounds=None),
        )


class TestBoundedGrammar(unittest.TestCase):
    def test_sokoban_8x8_coord_rules(self) -> None:
        pack = build_grammar_for_domain("sokoban", max_tool_calls=20, coord_bounds=(8, 8))
        g = pack.xgrammar_str
        self.assertEqual(pack.coord_bounds, (8, 8))
        self.assertIn('coord_y ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7"', g)
        self.assertIn('coord_x ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7"', g)
        # place_tile branch now references the explicit params rule
        self.assertIn('"\\"parameters\\"" ws ":" ws params_place_tile ws "}"', g)
        # required params in declared order, coords locked, end coords locked too
        self.assertIn('"\\"y\\"" ws ":" ws coord_y', g)
        self.assertIn('"\\"x\\"" ws ":" ws coord_x', g)
        self.assertIn('(ws "," ws "\\"end_y\\"" ws ":" ws coord_y)?', g)
        self.assertIn('(ws "," ws "\\"end_x\\"" ws ":" ws coord_x)?', g)

    def test_loderunner_11x16_not_transposed(self) -> None:
        # y bounded by height (rows) -> 0..10; x bounded by width (cols) -> 0..15.
        g = build_grammar_for_domain(
            "loderunner", max_tool_calls=20, coord_bounds=(11, 16)
        ).xgrammar_str
        self.assertIn('| "10"\n'.replace("\n", ""), g.split("coord_y ::=")[1].split("\n")[0])
        coord_y_line = [l for l in g.splitlines() if l.startswith("coord_y ::=")][0]
        coord_x_line = [l for l in g.splitlines() if l.startswith("coord_x ::=")][0]
        self.assertIn('"10"', coord_y_line)
        self.assertNotIn('"11"', coord_y_line)
        self.assertIn('"15"', coord_x_line)
        self.assertNotIn('"16"', coord_x_line)

    def test_json_schema_side_gets_min_max(self) -> None:
        pack = build_grammar_for_domain("sokoban", max_tool_calls=20, coord_bounds=(8, 8))
        branch = pack.json_schema["properties"]["tool_calls"]["items"]["oneOf"][0]
        props = branch["properties"]["parameters"]["properties"]
        self.assertEqual((props["y"]["minimum"], props["y"]["maximum"]), (0, 7))
        self.assertEqual((props["x"]["minimum"], props["x"]["maximum"]), (0, 7))
        self.assertEqual((props["end_y"]["minimum"], props["end_y"]["maximum"]), (0, 7))

    def test_all_domains_build_with_bounds(self) -> None:
        for domain, dims in DOMAIN_DIMS.items():
            with self.subTest(domain=domain):
                pack = build_grammar_for_domain(domain, max_tool_calls=20, coord_bounds=dims)
                self.assertTrue(pack.xgrammar_str)

    def test_invalid_bounds_raise(self) -> None:
        pack = build_grammar_for_domain("sokoban", max_tool_calls=20)
        with self.assertRaises(ValueError):
            build_step_xgrammar(pack.tool_specs, max_tool_calls=20, coord_bounds=(0, 8))


@unittest.skipUnless(_HAVE_XGRAMMAR, "xgrammar not installed in this venv")
class TestXGrammarAcceptance(unittest.TestCase):
    """Compile with the real xgrammar engine and check sampleability."""

    def test_oob_unsampleable_in_bounds_sampleable(self) -> None:
        for domain, (h, w) in [("sokoban", (8, 8)), ("loderunner", (11, 16)), ("smb", (16, 32))]:
            with self.subTest(domain=domain):
                bounded = _XGrammar.from_ebnf(
                    build_grammar_for_domain(
                        domain, max_tool_calls=20, coord_bounds=(h, w)
                    ).xgrammar_str
                )
                baseline = _XGrammar.from_ebnf(
                    build_grammar_for_domain(domain, max_tool_calls=20).xgrammar_str
                )
                # in-bounds accepted (single and rect/end coords)
                self.assertTrue(_xg_accept(bounded, _place_tile_step(h - 1, w - 1)))
                self.assertTrue(
                    _xg_accept(
                        bounded,
                        _place_tile_step(1, 1, {"end_y": h - 1, "end_x": w - 1}),
                    )
                )
                # the B8 failure modes are unsampleable under the bounded grammar
                self.assertFalse(_xg_accept(bounded, _place_tile_step(20, 0)))
                self.assertFalse(_xg_accept(bounded, _place_tile_step(0, 50)))
                self.assertFalse(
                    _xg_accept(bounded, _place_tile_step(1, 1, {"end_y": 99, "end_x": 99}))
                )
                # ... while the baseline grammar accepts them (the exact delta)
                self.assertTrue(_xg_accept(baseline, _place_tile_step(20, 0)))

    def test_bounded_grammar_compiles_for_all_domains(self) -> None:
        for domain, dims in DOMAIN_DIMS.items():
            with self.subTest(domain=domain):
                _XGrammar.from_ebnf(
                    build_grammar_for_domain(
                        domain, max_tool_calls=20, coord_bounds=dims
                    ).xgrammar_str
                )


class TestGreedyEnvGate(unittest.TestCase):
    """RPE_PCG_COORD_GRAMMAR=on wires config.env dims into the decode grammar."""

    @classmethod
    def setUpClass(cls) -> None:
        from toolusepcg.config import load_config
        from toolusepcg.optimizer.greedy import GreedyOptimizer

        cfg = load_config(
            "configs/sokoban.yaml",
            overrides={
                "optimizer": {"init_strategy": "empty"},
                "llm": {"grammar_domain": "sokoban"},
            },
        )
        cls.opt = GreedyOptimizer(cfg, verbose=False)
        cls.opt.initialize(seed=0)

    def _schema_with_env(self, value: str | None) -> dict:
        self.opt._openai_structured_outputs = None  # reset the one-shot cache
        self.opt._openai_step_schema = None
        with mock.patch.dict(os.environ, {}, clear=False):
            if value is None:
                os.environ.pop("RPE_PCG_COORD_GRAMMAR", None)
            else:
                os.environ["RPE_PCG_COORD_GRAMMAR"] = value
            extra_body = self.opt._build_openai_extra_body()
        return extra_body["structured_outputs"]["json"]

    def test_gate_off_matches_baseline_grammar(self) -> None:
        baseline = build_grammar_for_domain("sokoban", max_tool_calls=20).json_schema
        self.assertEqual(self._schema_with_env(None), baseline)
        # any non-"on" value behaves exactly like unset
        for value in ("", "off", "ON", "explicit", "1"):
            with self.subTest(value=value):
                self.assertEqual(self._schema_with_env(value), baseline)

    def test_gate_on_uses_config_env_dims(self) -> None:
        schema = self._schema_with_env("on")
        expected = build_grammar_for_domain(
            "sokoban", max_tool_calls=20, coord_bounds=(8, 8)
        ).json_schema
        self.assertEqual(schema, expected)
        encoded = json.dumps(schema, sort_keys=True)
        self.assertIn('"maximum": 7', encoded)
        self.assertIn('"minimum": 0', encoded)


if __name__ == "__main__":
    unittest.main()
