#!/usr/bin/env python3
"""Coordinate-bounded STEP schemas and grammars for the inference harness."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Grammar pack
# ---------------------------------------------------------------------------


@dataclass
class GrammarPack:
    """Bundle of equivalent schema representations for the STEP message."""

    tool_specs: list[dict] = field(default_factory=list)
    max_tool_calls: int = 20
    json_schema: dict = field(default_factory=dict)
    xgrammar_str: str = ""
    # (height, width) the coordinate params were bounded to, or None when the
    # grammar is the unconstrained baseline (R033 coordinate-grammar arm).
    coord_bounds: tuple[int, int] | None = None


# ---------------------------------------------------------------------------
# Coordinate-bounded decoding (R033)
# ---------------------------------------------------------------------------

# Integer tool parameters that address grid rows (y axis, bounded by height)
# vs columns (x axis, bounded by width). Covers every edit tool in
# toolusepcg/tools (place_tile, place_line, place_patch, place_single,
# place_wall); generate_* tools carry no coordinate parameters.
_COORD_Y_PARAMS = frozenset({"y", "end_y", "start_y"})
_COORD_X_PARAMS = frozenset({"x", "end_x", "start_x"})


def _coord_axis_for_param(name: str, schema: dict) -> str | None:
    """Return 'y'/'x' if this parameter is a grid coordinate, else None."""
    if schema.get("type") != "integer":
        return None
    if name in _COORD_Y_PARAMS:
        return "y"
    if name in _COORD_X_PARAMS:
        return "x"
    return None


def _spec_coord_params(spec: dict) -> dict[str, str]:
    """Map of param name -> axis ('y'/'x') for a tool spec's coordinate params."""
    props = (spec.get("parameters") or {}).get("properties") or {}
    out: dict[str, str] = {}
    for name, schema in props.items():
        axis = _coord_axis_for_param(name, schema if isinstance(schema, dict) else {})
        if axis:
            out[name] = axis
    return out


# ---------------------------------------------------------------------------
# JSON schema builder
# ---------------------------------------------------------------------------


def _enum_tool_name_schema(tool_names: list[str]) -> dict:
    return {"type": "string", "enum": list(tool_names)}


def _bounded_parameters_schema(spec: dict, coord_bounds: tuple[int, int]) -> dict:
    """Deep-copied parameters_schema with min/max stamped on coordinate params."""
    import copy

    height, width = coord_bounds
    params = copy.deepcopy(spec.get("parameters", {"type": "object"}))
    props = params.get("properties") or {}
    for name, schema in props.items():
        axis = _coord_axis_for_param(name, schema if isinstance(schema, dict) else {})
        if axis == "y":
            schema["minimum"], schema["maximum"] = 0, height - 1
        elif axis == "x":
            schema["minimum"], schema["maximum"] = 0, width - 1
    return params


def _build_tool_call_schema(
    tool_specs: list[dict],
    coord_bounds: tuple[int, int] | None = None,
) -> dict:
    """OneOf over per-tool schemas so the grammar can constrain parameters
    to those declared by the chosen tool.

    Each branch has shape::

        {
          "type": "object",
          "properties": {
            "tool_name": {"const": "<name>"},
            "parameters": <tool's parameters_schema>
          },
          "required": ["tool_name", "parameters"],
          "additionalProperties": false
        }

    When ``coord_bounds=(height, width)`` is given, integer coordinate
    parameters get ``minimum``/``maximum`` stamped in (R033).
    """
    branches: list[dict] = []
    for spec in tool_specs:
        if coord_bounds is not None:
            parameters = _bounded_parameters_schema(spec, coord_bounds)
        else:
            parameters = spec.get("parameters", {"type": "object"})
        branches.append(
            {
                "type": "object",
                "properties": {
                    "tool_name": {"const": spec["name"]},
                    "parameters": parameters,
                },
                "required": ["tool_name", "parameters"],
                "additionalProperties": False,
            }
        )
    if not branches:
        # degenerate fallback: permit any object
        return {"type": "object"}
    return {"oneOf": branches}


def build_step_json_schema(
    tool_specs: list[dict],
    *,
    max_tool_calls: int = 20,
    acceptance_hint_enum: list[str] | None = None,
    coord_bounds: tuple[int, int] | None = None,
) -> dict:
    """Build the top-level JSON schema for a STEP message."""
    tool_call_schema = _build_tool_call_schema(tool_specs, coord_bounds=coord_bounds)
    hint_schema: dict = {"type": "string"}
    if acceptance_hint_enum:
        hint_schema = {"type": "string", "enum": list(acceptance_hint_enum)}

    return {
        "type": "object",
        "properties": {
            "type": {"const": "STEP"},
            "rationale": {"type": "string"},
            "plan": {"type": "string"},
            "tool_calls": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_tool_calls,
                "items": tool_call_schema,
            },
            "acceptance_hint": hint_schema,
        },
        "required": ["type", "rationale", "plan", "tool_calls", "acceptance_hint"],
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# xgrammar GBNF builder
# ---------------------------------------------------------------------------


def _xgbnf_escape(s: str) -> str:
    """Escape a string for inclusion inside a GBNF double-quoted literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _xgbnf_for_param_schema(schema: dict) -> str:
    """Produce a GBNF fragment for a single parameter's value slot.

    xgrammar already ships a ``json`` primitive rule that matches any valid
    JSON value; for parameter *value* nesting we fall back to that. The
    top-level STEP structure and the tool-call object keys are spelled out
    explicitly below, so grammar-mask entropy is concentrated at the
    places where it actually matters (message type, tool name enum,
    acceptance_hint enum).
    """
    # Intentionally permissive at value level — parameters_schema is
    # validated post-hoc by json_schema.validate_message anyway.
    return "json"


def _xgbnf_tool_call_branch(spec: dict, params_rule: str | None = None) -> str:
    """GBNF alternative for one tool-call branch.

    Emits the literal ``"tool_name":"<name>"`` guard, then allows any JSON
    object for parameters (post-validated server-side) — or, when
    ``params_rule`` is given, that explicit per-tool parameters rule
    (R033 coordinate-bounded decoding).
    """
    name_lit = _xgbnf_escape(spec["name"])
    return (
        '"{" ws "\\"tool_name\\"" ws ":" ws "\\"'
        + name_lit
        + '\\"" ws "," ws "\\"parameters\\"" ws ":" ws '
        + (params_rule if params_rule is not None else "object")
        + ' ws "}"'
    )


def _xgbnf_bounded_int_rule(name: str, max_inclusive: int) -> str:
    """GBNF rule matching the decimal integers 0..max_inclusive."""
    alts = " | ".join(f'"{i}"' for i in range(max_inclusive + 1))
    return f"{name} ::= {alts}"


def _xgbnf_value_rule(name: str, schema: dict, coord_axes: dict[str, str]) -> str:
    """GBNF value slot for one parameter under coordinate-bounded decoding.

    Coordinate params get the bounded ``coord_y``/``coord_x`` rules; all other
    params stay maximally permissive for their declared type so the only real
    delta vs the baseline grammar is the coordinate range (plus the explicit
    key layout the rule needs in order to know which value slot is which).
    """
    axis = coord_axes.get(name)
    if axis == "y":
        return "coord_y"
    if axis == "x":
        return "coord_x"
    ptype = schema.get("type")
    if ptype == "string":
        return "string"
    if ptype in ("integer", "number"):
        return "number"
    if ptype == "boolean":
        return '("true" | "false")'
    return "json"


def _xgbnf_params_rule(spec: dict, rule_name: str) -> str | None:
    """Explicit GBNF rule for one coordinate-bearing tool's parameters object.

    Emits required params first (in declared order, comma-joined), then each
    optional param as an optional ``(ws "," ws <member>)?`` group in declared
    order. Returns None when the layout cannot be expressed this way (no
    required params) — the caller then falls back to the permissive object.
    """
    params = spec.get("parameters") or {}
    props = params.get("properties") or {}
    required = [k for k in props if k in set(params.get("required") or [])]
    optional = [k for k in props if k not in set(params.get("required") or [])]
    if not required:
        return None
    coord_axes = _spec_coord_params(spec)

    def member(key: str) -> str:
        value = _xgbnf_value_rule(key, props[key] if isinstance(props[key], dict) else {}, coord_axes)
        return f'"\\"{_xgbnf_escape(key)}\\"" ws ":" ws {value}'

    body = ' ws "," ws '.join(member(k) for k in required)
    for key in optional:
        body += f' (ws "," ws {member(key)})?'
    return f'{rule_name} ::= "{{" ws {body} ws "}}"'


def _params_block(rules: list[str]) -> str:
    """Grammar text block for the R033 per-tool parameter rules ('' when off)."""
    if not rules:
        return ""
    return (
        "\n# --- R033: coordinate-bounded tool parameters --------------------------\n"
        + "\n".join(rules)
        + "\n"
    )


def build_step_xgrammar(
    tool_specs: list[dict],
    *,
    max_tool_calls: int = 20,
    acceptance_hint_enum: list[str] | None = None,
    coord_bounds: tuple[int, int] | None = None,
) -> str:
    """Build a GBNF grammar string for the STEP message.

    The ``root`` production matches the whole STEP JSON object. The tool
    call alternatives enforce one branch per registered tool name, so the
    model cannot emit an unknown tool. Parameter bodies fall back to the
    generic ``object`` production from the JSON primitive block at the
    bottom of the grammar.

    ``coord_bounds=(height, width)`` (R033) additionally locks every
    coordinate-bearing tool's parameters object to an explicit rule whose
    y-axis params can only decode to 0..height-1 and x-axis params to
    0..width-1 — out-of-bounds coordinates become unsampleable, which
    renormalizes the model's coordinate distribution onto the grid.
    ``None`` (default) produces the baseline grammar byte-identically.
    """
    if not tool_specs:
        raise ValueError("tool_specs is empty — cannot build grammar")

    params_rules: list[str] = []
    params_rule_names: dict[str, str] = {}
    if coord_bounds is not None:
        height, width = int(coord_bounds[0]), int(coord_bounds[1])
        if height < 1 or width < 1:
            raise ValueError(f"coord_bounds must be positive, got {coord_bounds}")
        for spec in tool_specs:
            if not _spec_coord_params(spec):
                continue
            rule_name = "params_" + re.sub(r"[^a-zA-Z0-9_]", "_", spec["name"])
            rule = _xgbnf_params_rule(spec, rule_name)
            if rule is None:
                continue
            params_rule_names[spec["name"]] = rule_name
            params_rules.append(rule)
        params_rules.append(_xgbnf_bounded_int_rule("coord_y", height - 1))
        params_rules.append(_xgbnf_bounded_int_rule("coord_x", width - 1))

    branches = "\n  | ".join(
        _xgbnf_tool_call_branch(s, params_rule_names.get(s["name"])) for s in tool_specs
    )

    # acceptance_hint: enum if provided, else any string
    if acceptance_hint_enum:
        hint_alts = " | ".join(
            f'"\\"{_xgbnf_escape(v)}\\""' for v in acceptance_hint_enum
        )
    else:
        hint_alts = "string"

    # Quantifier on tool_calls: ensures 1..max_tool_calls entries.
    # We emit a bounded repetition as an explicit nested rule.
    mx = int(max_tool_calls)
    tc_repeat_lines = []
    # Pattern: "[" ws tc (ws "," ws tc){0, mx-1} ws "]"
    tc_repeat = (
        '"[" ws tool_call'
        + "".join([f" (ws \",\" ws tool_call)?" for _ in range(mx - 1)])
        + ' ws "]"'
    )
    tc_repeat_lines.append(tc_repeat)
    tool_calls_rule = " | ".join(tc_repeat_lines)

    gbnf = rf"""# STEP message GBNF for Agentic-PCG tool calling.
# Generated by toolusepcg/protocol/grammar.py — do not edit by hand.
root ::= ws "{{" ws
  "\"type\"" ws ":" ws "\"STEP\"" ws "," ws
  "\"rationale\"" ws ":" ws string ws "," ws
  "\"plan\"" ws ":" ws string ws "," ws
  "\"tool_calls\"" ws ":" ws tool_calls ws "," ws
  "\"acceptance_hint\"" ws ":" ws ({hint_alts}) ws
  "}}" ws

tool_calls ::= {tool_calls_rule}

tool_call ::= {branches}
{_params_block(params_rules)}
# --- JSON primitive rules (subset) -------------------------------------
ws ::= [ \t\n\r]*
string ::= "\"" char* "\""
char ::= [^"\\] | "\\" ["\\/bfnrt] | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
number ::= "-"? (onenine digits | "0") ("." digits)? ([eE] [+-]? digits)?
onenine ::= [1-9]
digits ::= [0-9]+
object ::= "{{" ws (member (ws "," ws member)*)? ws "}}"
member ::= string ws ":" ws json
array ::= "[" ws (json (ws "," ws json)*)? ws "]"
json ::= string | number | object | array | "true" | "false" | "null"
"""
    return gbnf


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_grammar_for_domain(
    domain: str,
    *,
    max_tool_calls: int = 20,
    acceptance_hint_enum: list[str] | None = None,
    coord_bounds: tuple[int, int] | None = None,
) -> "GrammarPack":
    """Build a grammar pack for one PCG-Bench domain.

    Uses :class:`toolusepcg.protocol.prompt_builder.HarnessPromptBuilder` internally so
    the grammar matches the tool registry the SFT / RL harness will
    actually use at rollout time — same inventory, same tool name strings.
    """
    from toolusepcg.protocol.prompt_builder import HarnessPromptBuilder

    builder = HarnessPromptBuilder.for_domain(domain)
    return build_step_grammar_pack(
        builder._opt.registry,
        max_tool_calls=max_tool_calls,
        acceptance_hint_enum=acceptance_hint_enum,
        coord_bounds=coord_bounds,
    )


def build_step_grammar_pack(
    registry: Any,
    *,
    max_tool_calls: int = 20,
    acceptance_hint_enum: list[str] | None = None,
    coord_bounds: tuple[int, int] | None = None,
) -> GrammarPack:
    """Build both JSON-schema and xgrammar representations in one call.

    The passed ``registry`` is a ``toolusepcg.tools.ToolRegistry`` instance.
    We read its tool specs via ``registry.get_all_specs()`` and feed them
    to both builders.
    """
    tool_specs = registry.get_all_specs()
    if not tool_specs:
        raise ValueError("registry has no registered tools")
    # Sort for determinism so that grammar string is hashable / cacheable.
    tool_specs = sorted(tool_specs, key=lambda s: s["name"])

    return GrammarPack(
        tool_specs=tool_specs,
        max_tool_calls=max_tool_calls,
        json_schema=build_step_json_schema(
            tool_specs,
            max_tool_calls=max_tool_calls,
            acceptance_hint_enum=acceptance_hint_enum,
            coord_bounds=coord_bounds,
        ),
        xgrammar_str=build_step_xgrammar(
            tool_specs,
            max_tool_calls=max_tool_calls,
            acceptance_hint_enum=acceptance_hint_enum,
            coord_bounds=coord_bounds,
        ),
        coord_bounds=coord_bounds,
    )


# ---------------------------------------------------------------------------
# Smoke test CLI
# ---------------------------------------------------------------------------


def _smoke() -> int:
    """CLI smoke test: build the grammar for every domain, print summary."""
    import argparse
    import os
    import sys
    from pathlib import Path

    ap = argparse.ArgumentParser(description="grammar builder smoke test")
    ap.add_argument(
        "--domains",
        nargs="+",
        default=["binary", "binarydoor", "zelda", "sokoban", "loderunner", "smb"],
    )
    ap.add_argument("--out-dir", type=Path, default=Path("analysis/grammars"))
    args = ap.parse_args()

    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, int, int, int]] = []
    for d in args.domains:
        pack = build_grammar_for_domain(d, max_tool_calls=20)
        schema_path = args.out_dir / f"step_{d}.schema.json"
        gbnf_path = args.out_dir / f"step_{d}.gbnf"
        schema_path.write_text(json.dumps(pack.json_schema, indent=2))
        gbnf_path.write_text(pack.xgrammar_str)
        rows.append(
            (
                d,
                len(pack.tool_specs),
                len(pack.xgrammar_str),
                len(json.dumps(pack.json_schema)),
            )
        )
        print(
            f"  [{d}] n_tools={len(pack.tool_specs)} "
            f"gbnf_chars={len(pack.xgrammar_str)} schema_chars={len(json.dumps(pack.json_schema))}"
        )

    print()
    print("Wrote grammars to", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke())
