"""R108-A domain-routed SMB functional resolver and language front end.

Known structured targets bypass the LLM.  Natural language is used only to
produce a formal :class:`FunctionalSpec`, after which the same deterministic
resolver performs archive lookup, constructive compilation, or bounded
domain-specific search and verifies the resulting level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from experiments.r061_qwen38_m1 import _write_json
from experiments.r083_smb_atlas import ArchiveEntry, Candidate, _trajectory_coin_sites
from experiments.r083_smb_atlas_expansion import target_conditioned_candidates
from experiments.r083_smb_contract import build_evaluator
from experiments.r100_smb_corridor_stomp_atlas import (
    coin_overlay_candidates as stomp_coin_candidates,
    structural_candidates as stomp_structural_candidates,
)
from experiments.r101_smb_jump_episode_atlas import (
    coin_candidates as jump_coin_candidates,
    jump_episode_level,
)
from experiments.r107_smb_mechanic_library import (
    BACKBONES,
    LIBRARY_SHA256,
    load_backbone,
)
from toolusepcg.config import load_config
from toolusepcg.evaluator import EvalResult, StatsEvaluator
from toolusepcg.factory import LevelFactory
from toolusepcg.functional_resolver import (
    FunctionalCandidate,
    FunctionalObjective,
    FunctionalResolver,
    FunctionalSpec,
    MetricConstraint,
    SearchOutcome,
    select_satisfying_target,
)
from toolusepcg.level import Level, SMBTiles
from toolusepcg.smb_contract import get_smb_target_envelope
from toolusepcg.smb_static_metrics import (
    STATIC_TARGET_METRICS,
    build_static_support_scaffold,
)


RUN_ID = "R108-A-FUNCTIONAL-RESOLVER-LANGUAGE"
SEED = 108100
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LANGUAGE_MANIFEST = (
    REPO_ROOT
    / "benchmarks/smb/language.json"
)
INITIAL_ATLAS_PATH = REPO_ROOT / "benchmarks/smb/search_seeds/atlas.json"
DYNAMIC_NAMES = ("enemies", "coins", "jumps")
STATIC_NAMES = tuple(STATIC_TARGET_METRICS)
ALL_LANGUAGE_METRICS = DYNAMIC_NAMES + STATIC_NAMES
STATIC_ENVELOPE = {
    "pipe_count": (0, 4),
    "enemy_tile_count": (0, 6),
    "coin_tile_count": (0, 6),
    "block_tile_count": (0, 32),
}


def _functional_exact(result: EvalResult, target: Mapping[str, int]) -> bool:
    return bool(
        result.valid
        and float(result.metrics.get("complete", 1.0)) == 1.0
        and all(int(result.metrics.get(name, -1)) == int(value) for name, value in target.items())
    )


def _library_level(target: Mapping[str, int]) -> Level:
    backbone_id = f"k{int(target['enemies'])}_j{int(target['jumps'])}"
    level = load_backbone(backbone_id)
    profile = BACKBONES[backbone_id]
    for slot_id in range(int(target["coins"])):
        x, y = profile["slots"][slot_id]
        if int(level.data[y, x]) != int(SMBTiles.EMPTY):
            raise RuntimeError(f"library coin slot drift: {backbone_id}:{slot_id}")
        level.data[y, x] = int(SMBTiles.COIN)
    return level


class SmbDynamicFunctionalAdapter:
    domain = "smb_dynamic"
    target_names = DYNAMIC_NAMES

    def __init__(self, *, disabled_backbones: Iterable[str] = ()) -> None:
        self.disabled_backbones = frozenset(disabled_backbones)
        self._evaluator, _ = build_evaluator()
        self._discovered: dict[tuple[int, int, int], FunctionalCandidate] = {}
        self._envelope = get_smb_target_envelope("expanded")

    def validate(self, spec: FunctionalSpec) -> list[str]:
        errors: list[str] = []
        allowed = set(self.target_names)
        if set(spec.constraints) - allowed:
            errors.append(f"dynamic SMB received foreign metrics: {sorted(set(spec.constraints)-allowed)}")
        if not set(spec.constraints):
            errors.append("dynamic SMB requires at least one gameplay constraint")
        for name, constraint in spec.constraints.items():
            if name not in self._envelope:
                continue
            lower, upper = self._envelope[name]
            if not lower <= constraint.value <= upper:
                errors.append(f"{name}={constraint.value} outside expanded envelope [{lower},{upper}]")
        for objective in spec.objectives:
            if objective.metric not in allowed:
                errors.append(f"dynamic SMB objective uses foreign metric: {objective.metric}")
        return errors

    @staticmethod
    def _catalog_targets(backbone_ids: Iterable[str]) -> list[dict[str, int]]:
        return [
            {"enemies": int(BACKBONES[key]["behavior"][0]), "coins": coins,
             "jumps": int(BACKBONES[key]["behavior"][1])}
            for key in backbone_ids
            for coins in range(6)
        ]

    def lookup(self, spec: FunctionalSpec) -> FunctionalCandidate | None:
        discovered_target = select_satisfying_target(
            spec,
            ({name: key[index] for index, name in enumerate(self.target_names)}
             for key in self._discovered),
            stable_metric_order=self.target_names,
        )
        if discovered_target is not None:
            key = tuple(discovered_target[name] for name in self.target_names)
            return self._discovered[key]

        available = [key for key in BACKBONES if key not in self.disabled_backbones]
        target = select_satisfying_target(
            spec, self._catalog_targets(available), stable_metric_order=self.target_names
        )
        if target is None:
            return None
        backbone_id = f"k{target['enemies']}_j{target['jumps']}"
        return FunctionalCandidate(
            level=_library_level(target),
            selected_target=target,
            strategy="joint_backbone_plus_trajectory_coin_prefix",
            provenance={
                "library_sha256": LIBRARY_SHA256,
                "backbone_id": backbone_id,
                "dev_derived": True,
                "target_specific_level_lookup": False,
            },
        )

    def evaluate(self, level: Level) -> EvalResult:
        return self._evaluator.evaluate(level)

    def result_matches(self, result: EvalResult, selected_target: Mapping[str, int]) -> bool:
        return _functional_exact(result, selected_target)

    @staticmethod
    def _exact_search_target(spec: FunctionalSpec) -> dict[str, int] | None:
        return spec.exact_target(DYNAMIC_NAMES)

    def _cache(
        self,
        level: Level,
        target: dict[str, int],
        strategy: str,
        provenance: dict[str, Any],
    ) -> FunctionalCandidate:
        candidate = FunctionalCandidate(level.copy(), dict(target), strategy, provenance)
        key = tuple(target[name] for name in self.target_names)
        self._discovered[key] = candidate
        return candidate

    def search(
        self,
        spec: FunctionalSpec,
        evaluate: Callable[[Level], EvalResult],
        remaining_calls: Callable[[], int],
    ) -> SearchOutcome:
        target = self._exact_search_target(spec)
        if target is None:
            return SearchOutcome(
                None, None, "exact_target_required_for_discovery",
                {"reason": "range/objective queries search certified archive only"},
            )
        kills, coins, jumps = (target[name] for name in DYNAMIC_NAMES)
        if remaining_calls() <= 0:
            return SearchOutcome(None, None, "budget_exhausted", {"budget_exhausted": True})

        if kills == 0 and jumps in range(1, 6):
            return self._search_jump(target, evaluate, remaining_calls)
        if jumps == 0 and kills in range(1, 4):
            return self._search_stomp(target, evaluate, remaining_calls)
        return self._search_joint_atlas(target, evaluate, remaining_calls)

    def _search_jump(self, target, evaluate, remaining_calls) -> SearchOutcome:
        calls_before = remaining_calls()
        base = jump_episode_level(target["jumps"])
        base_result = evaluate(base)
        if target["coins"] == 0 and self.result_matches(base_result, target):
            candidate = self._cache(base, target, "jump_episode_search", {"family": "jump_episode"})
            return SearchOutcome(candidate, base_result, candidate.strategy, {"route": "jump_episode"})
        searched = 0
        for raw in jump_coin_candidates(base, base_result, target["coins"], target["jumps"]):
            if remaining_calls() <= 0:
                break
            searched += 1
            result = evaluate(raw.level)
            if self.result_matches(result, target):
                candidate = self._cache(
                    raw.level, target, "jump_episode_search",
                    {"family": raw.family, "parameters": raw.parameters},
                )
                return SearchOutcome(
                    candidate, result, candidate.strategy,
                    {"route": "jump_episode", "overlay_candidates": searched,
                     "evaluator_calls_used": calls_before - remaining_calls()},
                )
        return SearchOutcome(
            None, None, "jump_episode_search",
            {"route": "jump_episode", "overlay_candidates": searched,
             "budget_exhausted": remaining_calls() == 0},
        )

    def _search_stomp(self, target, evaluate, remaining_calls) -> SearchOutcome:
        calls_before = remaining_calls()
        structural = overlays = 0
        for base in stomp_structural_candidates(target["enemies"]):
            if remaining_calls() <= 0:
                break
            structural += 1
            base_result = evaluate(base.level)
            base_target = {"enemies": target["enemies"], "coins": 0, "jumps": 0}
            if not self.result_matches(base_result, base_target):
                continue
            if target["coins"] == 0:
                candidate = self._cache(
                    base.level, target, "stomp_corridor_search",
                    {"family": base.family, "parameters": base.parameters},
                )
                return SearchOutcome(candidate, base_result, candidate.strategy, {"route": "stomp_corridor"})
            for raw in stomp_coin_candidates(base, base_result, target["coins"]):
                if remaining_calls() <= 0:
                    break
                overlays += 1
                result = evaluate(raw.level)
                if self.result_matches(result, target):
                    candidate = self._cache(
                        raw.level, target, "stomp_corridor_search",
                        {"family": raw.family, "parameters": raw.parameters},
                    )
                    return SearchOutcome(
                        candidate, result, candidate.strategy,
                        {"route": "stomp_corridor", "structural_candidates": structural,
                         "overlay_candidates": overlays,
                         "evaluator_calls_used": calls_before - remaining_calls()},
                    )
        return SearchOutcome(
            None, None, "stomp_corridor_search",
            {"route": "stomp_corridor", "structural_candidates": structural,
             "overlay_candidates": overlays, "budget_exhausted": remaining_calls() == 0},
        )

    def _search_joint_atlas(self, target, evaluate, remaining_calls) -> SearchOutcome:
        structural = overlays = 0
        for raw in target_conditioned_candidates(INITIAL_ATLAS_PATH):
            if remaining_calls() <= 0:
                break
            structural += 1
            result = evaluate(raw.level)
            base_target = {"enemies": target["enemies"], "coins": 0, "jumps": target["jumps"]}
            if not self.result_matches(result, base_target):
                continue
            if target["coins"] == 0:
                candidate = self._cache(
                    raw.level, target, "joint_atlas_search",
                    {"family": raw.family, "parameters": raw.parameters},
                )
                return SearchOutcome(candidate, result, candidate.strategy, {"route": "joint_atlas"})
            entry = ArchiveEntry(
                behavior=(target["enemies"], 0, target["jumps"]), level=raw.level,
                family=raw.family, parameters=raw.parameters, primitive_cost=0,
                level_sha256=hashlib.sha256(raw.level.to_string().encode()).hexdigest(),
                result=result,
            )
            sites = _trajectory_coin_sites(entry)
            if len(sites) < target["coins"]:
                continue
            data = raw.level.data.copy()
            for x, y in sites[: target["coins"]]:
                data[y, x] = int(SMBTiles.COIN)
            if remaining_calls() <= 0:
                break
            overlays += 1
            level = Level(data, "smb")
            overlay_result = evaluate(level)
            if self.result_matches(overlay_result, target):
                candidate = self._cache(
                    level, target, "joint_atlas_search",
                    {"family": raw.family, "parameters": raw.parameters,
                     "coin_rule": "trajectory_prefix"},
                )
                return SearchOutcome(
                    candidate, overlay_result, candidate.strategy,
                    {"route": "joint_atlas", "structural_candidates": structural,
                     "overlay_candidates": overlays},
                )
        return SearchOutcome(
            None, None, "joint_atlas_search",
            {"route": "joint_atlas", "structural_candidates": structural,
             "overlay_candidates": overlays, "budget_exhausted": remaining_calls() == 0,
             "unknown_is_not_infeasible": True},
        )


class SmbStaticFunctionalAdapter:
    domain = "smb_static"
    target_names = STATIC_NAMES

    def __init__(self) -> None:
        self._config = load_config("configs/smb_static.yaml")
        self._factory = LevelFactory(self._config)
        self._evaluator = StatsEvaluator(self._config, self._factory.env)

    def validate(self, spec: FunctionalSpec) -> list[str]:
        errors: list[str] = []
        allowed = set(self.target_names)
        if set(spec.constraints) - allowed:
            errors.append(f"static SMB received foreign metrics: {sorted(set(spec.constraints)-allowed)}")
        for name, constraint in spec.constraints.items():
            if name not in STATIC_ENVELOPE:
                continue
            lower, upper = STATIC_ENVELOPE[name]
            if not lower <= constraint.value <= upper:
                errors.append(f"{name}={constraint.value} outside static envelope [{lower},{upper}]")
        for objective in spec.objectives:
            if objective.metric not in allowed:
                errors.append(f"static SMB objective uses foreign metric: {objective.metric}")
        return errors

    @staticmethod
    def _targets() -> Iterable[dict[str, int]]:
        for pipes in range(5):
            for enemies in range(7):
                for coins in range(7):
                    for blocks in range(33):
                        yield {
                            "pipe_count": pipes, "enemy_tile_count": enemies,
                            "coin_tile_count": coins, "block_tile_count": blocks,
                        }

    @staticmethod
    def _compile_structure(target: Mapping[str, int]) -> Level:
        """Build exact non-coin counts on a traversable support scaffold."""
        data = build_static_support_scaffold()
        blocks = int(target["block_tile_count"])
        data[15, :blocks] = int(SMBTiles.SOLID)

        pipe_starts = (6, 13, 20, 27)
        for x in pipe_starts[: int(target["pipe_count"])]:
            # Two-wide, two-high pipes stand on the fixed bottom platform.
            data[13:15, x : x + 2] = int(SMBTiles.TUBE)

        occupied = {
            x
            for start in pipe_starts[: int(target["pipe_count"])]
            for x in (start, start + 1)
        }
        enemy_starts = (3, 10, 17, 24, 30, 1)
        placed = 0
        requested_enemies = int(target["enemy_tile_count"])
        if requested_enemies:
            for x in enemy_starts:
                if x in occupied:
                    continue
                data[14, x] = int(SMBTiles.GOOMBA)
                placed += 1
                if placed == requested_enemies:
                    break
        if placed != requested_enemies:
            raise RuntimeError("static compiler exhausted supported enemy slots")
        return Level(data, "smb_static")

    @staticmethod
    def _trajectory_sites(level: Level, result: EvalResult) -> list[tuple[int, int]]:
        """Return empty cells actually occupied by Mario, spread by column."""
        by_x: dict[int, list[int]] = {}
        for x_px, y_px in (result.info or {}).get("locations", []):
            x = max(0, min(level.width - 1, int(float(x_px) / 16) - 3))
            y = max(0, min(level.height - 1, int(float(y_px) / 16)))
            if not 2 <= x <= 29:
                continue
            if int(level.data[y, x]) != int(SMBTiles.EMPTY):
                continue
            by_x.setdefault(x, []).append(y)
        sites = [(x, max(by_x[x])) for x in sorted(by_x)]
        if len(sites) <= 12:
            return sites
        indexes = np.linspace(0, len(sites) - 1, num=12, dtype=int)
        return [sites[int(index)] for index in indexes]

    def lookup(self, spec: FunctionalSpec) -> FunctionalCandidate | None:
        # A count-matching grid is not a certified lookup hit.  Static targets
        # go through bounded construction so coin placement can be derived from
        # and then verified against a real A* trajectory.
        return None

    def evaluate(self, level: Level) -> EvalResult:
        return self._evaluator.evaluate(level)

    def result_matches(self, result: EvalResult, selected_target: Mapping[str, int]) -> bool:
        return bool(
            result.valid
            and result.problem_type == "smb_static"
            and (result.info or {}).get("evaluation_mode")
            == "static_grid_plus_gameplay"
            and (result.info or {}).get("simulation_called") is True
            and (result.info or {}).get("engine_executed") is True
            and all(int(result.metrics.get(name, -1)) == int(value)
                    for name, value in selected_target.items())
        )

    def search(self, spec, evaluate, remaining_calls) -> SearchOutcome:
        target = select_satisfying_target(
            spec, self._targets(), stable_metric_order=self.target_names
        )
        if target is None:
            return SearchOutcome(None, None, "no_satisfying_static_target", {})
        if remaining_calls() <= 0:
            return SearchOutcome(
                None, None, "budget_exhausted", {"budget_exhausted": True}
            )

        base = self._compile_structure(target)
        base_result = evaluate(base)
        if not base_result.valid:
            return SearchOutcome(
                None,
                base_result,
                "static_playable_compiler",
                {"phase": "structure", "errors": list(base_result.errors)},
            )

        requested_coins = int(target["coin_tile_count"])
        provenance = {
            "simulation_called": True,
            "search_required": True,
            "support_scaffold": "full_width_platform_floor",
            "coin_rule": "verified_trajectory_overlay",
        }
        if requested_coins == 0:
            candidate = FunctionalCandidate(
                base, target, "static_playable_compiler", provenance
            )
            return SearchOutcome(
                candidate,
                base_result,
                candidate.strategy,
                {"phase": "structure", "overlay_candidates": 0},
            )

        sites = self._trajectory_sites(base, base_result)
        profiles = [sites, list(reversed(sites))]
        if len(sites) >= requested_coins:
            profiles.append(sites[::2] + sites[1::2])
        tried = 0
        seen: set[tuple[tuple[int, int], ...]] = set()
        for profile in profiles:
            selected = tuple(profile[:requested_coins])
            if len(selected) != requested_coins or selected in seen:
                continue
            seen.add(selected)
            if remaining_calls() <= 0:
                break
            data = base.data.copy()
            for x, y in selected:
                data[y, x] = int(SMBTiles.COIN)
            level = Level(data, "smb_static")
            tried += 1
            result = evaluate(level)
            if self.result_matches(result, target):
                candidate = FunctionalCandidate(
                    level,
                    target,
                    "static_playable_compiler",
                    {**provenance, "coin_sites": [[x, y] for x, y in selected]},
                )
                return SearchOutcome(
                    candidate,
                    result,
                    candidate.strategy,
                    {
                        "phase": "trajectory_coin_overlay",
                        "overlay_candidates": tried,
                    },
                )
        return SearchOutcome(
            None,
            None,
            "static_playable_compiler",
            {
                "phase": "trajectory_coin_overlay",
                "overlay_candidates": tried,
                "available_trajectory_sites": len(sites),
                "budget_exhausted": remaining_calls() == 0,
            },
        )


def build_resolver(*, disabled_backbones: Iterable[str] = ()) -> FunctionalResolver:
    resolver = FunctionalResolver()
    resolver.register(SmbDynamicFunctionalAdapter(disabled_backbones=disabled_backbones))
    resolver.register(SmbStaticFunctionalAdapter())
    return resolver


LANGUAGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "domain": {"type": "string", "enum": ["smb_dynamic", "smb_static", "unknown"]},
        "constraints": {
            "type": "object", "additionalProperties": False,
            "properties": {
                name: {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "relation": {"type": "string", "enum": ["exact", "min", "max", "unspecified"]},
                        "value": {"type": "integer", "minimum": -1, "maximum": 32},
                    },
                    "required": ["relation", "value"],
                }
                for name in ALL_LANGUAGE_METRICS
            },
            "required": list(ALL_LANGUAGE_METRICS),
        },
        "objectives": {
            "type": "array", "maxItems": 3,
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "metric": {"type": "string", "enum": list(ALL_LANGUAGE_METRICS)},
                    "direction": {"type": "string", "enum": ["minimize", "maximize"]},
                },
                "required": ["metric", "direction"],
            },
        },
        "ambiguities": {
            "type": "array", "maxItems": 4,
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": [
                            "vague_numeric",
                            "conflicting_constraints",
                            "dynamic_static_ambiguity",
                            "missing_functional_target",
                            "other",
                        ],
                    },
                    "evidence": {"type": "string"},
                },
                "required": ["kind", "evidence"],
            },
        },
    },
    "required": ["domain", "constraints", "objectives", "ambiguities"],
}


LANGUAGE_SYSTEM_PROMPT = """You compile free-form Super Mario level requests into formal functional constraints.

There are two separate domains:
- smb_dynamic: outcomes observed while Mario plays: enemies=kills counted by simulation, coins=coins collected, jumps=ground jumps.
- smb_static: literal tile-grid targets: pipe_count, enemy_tile_count, coin_tile_count, block_tile_count. These values are not gameplay outcomes, but the downstream verifier also requires supported objects, A* completion, and collection of every coin tile.

Supported ranges are dynamic enemies 0..4, coins 0..5, jumps 0..6; static pipes 0..4, enemy tiles 0..6, coin tiles 0..6, blocks 0..32. Extract an explicit out-of-range value faithfully; the harness, not you, validates support. Partial constraints are allowed when the user intentionally leaves a metric free.

Use relation exact/min/max. Use relation=unspecified,value=-1 for every irrelevant or omitted metric. The two SMB domains are mutually exclusive in the output: when domain=smb_static, enemies, coins, and jumps MUST all be unspecified; when domain=smb_dynamic, pipe_count, enemy_tile_count, coin_tile_count, and block_tile_count MUST all be unspecified. Never copy a tile count into its similarly named gameplay-event field or vice versa. Put soft requests such as 'as few jumps as possible' in objectives. Record genuine ambiguity, conflicting constraints, or the absence of a functional target in ambiguities with brief evidence. Words such as enemies or coins "in the level" are dynamic/static ambiguous unless the request says gameplay outcomes or tile/grid counts. An unspecified metric in the other SMB domain is normal and is not an ambiguity. For a different game/domain use domain=unknown. Do not decide whether to execute, do not silently invent numbers, and return only the schema object."""


def parser_payload_to_spec(payload: Mapping[str, Any]) -> FunctionalSpec:
    domain = str(payload["domain"])
    constraints = {
        name: MetricConstraint(str(item["relation"]), int(item["value"]))
        for name, item in payload["constraints"].items()
        if item["relation"] != "unspecified"
    }
    objectives = tuple(
        FunctionalObjective(str(item["metric"]), str(item["direction"]))
        for item in payload.get("objectives", [])
    )
    return FunctionalSpec(domain, constraints, objectives)


def _concrete_constraints(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: item
        for name, item in payload.get("constraints", {}).items()
        if item.get("relation") != "unspecified"
    }


def derive_parser_disposition(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Derive execution status without trusting an LLM confidence decision."""

    domain = str(payload.get("domain", "unknown"))
    if domain == "unknown":
        return {"status": "unsupported", "reason": "unregistered_or_unknown_domain", "errors": []}
    ambiguities = list(payload.get("ambiguities", []))
    if ambiguities:
        return {"status": "needs_clarification", "reason": "explicit_ambiguity", "errors": []}
    if not _concrete_constraints(payload):
        return {"status": "needs_clarification", "reason": "no_concrete_constraint", "errors": []}
    try:
        spec = parser_payload_to_spec(payload)
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "needs_clarification", "reason": "invalid_typed_parse", "errors": [str(exc)]}
    adapter = (
        SmbDynamicFunctionalAdapter()
        if spec.domain == "smb_dynamic"
        else SmbStaticFunctionalAdapter()
        if spec.domain == "smb_static"
        else None
    )
    if adapter is None:
        return {"status": "unsupported", "reason": "unregistered_or_unknown_domain", "errors": []}
    errors = adapter.validate(spec)
    if errors:
        return {"status": "unsupported", "reason": "domain_validation_failed", "errors": errors}
    return {"status": "resolved", "reason": "validator_clean_typed_spec", "errors": []}


def is_fail_closed(
    expected_status: str, actual_status: str, resolution: Any | None
) -> bool:
    """Safety outcome: a guard remains closed, regardless of closed subtype."""

    closed = {"needs_clarification", "unsupported"}
    return expected_status in closed and actual_status in closed and resolution is None


def _canonical_parser_payload(
    payload: Mapping[str, Any], *, status: str | None = None
) -> dict[str, Any]:
    raw_constraints = payload.get("constraints", {})
    canonical = {
        "domain": payload["domain"],
        "constraints": {
            name: {
                "relation": raw_constraints.get(
                    name, {"relation": "unspecified"}
                )["relation"],
                "value": int(raw_constraints.get(name, {"value": -1})["value"]),
            }
            for name in ALL_LANGUAGE_METRICS
        },
        "objectives": sorted(
            ({"metric": item["metric"], "direction": item["direction"]}
             for item in payload.get("objectives", [])),
            key=lambda item: (item["metric"], item["direction"]),
        ),
        "ambiguities": sorted(
            ({"kind": item["kind"], "evidence": item.get("evidence", "")}
             for item in payload.get("ambiguities", [])),
            key=lambda item: (item["kind"], item["evidence"]),
        ),
    }
    effective_status = status if status is not None else payload.get("status")
    if effective_status is not None:
        canonical["status"] = effective_status
    return canonical


def preflight(output: Path) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    exact = FunctionalSpec("smb_dynamic", {
        "enemies": MetricConstraint("exact", 3),
        "coins": MetricConstraint("exact", 4),
        "jumps": MetricConstraint("exact", 1),
    })
    hit = build_resolver().resolve(exact, max_evaluator_calls=4)
    cases.append({"case": "dynamic_archive_hit", **hit.as_dict(include_level=False)})

    jump = FunctionalSpec("smb_dynamic", {
        "enemies": MetricConstraint("exact", 0),
        "coins": MetricConstraint("exact", 5),
        "jumps": MetricConstraint("exact", 5),
    })
    jump_search = build_resolver(disabled_backbones={"k0_j5"}).resolve(
        jump, max_evaluator_calls=8
    )
    cases.append({"case": "dynamic_jump_search_on_miss", **jump_search.as_dict(include_level=False)})

    stomp = FunctionalSpec("smb_dynamic", {
        "enemies": MetricConstraint("exact", 3),
        "coins": MetricConstraint("exact", 3),
        "jumps": MetricConstraint("exact", 0),
    })
    stomp_search = build_resolver(disabled_backbones={"k3_j0"}).resolve(
        stomp, max_evaluator_calls=8
    )
    cases.append({"case": "dynamic_stomp_search_on_miss", **stomp_search.as_dict(include_level=False)})

    preference = FunctionalSpec(
        "smb_dynamic",
        {"enemies": MetricConstraint("min", 2), "coins": MetricConstraint("exact", 3)},
        (FunctionalObjective("jumps", "minimize"),),
    )
    preferred = build_resolver().resolve(preference, max_evaluator_calls=4)
    cases.append({"case": "dynamic_range_plus_objective", **preferred.as_dict(include_level=False)})

    static = FunctionalSpec("smb_static", {
        "pipe_count": MetricConstraint("exact", 4),
        "enemy_tile_count": MetricConstraint("exact", 6),
        "coin_tile_count": MetricConstraint("exact", 5),
        "block_tile_count": MetricConstraint("exact", 32),
    })
    compiled = build_resolver().resolve(static, max_evaluator_calls=4)
    cases.append({"case": "static_playable_compile", **compiled.as_dict(include_level=True)})

    unknown = FunctionalSpec("smb_dynamic", {
        "enemies": MetricConstraint("exact", 4),
        "coins": MetricConstraint("exact", 0),
        "jumps": MetricConstraint("exact", 6),
    })
    miss = build_resolver().resolve(unknown, max_evaluator_calls=0)
    cases.append({"case": "bounded_miss_is_unknown", **miss.as_dict(include_level=False)})

    criteria = {
        "archive_hit_exact": hit.status == "exact" and hit.evaluator_calls == 1,
        "jump_search_exact": jump_search.status == "exact" and jump_search.source == "bounded_search",
        "stomp_search_exact": stomp_search.status == "exact" and stomp_search.source == "bounded_search",
        "range_objective_exact": preferred.status == "exact" and preferred.selected_target == {"enemies": 2, "coins": 3, "jumps": 0},
        "static_compile_playable_after_search": (
            compiled.status == "exact"
            and compiled.search["attempted"]
            and compiled.provenance.get("simulation_called") is True
            and compiled.observed_metrics.get("complete") == 1.0
            and compiled.observed_metrics.get("coin_tiles_collected") == 5
            and compiled.observed_metrics.get("unsupported_enemy_count") == 0
            and compiled.observed_metrics.get("unsupported_pipe_count") == 0
        ),
        "bounded_miss_unknown": miss.status == "unknown" and miss.evaluator_calls == 0,
    }
    payload = {
        "run_id": RUN_ID,
        "gate_pass": all(criteria.values()),
        "criteria": criteria,
        "domains": list(build_resolver().domains()),
        "evaluator_calls_are_monetary_cost": False,
        "unknown_is_not_infeasible": True,
        "cases": cases,
    }
    _write_json(output, payload)
    if not payload["gate_pass"]:
        raise RuntimeError("R108 functional resolver preflight failed")
    return payload


def run_language(
    manifest: Path,
    base_url: str,
    output: Path,
    *,
    model_id: str,
    revision: str,
    seed: int = SEED,
) -> dict[str, Any]:
    from openai import OpenAI

    frozen = json.loads(manifest.read_text())
    client = OpenAI(base_url=base_url, api_key="EMPTY", max_retries=0, timeout=600.0)
    records: list[dict[str, Any]] = []
    for index, case in enumerate(frozen["cases"]):
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": LANGUAGE_SYSTEM_PROMPT},
                {"role": "user", "content": case["instruction"]},
            ],
            temperature=0.0,
            seed=seed + index,
            max_tokens=1024,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "smb_functional_spec", "strict": True, "schema": LANGUAGE_SCHEMA},
            },
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        latency_ms = int(round((time.perf_counter() - started) * 1000))
        message = response.choices[0].message
        parsed = json.loads(message.content or "{}")
        disposition = derive_parser_disposition(parsed)
        canonical = _canonical_parser_payload(parsed, status=disposition["status"])
        expected = _canonical_parser_payload(case["expected"])
        expected_status = case["expected"]["status"]
        parse_exact = bool(
            disposition["status"] == expected_status
            and parsed["domain"] == case["expected"]["domain"]
            and (
                expected_status != "resolved"
                or (
                    canonical["constraints"] == expected["constraints"]
                    and canonical["objectives"] == expected["objectives"]
                    and not canonical["ambiguities"]
                )
            )
        )
        resolution = None
        if disposition["status"] == "resolved":
            try:
                spec = parser_payload_to_spec(parsed)
                disabled = set(case.get("disabled_backbones", []))
                resolution = build_resolver(disabled_backbones=disabled).resolve(
                    spec,
                    allow_search=True,
                    max_evaluator_calls=int(case.get("max_evaluator_calls", 64)),
                )
            except (TypeError, ValueError) as exc:
                resolution = {"status": "invalid_request", "errors": [str(exc)]}
        resolution_payload = (
            resolution.as_dict(include_level=True)
            if hasattr(resolution, "as_dict")
            else resolution
        )
        fail_closed = is_fail_closed(
            expected_status, disposition["status"], resolution
        )
        functional_exact = bool(
            expected_status == "resolved"
            and parse_exact
            and resolution_payload
            and resolution_payload["status"] == "exact"
        )
        usage = response.usage
        records.append({
            "case_id": case["case_id"],
            "instruction": case["instruction"],
            "expected": expected,
            "raw_parser_payload": parsed,
            "parsed": canonical,
            "disposition": disposition,
            "parse_exact": parse_exact,
            "functional_exact": functional_exact,
            "fail_closed": fail_closed,
            "resolution": resolution_payload,
            "costs": {
                "llm_calls": 1,
                "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "latency_ms": latency_ms,
            },
        })
        _write_json(output, {"execution_complete": False, "records": records})

    resolved = [row for row in records if row["expected"]["status"] == "resolved"]
    guarded = [row for row in records if row["expected"]["status"] != "resolved"]
    aggregate = {
        "cases": len(records),
        "parse_exact": sum(row["parse_exact"] for row in records),
        "resolved_cases": len(resolved),
        "functional_exact": sum(row["functional_exact"] for row in resolved),
        "guard_cases": len(guarded),
        "fail_closed": sum(row["fail_closed"] for row in guarded),
        "archive_hits": sum((row.get("resolution") or {}).get("source") == "archive_or_compiler_hit" for row in resolved),
        "bounded_search_hits": sum((row.get("resolution") or {}).get("source") == "bounded_search" for row in resolved),
        "evaluator_calls": sum(int((row.get("resolution") or {}).get("evaluator_calls", 0)) for row in resolved),
        "llm_calls": len(records),
        "input_tokens": sum(row["costs"]["input_tokens"] for row in records),
        "output_tokens": sum(row["costs"]["output_tokens"] for row in records),
        "latency_ms": sum(row["costs"]["latency_ms"] for row in records),
    }
    criteria = {
        "execution_complete": len(records) == len(frozen["cases"]),
        "parse_exact_at_least_90pct": aggregate["parse_exact"] >= int(0.9 * len(records) + 0.999999),
        "resolved_functional_exact_all": aggregate["functional_exact"] == len(resolved),
        "ambiguous_or_unsupported_fail_closed_all": aggregate["fail_closed"] == len(guarded),
    }
    payload = {
        "run_id": RUN_ID,
        "execution_complete": criteria["execution_complete"],
        "scientific_gate_pass": all(criteria.values()),
        "criteria": criteria,
        "model_id": model_id,
        "revision": revision,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "architecture": "free-form language -> FunctionalSpec -> domain resolver -> verifier",
        "llm_role": "language front end only",
        "status_policy": "deterministic harness decision from domain, ambiguities, and adapter validation",
        "evaluator_calls_are_monetary_cost": False,
        "aggregate": aggregate,
        "records": records,
    }
    _write_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run-language")
    run.add_argument("--manifest", type=Path, default=DEFAULT_LANGUAGE_MANIFEST)
    run.add_argument("--base-url", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--model-id", required=True)
    run.add_argument("--revision", required=True)
    run.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.command == "preflight":
        payload = preflight(args.output)
    else:
        payload = run_language(
            args.manifest, args.base_url, args.output,
            model_id=args.model_id, revision=args.revision, seed=args.seed,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
