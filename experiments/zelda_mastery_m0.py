"""R060 Zelda Mastery measurement foundations.

This module is deliberately CPU-only and model-free.  It defines the v0
feasibility oracle, frozen target split, and versioned episode contract used by
all later Qwen3.8-27B harness experiments.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from toolusepcg.config import load_config
from toolusepcg.evaluator import EvalResult, StatsEvaluator
from toolusepcg.factory import LevelFactory
from toolusepcg.level import Level, ZeldaTiles


DOMAIN_MIN = 10
DOMAIN_MAX = 40
DOMAIN_SIZE = (DOMAIN_MAX - DOMAIN_MIN + 1) ** 2
GRID_SIZE = 16
ORACLE_VERSION = "r060-corridor-v1"
MANIFEST_VERSION = "r060-target-split-v2"
EPISODE_SCHEMA_VERSION = "zelda-mastery-episode-v2"
MANIFEST_SEED = 60017
FAILURE_TAXONOMY = (
    "protocol",
    "hard-validity",
    "tool-effect",
    "oscillation",
    "budget",
    "planning",
    "target-miss",
)
SPLIT_SIZES = {"dev": 120, "validation": 120, "test": 721}
# Frozen before the v2 split is generated.  "Corners" means the four exact
# 4x4 neighborhoods [10,13]x[10,13], [10,13]x[37,40], etc.; it never means
# merely the four exact corner points.
STRATA_MINIMUMS = {
    "dev": {"corner": 6, "boundary": 6, "extreme_imbalance": 12,
            "sum_band": 8, "parity": 24},
    "validation": {"corner": 6, "boundary": 6, "extreme_imbalance": 12,
                   "sum_band": 8, "parity": 24},
    "test": {"corner": 4, "boundary": 4, "extreme_imbalance": 8,
             "sum_band": 20, "parity": 150},
}


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for content-addressed artifacts."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _corridor_coordinates() -> list[tuple[int, int]]:
    """Return a long induced path in the 16x16 grid as (y, x) cells.

    Horizontal lanes are separated by a wall row and connected at alternating
    ends.  Consequently, no two non-consecutive corridor cells are adjacent,
    so every prefix has exactly the intended shortest-path distances.
    """
    coordinates: list[tuple[int, int]] = []
    for lane_index, y in enumerate(range(1, GRID_SIZE - 1, 2)):
        xs: Iterable[int]
        if lane_index % 2 == 0:
            xs = range(1, GRID_SIZE - 1)
        else:
            xs = range(GRID_SIZE - 2, 0, -1)
        coordinates.extend((y, x) for x in xs)
        if y + 2 < GRID_SIZE - 1:
            coordinates.append((y + 1, coordinates[-1][1]))
    return coordinates


_CORRIDOR = _corridor_coordinates()


def construct_zelda_corridor(
    player_key: int,
    key_door: int,
    *,
    enemies: int = 3,
) -> Level:
    """Construct a hard-valid Zelda level with exact component distances."""
    for name, value in (("player_key", player_key), ("key_door", key_door)):
        if not DOMAIN_MIN <= value <= DOMAIN_MAX:
            raise ValueError(
                f"{name}={value} is outside [{DOMAIN_MIN}, {DOMAIN_MAX}]"
            )
    total = player_key + key_door
    if total >= len(_CORRIDOR):
        raise ValueError("corridor template is too short for requested target")
    if enemies < 0 or enemies > total - 2:
        raise ValueError("enemy count cannot fit without replacing P/K/D")

    data = np.full(
        (GRID_SIZE, GRID_SIZE), int(ZeldaTiles.WALL), dtype=np.int32
    )
    path = _CORRIDOR[: total + 1]
    for y, x in path:
        data[y, x] = int(ZeldaTiles.EMPTY)

    special_indices = {0, player_key, total}
    enemy_indices = [
        index for index in range(1, total) if index not in special_indices
    ][:enemies]
    for index in enemy_indices:
        y, x = path[index]
        data[y, x] = int(ZeldaTiles.ENEMY)

    py, px = path[0]
    ky, kx = path[player_key]
    dy, dx = path[total]
    data[py, px] = int(ZeldaTiles.PLAYER)
    data[ky, kx] = int(ZeldaTiles.KEY)
    data[dy, dx] = int(ZeldaTiles.DOOR)
    return Level(data=data, problem_type="zelda")


def evaluate_oracle_domain() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate all 961 oracle levels through the production evaluator."""
    config = load_config("configs/zelda.yaml")
    factory = LevelFactory(config)
    evaluator = StatsEvaluator(config, factory.env)
    records: list[dict[str, Any]] = []
    for player_key in range(DOMAIN_MIN, DOMAIN_MAX + 1):
        for key_door in range(DOMAIN_MIN, DOMAIN_MAX + 1):
            level = construct_zelda_corridor(player_key, key_door)
            result = evaluator.evaluate(level)
            metrics = {
                key: int(result.metrics[key])
                for key in (
                    "players",
                    "keys",
                    "doors",
                    "enemies",
                    "regions",
                    "player_key",
                    "key_door",
                    "solution_length",
                )
            }
            exact = (
                result.valid
                and metrics["player_key"] == player_key
                and metrics["key_door"] == key_door
            )
            records.append(
                {
                    "target": {
                        "player_key": player_key,
                        "key_door": key_door,
                    },
                    "metrics": metrics,
                    "hard_valid": bool(result.valid),
                    "exact": bool(exact),
                    "errors": list(result.errors),
                    "level_sha256": hashlib.sha256(
                        level.data.tobytes(order="C")
                    ).hexdigest(),
                }
            )
    exact_count = sum(record["exact"] for record in records)
    hard_valid_count = sum(record["hard_valid"] for record in records)
    summary = {
        "artifact_version": ORACLE_VERSION,
        "domain": {
            "grid": [GRID_SIZE, GRID_SIZE],
            "player_key": [DOMAIN_MIN, DOMAIN_MAX],
            "key_door": [DOMAIN_MIN, DOMAIN_MAX],
            "targets": DOMAIN_SIZE,
            "enemies": 3,
        },
        "evaluator": "toolusepcg.evaluator.StatsEvaluator",
        "total": len(records),
        "hard_valid": hard_valid_count,
        "exact": exact_count,
        "gate_pass": (
            len(records) == DOMAIN_SIZE
            and hard_valid_count == DOMAIN_SIZE
            and exact_count == DOMAIN_SIZE
        ),
        "records_sha256": hashlib.sha256(
            b"".join(canonical_json_bytes(record) + b"\n" for record in records)
        ).hexdigest(),
    }
    return records, summary


def _target_tags(player_key: int, key_door: int) -> dict[str, Any]:
    boundary = []
    if player_key == DOMAIN_MIN:
        boundary.append("pk_min")
    if player_key == DOMAIN_MAX:
        boundary.append("pk_max")
    if key_door == DOMAIN_MIN:
        boundary.append("kd_min")
    if key_door == DOMAIN_MAX:
        boundary.append("kd_max")

    corner = None
    if player_key <= 13 and key_door <= 13:
        corner = "low_low"
    elif player_key <= 13 and key_door >= 37:
        corner = "low_high"
    elif player_key >= 37 and key_door <= 13:
        corner = "high_low"
    elif player_key >= 37 and key_door >= 37:
        corner = "high_high"

    difference = player_key - key_door
    if difference >= 20:
        imbalance = "pk_much_longer"
    elif difference <= -20:
        imbalance = "kd_much_longer"
    elif abs(difference) <= 2:
        imbalance = "balanced"
    else:
        imbalance = "moderate"

    path_sum = player_key + key_door
    sum_floor = min((path_sum // 10) * 10, 70)
    sum_band = f"{sum_floor}-{29 if sum_floor == 20 else sum_floor + 9}"
    if sum_floor == 70:
        sum_band = "70-80"
    parity = ("e" if player_key % 2 == 0 else "o") + (
        "e" if key_door % 2 == 0 else "o"
    )
    return {
        "corner_quadrant": corner,
        "boundary": boundary,
        "imbalance": imbalance,
        "path_sum": path_sum,
        "sum_band": sum_band,
        "parity": parity,
    }


def _coverage_tokens(target: tuple[int, int]) -> tuple[str, ...]:
    tags = _target_tags(*target)
    tokens = [
        f"imbalance:{tags['imbalance']}",
        f"sum_band:{tags['sum_band']}",
        f"parity:{tags['parity']}",
    ]
    if tags["corner_quadrant"]:
        tokens.append(f"corner:{tags['corner_quadrant']}")
    tokens.extend(f"boundary:{side}" for side in tags["boundary"])
    return tuple(tokens)


def _expanded_minimums(split: str) -> dict[str, int]:
    minimum = STRATA_MINIMUMS[split]
    tokens: dict[str, int] = {}
    for name in ("low_low", "low_high", "high_low", "high_high"):
        tokens[f"corner:{name}"] = minimum["corner"]
    for name in ("pk_min", "pk_max", "kd_min", "kd_max"):
        tokens[f"boundary:{name}"] = minimum["boundary"]
    for name in ("pk_much_longer", "kd_much_longer"):
        tokens[f"imbalance:{name}"] = minimum["extreme_imbalance"]
    for name in ("20-29", "30-39", "40-49", "50-59", "60-69", "70-80"):
        tokens[f"sum_band:{name}"] = minimum["sum_band"]
    for name in ("ee", "eo", "oe", "oo"):
        tokens[f"parity:{name}"] = minimum["parity"]
    return tokens


def _fill_stratified(
    selected: set[tuple[int, int]],
    available: set[tuple[int, int]],
    size: int,
    split: str,
) -> None:
    counts: Counter[str] = Counter()
    for target in selected:
        counts.update(_coverage_tokens(target))
    while len(selected) < size:
        def rank(target: tuple[int, int]) -> tuple[float, str]:
            novelty = sum(
                1.0 / (1 + counts[token]) for token in _coverage_tokens(target)
            )
            tie = hashlib.sha256(
                f"{MANIFEST_SEED}:{split}:{target[0]}:{target[1]}".encode()
            ).hexdigest()
            return novelty, tie

        chosen = max(available, key=rank)
        available.remove(chosen)
        selected.add(chosen)
        counts.update(_coverage_tokens(chosen))


def _allocate_minimum_strata(
    all_targets: set[tuple[int, int]],
) -> dict[str, set[tuple[int, int]]]:
    """Reserve every minimum category before deterministically filling splits."""
    assigned = {split: set() for split in SPLIT_SIZES}
    remaining = set(all_targets)
    requirements = {split: _expanded_minimums(split) for split in SPLIT_SIZES}

    category_order = ("corner:", "boundary:", "imbalance:", "sum_band:", "parity:")
    for prefix in category_order:
        tokens = sorted({
            token for required in requirements.values() for token in required
            if token.startswith(prefix)
        })
        for token in tokens:
            # Allocate the largest split first so its intentionally larger test
            # minimum is reserved rather than consuming leftovers.
            split_order = sorted(
                SPLIT_SIZES,
                key=lambda split: (-requirements[split][token], split),
            )
            for split in split_order:
                counts = Counter(
                    item for target in assigned[split]
                    for item in _coverage_tokens(target)
                )
                while counts[token] < requirements[split][token]:
                    candidates = [
                        target for target in remaining
                        if token in _coverage_tokens(target)
                    ]
                    if not candidates or len(assigned[split]) >= SPLIT_SIZES[split]:
                        raise RuntimeError(f"cannot reserve {split} {token}")
                    active = {
                        name for name, minimum in requirements[split].items()
                        if counts[name] < minimum
                    }

                    def rank(target: tuple[int, int]) -> tuple[int, str]:
                        covered = len(active & set(_coverage_tokens(target)))
                        tie = hashlib.sha256(
                            f"{MANIFEST_SEED}:reserve:{split}:{token}:{target}".encode()
                        ).hexdigest()
                        return covered, tie

                    chosen = max(candidates, key=rank)
                    assigned[split].add(chosen)
                    remaining.remove(chosen)
                    counts.update(_coverage_tokens(chosen))

    for split in ("dev", "validation"):
        _fill_stratified(assigned[split], remaining, SPLIT_SIZES[split], split)
    assigned["test"].update(remaining)

    if any(len(assigned[split]) != size for split, size in SPLIT_SIZES.items()):
        raise RuntimeError("stratified allocator produced wrong split sizes")
    return assigned


def _split_coverage(targets: list[tuple[int, int]]) -> dict[str, Any]:
    tags = [_target_tags(*target) for target in targets]
    return {
        "corner_quadrants": dict(
            sorted(Counter(t["corner_quadrant"] for t in tags if t["corner_quadrant"]).items())
        ),
        "boundary_sides": dict(
            sorted(Counter(side for t in tags for side in t["boundary"]).items())
        ),
        "imbalance": dict(sorted(Counter(t["imbalance"] for t in tags).items())),
        "sum_bands": dict(sorted(Counter(t["sum_band"] for t in tags).items())),
        "parity": dict(sorted(Counter(t["parity"] for t in tags).items())),
        "path_sum_min": min(t["path_sum"] for t in tags),
        "path_sum_max": max(t["path_sum"] for t in tags),
    }


def build_target_manifest() -> dict[str, Any]:
    """Build the deterministic 120/120/721 target partition and its hash."""
    all_targets = {
        (player_key, key_door)
        for player_key in range(DOMAIN_MIN, DOMAIN_MAX + 1)
        for key_door in range(DOMAIN_MIN, DOMAIN_MAX + 1)
    }
    allocated = _allocate_minimum_strata(all_targets)
    split_targets = {
        split: sorted(targets) for split, targets in allocated.items()
    }
    entries = []
    for split in ("dev", "validation", "test"):
        for player_key, key_door in split_targets[split]:
            entries.append(
                {
                    "target_id": f"pk{player_key:02d}-kd{key_door:02d}",
                    "split": split,
                    "player_key": player_key,
                    "key_door": key_door,
                    "tags": _target_tags(player_key, key_door),
                }
            )
    payload = {
        "schema_version": MANIFEST_VERSION,
        "generator_seed": MANIFEST_SEED,
        "domain": {
            "grid": [GRID_SIZE, GRID_SIZE],
            "player_key": [DOMAIN_MIN, DOMAIN_MAX],
            "key_door": [DOMAIN_MIN, DOMAIN_MAX],
            "total": DOMAIN_SIZE,
        },
        "split_sizes": SPLIT_SIZES,
        "strata_definition": {
            "corners": (
                "four 4x4 neighborhoods: each coordinate is respectively "
                "in [10,13] or [37,40]"
            ),
            "boundary": "pk or kd equals 10 or 40",
            "extreme_imbalance": "absolute component difference >= 20",
            "sum_bands": [
                "20-29", "30-39", "40-49", "50-59", "60-69", "70-80"
            ],
            "parity": ["ee", "eo", "oe", "oo"],
        },
        "strata_minimums": STRATA_MINIMUMS,
        "coverage": {
            split: _split_coverage(targets)
            for split, targets in split_targets.items()
        },
        "targets": entries,
    }
    return {"manifest_sha256": sha256_json(payload), **payload}


def validate_target_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Recompute completeness, tags, coverage and every frozen stratum gate."""
    payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    errors: list[str] = []
    if manifest.get("manifest_sha256") != sha256_json(payload):
        errors.append("manifest_sha256 mismatch")
    if manifest.get("split_sizes") != SPLIT_SIZES:
        errors.append("split sizes differ from frozen contract")
    if manifest.get("strata_minimums") != STRATA_MINIMUMS:
        errors.append("strata minimums differ from frozen contract")

    entries = manifest.get("targets", [])
    by_split: dict[str, list[tuple[int, int]]] = {split: [] for split in SPLIT_SIZES}
    for entry in entries:
        pair = (entry.get("player_key"), entry.get("key_door"))
        split = entry.get("split")
        if split not in by_split:
            errors.append(f"unknown split {split!r}")
            continue
        if entry.get("tags") != _target_tags(*pair):
            errors.append(f"tag mismatch for {pair}")
        by_split[split].append(pair)
    expected = {
        (pk, kd) for pk in range(DOMAIN_MIN, DOMAIN_MAX + 1)
        for kd in range(DOMAIN_MIN, DOMAIN_MAX + 1)
    }
    flat = [pair for pairs in by_split.values() for pair in pairs]
    if len(flat) != DOMAIN_SIZE or set(flat) != expected or len(set(flat)) != len(flat):
        errors.append("targets are not a complete disjoint domain partition")

    for split, size in SPLIT_SIZES.items():
        if len(by_split[split]) != size:
            errors.append(f"{split} size is {len(by_split[split])}, expected {size}")
        coverage = _split_coverage(by_split[split])
        if manifest.get("coverage", {}).get(split) != coverage:
            errors.append(f"{split} coverage is stale")
        token_counts = Counter(
            token for pair in by_split[split] for token in _coverage_tokens(pair)
        )
        for token, minimum in _expanded_minimums(split).items():
            if token_counts[token] < minimum:
                errors.append(
                    f"{split} {token}={token_counts[token]} below minimum {minimum}"
                )
    return {"gate_pass": not errors, "errors": errors}


def episode_schema() -> dict[str, Any]:
    """Return the canonical JSON Schema for every later model episode."""
    hash_schema = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    versioned_hash = {
        "type": "object",
        "additionalProperties": False,
        "required": ["version", "sha256"],
        "properties": {
            "version": {"type": "string", "minLength": 1},
            "sha256": hash_schema,
        },
    }
    delta = {
        "type": "object",
        "additionalProperties": False,
        "required": ["player_key", "key_door"],
        "properties": {
            "player_key": {"type": "integer"},
            "key_door": {"type": "integer"},
        },
    }
    state_snapshot = {
        "type": "object",
        "additionalProperties": False,
        "required": ["level_ascii", "sha256", "metrics"],
        "properties": {
            "level_ascii": {"type": "string", "minLength": 1},
            "sha256": hash_schema,
            "metrics": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "players", "keys", "doors", "enemies", "regions",
                    "player_key", "key_door", "solution_length",
                ],
                "properties": {
                    name: {"type": "integer"}
                    for name in (
                        "players", "keys", "doors", "enemies", "regions",
                        "player_key", "key_door", "solution_length",
                    )
                },
            },
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://prompt-evo.local/schemas/zelda-mastery-episode-v1.json",
        "title": "Zelda Mastery per-episode record",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version", "run_id", "episode_id", "target", "split",
            "manifest_sha256", "model", "harness", "initialization",
            "actions", "final", "primary_failure", "errors", "budget", "costs",
        ],
        "properties": {
            "schema_version": {"const": EPISODE_SCHEMA_VERSION},
            "run_id": {"type": "string", "minLength": 1},
            "episode_id": {"type": "string", "minLength": 1},
            "target": {
                "type": "object", "additionalProperties": False,
                "required": ["player_key", "key_door", "enemies"],
                "properties": {
                    "player_key": {"type": "integer", "minimum": 10, "maximum": 40},
                    "key_door": {"type": "integer", "minimum": 10, "maximum": 40},
                    "enemies": {"type": "integer", "minimum": 0},
                },
            },
            "split": {"enum": ["dev", "validation", "test", "synthetic"]},
            "manifest_sha256": hash_schema,
            "model": {
                "type": "object", "additionalProperties": False,
                "required": [
                    "id", "revision", "runtime", "quantization",
                    "chat_template", "sampling",
                ],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "revision": {"type": "string", "minLength": 1},
                    "runtime": {"type": "string", "minLength": 1},
                    "quantization": {"type": ["string", "null"]},
                    "chat_template": versioned_hash,
                    "sampling": {"type": "object", "minProperties": 1},
                },
            },
            "harness": {
                "type": "object", "additionalProperties": False,
                "required": ["prompt", "tool_schema", "feedback", "memory", "controller"],
                "properties": {
                    name: versioned_hash
                    for name in ("prompt", "tool_schema", "feedback", "memory", "controller")
                },
            },
            "initialization": {
                "type": "object", "additionalProperties": False,
                "required": ["sha256", "seed"],
                "properties": {"sha256": hash_schema, "seed": {"type": "integer"}},
            },
            "actions": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": [
                        "index", "semantic_action", "state_before",
                        "state_after", "expected_delta", "actual_delta",
                        "accepted", "decision_reason", "primitive_edits",
                        "evaluator_calls", "verifier_calls", "latency_ms",
                    ],
                    "properties": {
                        "index": {"type": "integer", "minimum": 0},
                        "semantic_action": {"type": "object"},
                        "state_before": state_snapshot,
                        "state_after": state_snapshot,
                        "expected_delta": delta,
                        "actual_delta": delta,
                        "accepted": {"type": "boolean"},
                        "decision_reason": {"type": "string"},
                        "primitive_edits": {"type": "integer", "minimum": 0},
                        "evaluator_calls": {"type": "integer", "minimum": 0},
                        "verifier_calls": {"type": "integer", "minimum": 0},
                        "latency_ms": {"type": "number", "minimum": 0},
                    },
                },
            },
            "final": {
                "type": "object", "additionalProperties": False,
                "required": ["hard_constraints", "metrics", "e_inf", "e_sum", "exact"],
                "properties": {
                    "hard_constraints": {
                        "type": "object", "additionalProperties": False,
                        "required": [
                            "one_player", "one_key", "one_door", "enemy_count",
                            "regions_one", "player_key_reachable", "key_door_reachable",
                            "protocol_clean", "tool_clean", "hard_valid",
                        ],
                        "properties": {name: {"type": "boolean"} for name in (
                            "one_player", "one_key", "one_door", "enemy_count",
                            "regions_one", "player_key_reachable", "key_door_reachable",
                            "protocol_clean", "tool_clean", "hard_valid",
                        )},
                    },
                    "metrics": {
                        "type": "object", "additionalProperties": False,
                        "required": [
                            "players", "keys", "doors", "enemies", "regions",
                            "player_key", "key_door", "solution_length",
                        ],
                        "properties": {name: {"type": "integer"} for name in (
                            "players", "keys", "doors", "enemies", "regions",
                            "player_key", "key_door", "solution_length",
                        )},
                    },
                    "e_inf": {"type": "integer", "minimum": 0},
                    "e_sum": {"type": "integer", "minimum": 0},
                    "exact": {"type": "boolean"},
                },
            },
            "errors": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["taxonomy", "code", "stage", "message"],
                    "properties": {
                        "taxonomy": {"enum": list(FAILURE_TAXONOMY)},
                        "code": {"type": "string", "minLength": 1},
                        "stage": {"type": "string", "minLength": 1},
                        "message": {"type": "string", "minLength": 1},
                    },
                },
            },
            "primary_failure": {
                "type": ["string", "null"],
                "enum": [*FAILURE_TAXONOMY, None],
            },
            "budget": {
                "type": "object", "additionalProperties": False,
                "required": ["semantic_actions", "verifier_calls", "total_tokens"],
                "properties": {
                    "semantic_actions": {"type": "integer", "minimum": 1},
                    "verifier_calls": {"type": "integer", "minimum": 1},
                    "total_tokens": {"type": "integer", "minimum": 1},
                },
            },
            "costs": {
                "type": "object", "additionalProperties": False,
                "required": [
                    "parser_errors", "tool_errors", "llm_calls", "input_tokens",
                    "output_tokens", "latency_ms", "retries", "semantic_actions",
                    "primitive_edits", "evaluator_calls", "verifier_calls",
                ],
                "properties": {
                    name: {"type": "integer", "minimum": 0}
                    for name in (
                        "parser_errors", "tool_errors", "llm_calls", "input_tokens",
                        "output_tokens", "latency_ms", "retries", "semantic_actions",
                        "primitive_edits", "evaluator_calls", "verifier_calls",
                    )
                },
            },
        },
    }


def _assert_exact_keys(value: dict[str, Any], required: set[str], path: str) -> None:
    missing = required - set(value)
    extra = set(value) - required
    if missing or extra:
        raise ValueError(f"{path}: missing={sorted(missing)}, extra={sorted(extra)}")


def validate_episode(record: dict[str, Any]) -> None:
    """Validate shape plus every deterministic/derived episode invariant."""
    from jsonschema import Draft202012Validator

    schema_errors = sorted(
        Draft202012Validator(episode_schema()).iter_errors(record),
        key=lambda error: list(error.absolute_path),
    )
    if schema_errors:
        error = schema_errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "episode"
        raise ValueError(f"JSON Schema violation at {location}: {error.message}")

    required_top = {
        "schema_version", "run_id", "episode_id", "target", "split",
        "manifest_sha256", "model", "harness", "initialization", "actions",
        "final", "primary_failure", "errors", "budget", "costs",
    }
    _assert_exact_keys(record, required_top, "episode")
    if record["schema_version"] != EPISODE_SCHEMA_VERSION:
        raise ValueError("unsupported episode schema version")
    if record["split"] not in {"dev", "validation", "test", "synthetic"}:
        raise ValueError("invalid split")
    if len(record["manifest_sha256"]) != 64:
        raise ValueError("manifest hash must be sha256")
    if set(record["target"]) != {"player_key", "key_door", "enemies"}:
        raise ValueError("target must contain paired component distances and enemies")

    required_model = {
        "id", "revision", "runtime", "quantization", "chat_template", "sampling"
    }
    _assert_exact_keys(record["model"], required_model, "model")
    _assert_exact_keys(
        record["model"]["chat_template"], {"version", "sha256"}, "chat_template"
    )
    _assert_exact_keys(record["initialization"], {"sha256", "seed"}, "initialization")

    required_harness = {"prompt", "tool_schema", "feedback", "memory", "controller"}
    _assert_exact_keys(record["harness"], required_harness, "harness")
    for item in record["harness"].values():
        _assert_exact_keys(item, {"version", "sha256"}, "harness component")

    required_action = {
        "index", "semantic_action", "state_before", "state_after",
        "expected_delta", "actual_delta", "accepted", "decision_reason",
        "primitive_edits", "evaluator_calls", "verifier_calls", "latency_ms",
    }
    previous_after = None
    seen_states: set[str] = set()
    oscillation = False
    for index, action in enumerate(record["actions"]):
        _assert_exact_keys(action, required_action, f"action[{index}]")
        if action["index"] != index:
            raise ValueError("action indices must be contiguous")
        required_state = {"level_ascii", "sha256", "metrics"}
        _assert_exact_keys(action["state_before"], required_state, "state_before")
        _assert_exact_keys(action["state_after"], required_state, "state_after")
        state_metric_keys = {
            "players", "keys", "doors", "enemies", "regions", "player_key",
            "key_door", "solution_length",
        }
        _assert_exact_keys(
            action["state_before"]["metrics"], state_metric_keys, "state_before.metrics"
        )
        _assert_exact_keys(
            action["state_after"]["metrics"], state_metric_keys, "state_after.metrics"
        )
        for state_name in ("state_before", "state_after"):
            state = action[state_name]
            expected_hash = hashlib.sha256(state["level_ascii"].encode("utf-8")).hexdigest()
            if state["sha256"] != expected_hash:
                raise ValueError(f"action[{index}].{state_name}.sha256 is inconsistent")
            metrics = state["metrics"]
            if metrics["solution_length"] != metrics["player_key"] + metrics["key_door"]:
                raise ValueError(f"action[{index}].{state_name} solution_length is inconsistent")
        if previous_after is not None and action["state_before"] != previous_after:
            raise ValueError("action state chain is not contiguous")
        if index == 0:
            seen_states.add(action["state_before"]["sha256"])
        if action["accepted"] and action["state_after"]["sha256"] in seen_states:
            oscillation = True
        seen_states.add(action["state_after"]["sha256"])
        previous_after = action["state_after"]
        if set(action["expected_delta"]) != {"player_key", "key_door"}:
            raise ValueError("expected_delta is incomplete")
        if set(action["actual_delta"]) != {"player_key", "key_door"}:
            raise ValueError("actual_delta is incomplete")
        derived_delta = {
            name: action["state_after"]["metrics"][name]
            - action["state_before"]["metrics"][name]
            for name in ("player_key", "key_door")
        }
        if action["actual_delta"] != derived_delta:
            raise ValueError(f"action[{index}] actual_delta is inconsistent")

    for error in record["errors"]:
        _assert_exact_keys(error, {"taxonomy", "code", "stage", "message"}, "error")
        if error["taxonomy"] not in FAILURE_TAXONOMY:
            raise ValueError("unknown failure taxonomy")

    final = record["final"]
    _assert_exact_keys(final, {"hard_constraints", "metrics", "e_inf", "e_sum", "exact"}, "final")
    hard = final["hard_constraints"]
    required_hard = {
        "one_player", "one_key", "one_door", "enemy_count", "regions_one",
        "player_key_reachable", "key_door_reachable", "protocol_clean",
        "tool_clean", "hard_valid",
    }
    _assert_exact_keys(hard, required_hard, "hard_constraints")
    required_metrics = {
        "players", "keys", "doors", "enemies", "regions", "player_key",
        "key_door", "solution_length",
    }
    _assert_exact_keys(final["metrics"], required_metrics, "final.metrics")
    metrics = final["metrics"]
    if metrics["solution_length"] != metrics["player_key"] + metrics["key_door"]:
        raise ValueError("final solution_length is inconsistent")
    if previous_after is not None and previous_after["metrics"] != metrics:
        raise ValueError("last action state_after does not match final metrics")
    derived_fields = {
        "one_player": metrics["players"] == 1,
        "one_key": metrics["keys"] == 1,
        "one_door": metrics["doors"] == 1,
        "enemy_count": metrics["enemies"] == record["target"]["enemies"],
        "regions_one": metrics["regions"] == 1,
        "player_key_reachable": metrics["player_key"] > 0,
        "key_door_reachable": metrics["key_door"] > 0,
        "protocol_clean": record["costs"]["parser_errors"] == 0,
        "tool_clean": record["costs"]["tool_errors"] == 0,
    }
    for name, expected in derived_fields.items():
        if hard[name] != expected:
            raise ValueError(f"hard constraint {name} disagrees with derived value")
    derived_hard = all(hard[name] for name in required_hard - {"hard_valid"})
    if hard["hard_valid"] != derived_hard:
        raise ValueError("hard_valid disagrees with hard-constraint fields")
    if final["e_inf"] != max(
        abs(final["metrics"]["player_key"] - record["target"]["player_key"]),
        abs(final["metrics"]["key_door"] - record["target"]["key_door"]),
    ):
        raise ValueError("e_inf is inconsistent")
    if final["e_sum"] != (
        abs(final["metrics"]["player_key"] - record["target"]["player_key"])
        + abs(final["metrics"]["key_door"] - record["target"]["key_door"])
    ):
        raise ValueError("e_sum is inconsistent")
    if final["exact"] != (hard["hard_valid"] and final["e_inf"] == 0):
        raise ValueError("exact is inconsistent")

    required_costs = {
        "parser_errors", "tool_errors", "llm_calls", "input_tokens",
        "output_tokens", "latency_ms", "retries", "semantic_actions",
        "primitive_edits", "evaluator_calls", "verifier_calls",
    }
    _assert_exact_keys(record["costs"], required_costs, "costs")
    if record["costs"]["semantic_actions"] != len(record["actions"]):
        raise ValueError("semantic action cost disagrees with action records")
    for name in ("primitive_edits", "verifier_calls"):
        expected = sum(action[name] for action in record["actions"])
        if record["costs"][name] != expected:
            raise ValueError(f"aggregate {name} disagrees with action records")
    action_evaluations = sum(action["evaluator_calls"] for action in record["actions"])
    if record["costs"]["evaluator_calls"] < action_evaluations:
        raise ValueError("aggregate evaluator_calls is below action records")
    action_latency = sum(action["latency_ms"] for action in record["actions"])
    if record["costs"]["latency_ms"] < action_latency:
        raise ValueError("aggregate latency is below action records")

    detected: list[str] = []
    if record["costs"]["parser_errors"] > 0:
        detected.append("protocol")
    physical_names = {
        "one_player", "one_key", "one_door", "enemy_count", "regions_one",
        "player_key_reachable", "key_door_reachable",
    }
    if not all(hard[name] for name in physical_names):
        detected.append("hard-validity")
    if record["costs"]["tool_errors"] > 0:
        detected.append("tool-effect")
    if oscillation:
        detected.append("oscillation")
    total_tokens = record["costs"]["input_tokens"] + record["costs"]["output_tokens"]
    if (
        record["costs"]["semantic_actions"] >= record["budget"]["semantic_actions"]
        or record["costs"]["verifier_calls"] >= record["budget"]["verifier_calls"]
        or total_tokens >= record["budget"]["total_tokens"]
    ):
        detected.append("budget")
    if not final["exact"] and not detected:
        detected.append("planning" if record["actions"] else "target-miss")

    labelled = [error["taxonomy"] for error in record["errors"]]
    if set(labelled) != set(detected):
        raise ValueError(
            f"failure taxonomy labels {sorted(set(labelled))} disagree with "
            f"derived failures {sorted(set(detected))}"
        )
    expected_primary = next(
        (taxonomy for taxonomy in FAILURE_TAXONOMY if taxonomy in detected), None
    )
    if record["primary_failure"] != expected_primary:
        raise ValueError(
            f"primary_failure must follow frozen precedence; expected {expected_primary!r}"
        )


def _snapshot(level_ascii: str, metrics: dict[str, int]) -> dict[str, Any]:
    return {
        "level_ascii": level_ascii,
        "sha256": hashlib.sha256(level_ascii.encode("utf-8")).hexdigest(),
        "metrics": deepcopy(metrics),
    }


def synthetic_failure_records() -> list[dict[str, Any]]:
    """Create semantically valid witnesses, not merely enum-labelled records."""
    zero_hash = "0" * 64
    component = {"version": "synthetic-v1", "sha256": zero_hash}
    base = {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "run_id": "R060-D-synthetic",
        "episode_id": "",
        "target": {"player_key": 20, "key_door": 20, "enemies": 3},
        "split": "synthetic",
        "manifest_sha256": zero_hash,
        "model": {
            "id": "synthetic", "revision": "synthetic-revision", "runtime": "cpu-fixture",
            "quantization": None, "chat_template": deepcopy(component),
            "sampling": {"temperature": 0.0},
        },
        "harness": {
            name: deepcopy(component)
            for name in ("prompt", "tool_schema", "feedback", "memory", "controller")
        },
        "initialization": {"sha256": zero_hash, "seed": 0},
        "actions": [],
        "final": {
            "hard_constraints": {
                "one_player": True, "one_key": True, "one_door": True,
                "enemy_count": True, "regions_one": True,
                "player_key_reachable": True, "key_door_reachable": True,
                "protocol_clean": True, "tool_clean": True, "hard_valid": True,
            },
            "metrics": {
                "players": 1, "keys": 1, "doors": 1, "enemies": 3,
                "regions": 1, "player_key": 20, "key_door": 21,
                "solution_length": 41,
            },
            "e_inf": 1, "e_sum": 1, "exact": False,
        },
        "primary_failure": "target-miss",
        "errors": [],
        "budget": {
            "semantic_actions": 8, "verifier_calls": 32, "total_tokens": 1000,
        },
        "costs": {
            "parser_errors": 0, "tool_errors": 0, "llm_calls": 1,
            "input_tokens": 100, "output_tokens": 20, "latency_ms": 10,
            "retries": 0, "semantic_actions": 0, "primitive_edits": 0,
            "evaluator_calls": 1, "verifier_calls": 0,
        },
    }
    records = []
    for taxonomy in FAILURE_TAXONOMY:
        record = deepcopy(base)
        record["episode_id"] = f"synthetic-{taxonomy}"
        record["primary_failure"] = taxonomy
        if taxonomy == "protocol":
            record["final"]["hard_constraints"]["protocol_clean"] = False
            record["costs"]["parser_errors"] = 1
        elif taxonomy == "hard-validity":
            record["final"]["hard_constraints"]["regions_one"] = False
            record["final"]["metrics"]["regions"] = 2
        elif taxonomy == "tool-effect":
            record["final"]["hard_constraints"]["tool_clean"] = False
            record["costs"]["tool_errors"] = 1
        elif taxonomy == "budget":
            record["costs"]["input_tokens"] = 980
            record["costs"]["output_tokens"] = 20

        if taxonomy in {"planning", "tool-effect", "oscillation"}:
            metrics_after = deepcopy(record["final"]["metrics"])
            metrics_before = deepcopy(metrics_after)
            if taxonomy == "planning":
                metrics_before["player_key"] = 19
                metrics_before["solution_length"] = 40
                before_ascii, after_ascii = "planning-before", "planning-after"
                expected = actual = {"player_key": 1, "key_door": 0}
                accepted = True
            elif taxonomy == "tool-effect":
                before_ascii = after_ascii = "tool-rollback"
                expected = {"player_key": 1, "key_door": 0}
                actual = {"player_key": 0, "key_door": 0}
                accepted = False
            else:
                before_ascii = after_ascii = "oscillation-state"
                expected = actual = {"player_key": 0, "key_door": 0}
                accepted = True
            record["actions"] = [{
                "index": 0,
                "semantic_action": {"name": "synthetic_edit", "parameters": {}},
                "state_before": _snapshot(before_ascii, metrics_before),
                "state_after": _snapshot(after_ascii, metrics_after),
                "expected_delta": expected,
                "actual_delta": actual,
                "accepted": accepted,
                "decision_reason": f"synthetic {taxonomy} outcome",
                "primitive_edits": 2 if taxonomy == "oscillation" else 1,
                "evaluator_calls": 1,
                "verifier_calls": 1,
                "latency_ms": 1,
            }]
            record["costs"].update({
                "semantic_actions": 1,
                "primitive_edits": record["actions"][0]["primitive_edits"],
                "evaluator_calls": 2,
                "verifier_calls": 1,
            })
        record["errors"] = [{
            "taxonomy": taxonomy,
            "code": f"SYNTHETIC_{taxonomy.upper().replace('-', '_')}",
            "stage": "synthetic-gate",
            "message": f"Semantically derived witness for {taxonomy}",
        }]
        record["final"]["hard_constraints"]["hard_valid"] = all(
            value for key, value in record["final"]["hard_constraints"].items()
            if key != "hard_valid"
        )
        records.append(record)
    return records


def run_contract_gate() -> dict[str, Any]:
    """Exercise the production config/prompt/history/evaluator/controller contract."""
    from toolusepcg.optimizer.greedy import (
        ConversationEntry,
        GreedyOptimizer,
        StepRecord,
    )
    from toolusepcg.optimizer.scoring import Scorer
    from toolusepcg.protocol import StepMessage, ToolCall
    from toolusepcg.tools.eval_stats import CalculateStatsTool

    cases: dict[str, bool] = {}
    exact_config = load_config(
        "configs/zelda.yaml",
        {
            "targets": {
                "player_key": {"mode": "target", "value": 17},
                "key_door": {"mode": "target", "value": 31},
            },
            "optimizer": {"conversation_length": 2},
        },
    )
    optimizer = GreedyOptimizer(exact_config, verbose=False)
    optimizer.initialize(seed=0)
    prompt = optimizer.build_prompt()
    cases["full_prompt_exact_no_maximize"] = (
        "exactly 17" in prompt
        and "exactly 31" in prompt
        and "~17" not in prompt
        and "~31" not in prompt
        and "maximize" not in prompt.lower()
    )

    step_record = StepRecord(
        step_num=1,
        message_type="STEP",
        metrics_before={"player_key": 16, "key_door": 32},
        metrics_after={"player_key": 17, "key_door": 31},
        accepted=True,
        accept_reason="synthetic prompt gate",
    )
    entry = ConversationEntry(
        prompt=prompt,
        raw_response='{"type":"STEP"}',
        accepted=True,
        accept_reason="synthetic prompt gate",
        message_type="STEP",
        step_record=step_record,
    )
    optimizer._conversation_history.append(entry)
    messages = optimizer._build_messages(optimizer.build_prompt())
    rendered_history = "\n".join(message["content"] for message in messages)
    cases["step_feedback_exact_no_approximation"] = (
        "target: exactly 17" in rendered_history
        and "target: exactly 31" in rendered_history
        and "target: ~" not in rendered_history
    )

    exact_result = optimizer.evaluator.evaluate(construct_zelda_corridor(17, 31))
    miss_result = optimizer.evaluator.evaluate(construct_zelda_corridor(17, 30))
    cases["real_exact_stop"] = (
        optimizer.termination.should_terminate(exact_result)
        and not optimizer.termination.should_terminate(miss_result)
    )

    # Hard validity must dominate the scalar score and every exploration mode.
    optimizer.current_result = miss_result
    optimizer.current_score = optimizer.scorer.score(miss_result)
    invalid_result = deepcopy(exact_result)
    invalid_result.valid = False
    accepted, reason = optimizer.decide_accept(invalid_result, 1)
    cases["hard_validity_acceptance_gate"] = (
        not accepted and reason == "rejected_hard_validity_regression"
    )

    # execute_step returns a tentative candidate but restores the live manager;
    # only decide_accept may later commit it.
    before = optimizer.edit_manager.level.data.copy()
    tile_name = "empty" if int(before[1, 1]) == int(ZeldaTiles.WALL) else "wall"
    candidate, _, changed = optimizer.execute_step(
        StepMessage(tool_calls=[ToolCall(
            tool_name="place_tile",
            parameters={"mode": "single", "tile_type": tile_name, "y": 1, "x": 1},
        )])
    )
    cases["transactional_rollback"] = (
        changed == 1
        and not np.array_equal(candidate.data, before)
        and np.array_equal(optimizer.edit_manager.level.data, before)
    )

    total_config = load_config(
        "configs/zelda.yaml",
        {"targets": {"solution_length": {"mode": "target", "value": 40}}},
    )
    total_exact = EvalResult(
        problem_type="zelda", valid=True,
        metrics={
            "players": 1, "keys": 1, "doors": 1, "enemies": 3,
            "regions": 1, "player_key": 10, "key_door": 30,
            "solution_length": 40,
        },
    )
    total_miss = deepcopy(total_exact)
    total_miss.metrics["key_door"] = 29
    total_miss.metrics["solution_length"] = 39
    total_tool = CalculateStatsTool(type("E", (), {"config": total_config})(), None)
    gaps, _ = total_tool._compute_zelda_gaps(total_exact)
    cases["solution_length_single_target"] = (
        Scorer(total_config).score(total_exact) > Scorer(total_config).score(total_miss)
        and gaps["solution_length"]["status"] == "ok"
    )

    zero_config = load_config(
        "configs/zelda.yaml",
        {"targets": {"enemies": {"mode": "target", "value": 0}}},
    )
    zero_eval = StatsEvaluator(zero_config, LevelFactory(zero_config).env)
    zero = zero_eval.evaluate(construct_zelda_corridor(20, 20, enemies=0))
    one = zero_eval.evaluate(construct_zelda_corridor(20, 20, enemies=1))
    cases["enemies_zero"] = zero.valid and not one.valid

    region_level = construct_zelda_corridor(20, 20)
    region_level.data[15, 15] = int(ZeldaTiles.EMPTY)
    region_result = optimizer.evaluator.evaluate(region_level)
    cases["regions_hard_invalid"] = (
        region_result.metrics["regions"] == 2 and not region_result.valid
    )
    return {
        "gate_pass": all(cases.values()),
        "passed": sum(cases.values()),
        "total": len(cases),
        "cases": cases,
    }


def run_schema_gate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate witnesses and prove representative contradictions fail closed."""
    cases: dict[str, bool] = {}
    for record in records:
        validate_episode(record)
        cases[f"witness:{record['primary_failure']}"] = True

    by_type = {record["primary_failure"]: record for record in records}
    mutations: dict[str, dict[str, Any]] = {}

    bad = deepcopy(by_type["target-miss"])
    bad["final"]["metrics"]["players"] = 2
    mutations["players_vs_one_player"] = bad

    bad = deepcopy(by_type["planning"])
    bad["actions"][0]["actual_delta"]["player_key"] = 999
    mutations["actual_delta_vs_snapshots"] = bad

    bad = deepcopy(by_type["planning"])
    bad["final"]["metrics"]["key_door"] = 22
    bad["final"]["metrics"]["solution_length"] = 42
    mutations["last_state_vs_final"] = bad

    bad = deepcopy(by_type["planning"])
    bad["costs"]["primitive_edits"] = 999
    mutations["action_vs_aggregate_cost"] = bad

    bad = deepcopy(by_type["protocol"])
    bad["final"]["hard_constraints"]["protocol_clean"] = True
    mutations["parser_errors_vs_protocol_clean"] = bad

    bad = deepcopy(by_type["planning"])
    bad["primary_failure"] = "target-miss"
    mutations["taxonomy_precedence"] = bad

    bad = deepcopy(by_type["target-miss"])
    bad["model"]["sampling"] = {}
    mutations["empty_sampling"] = bad

    bad = deepcopy(by_type["target-miss"])
    bad["model"]["revision"] = ""
    mutations["empty_model_revision"] = bad

    for name, bad_record in mutations.items():
        try:
            validate_episode(bad_record)
        except ValueError:
            cases[f"reject:{name}"] = True
        else:
            cases[f"reject:{name}"] = False
    return {
        "gate_pass": all(cases.values()),
        "passed": sum(cases.values()),
        "total": len(cases),
        "cases": cases,
        "taxonomy_precedence": list(FAILURE_TAXONOMY),
    }


def write_m0_artifacts(output_dir: str | Path) -> dict[str, Any]:
    """Generate all frozen R060 artifacts and fail closed if a gate misses."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    oracle_records, oracle_summary = evaluate_oracle_domain()
    oracle_jsonl = b"".join(
        canonical_json_bytes(record) + b"\n" for record in oracle_records
    )
    (output / "r060_a_oracle.jsonl").write_bytes(oracle_jsonl)
    (output / "r060_a_oracle_summary.json").write_bytes(
        canonical_json_bytes(oracle_summary) + b"\n"
    )

    manifest = build_target_manifest()
    manifest_gate = validate_target_manifest(manifest)
    (output / "r060_c_target_manifest.json").write_bytes(
        canonical_json_bytes(manifest) + b"\n"
    )

    schema = episode_schema()
    (output / "r060_d_episode_schema.json").write_bytes(
        canonical_json_bytes(schema) + b"\n"
    )
    synthetic = synthetic_failure_records()
    schema_gate = run_schema_gate(synthetic)
    synthetic_jsonl = b"".join(
        canonical_json_bytes(record) + b"\n" for record in synthetic
    )
    (output / "r060_d_synthetic_failures.jsonl").write_bytes(synthetic_jsonl)

    contract_gate = run_contract_gate()
    (output / "r060_b_contract_gate.json").write_bytes(
        canonical_json_bytes(contract_gate) + b"\n"
    )
    (output / "r060_c_manifest_gate.json").write_bytes(
        canonical_json_bytes(manifest_gate) + b"\n"
    )
    (output / "r060_d_schema_gate.json").write_bytes(
        canonical_json_bytes(schema_gate) + b"\n"
    )

    component_gates = {
        "R060-A": bool(oracle_summary["gate_pass"]),
        "R060-B": bool(contract_gate["gate_pass"]),
        "R060-C": bool(manifest_gate["gate_pass"]),
        "R060-D": bool(schema_gate["gate_pass"]),
    }

    summary = {
        "milestone": "M0.1-contract-closure",
        "oracle": oracle_summary,
        "manifest_sha256": manifest["manifest_sha256"],
        "split_sizes": manifest["split_sizes"],
        "episode_schema_version": EPISODE_SCHEMA_VERSION,
        "episode_schema_sha256": sha256_json(schema),
        "failure_taxonomy": list(FAILURE_TAXONOMY),
        "synthetic_failure_records": len(synthetic),
        "synthetic_failures_sha256": hashlib.sha256(synthetic_jsonl).hexdigest(),
        "component_gates": component_gates,
        "gate_pass": all(component_gates.values()),
    }
    (output / "r060_m0_summary.json").write_bytes(
        canonical_json_bytes(summary) + b"\n"
    )
    if not summary["gate_pass"]:
        raise RuntimeError(f"M0 aggregate gate failed closed: {component_gates}")
    return summary


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Regenerate the 961-target Zelda oracle and contract gates; no model needed.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_m0_artifacts(args.output), indent=2))


if __name__ == "__main__":
    main()
