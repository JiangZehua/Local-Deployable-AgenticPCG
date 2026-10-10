"""Reproducible enumeration and A* certification of the SMB patch library.

Development procedure (no LLM, no target levels): every candidate patch in a
fixed parameter grid is placed into a small set of composition contexts built
from canonical companions.  A candidate is certified for a role only if every
context yields exactly the expected (enemies, coins, jumps) with a valid,
complete, deterministic A* playthrough, only stomp kills and no fall kills.
Rejected candidates are kept in the output with the failing context.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from toolusepcg.smb_contract import FALL_KILL_EVENT, STOMP_KILL_EVENT
from toolusepcg.smb_patches import (
    Patch,
    WidthEvaluatorPool,
    assemble,
    level_sha256,
)


SCHEMA_VERSION = "smb-patch-library-v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks/smb/patches/patch_library.json"
REPEATS = 2

LEAD = Patch.make("corridor", width=4)
SEP = Patch.make("corridor", width=4)
TAIL = Patch.make("corridor", width=4)
STOMP0 = Patch.make("stomp", width=10, enemy_x=6)
JUMP0 = Patch.make("obstacle", width=6, height=1, x=3)
EXIT0 = Patch.make("obstacle", width=8, height=1, x=7)

COIN_ROLES = ("coins_after_lead", "coins_after_stomps", "coins_before_exit")
ROLES = ("exit", "stomp", "jump", *COIN_ROLES, "sep", "tail")


def candidates() -> dict[str, list[Patch]]:
    """Fixed candidate grid; canonical companions are listed first."""
    stomps = [STOMP0] + [
        Patch.make("stomp", width=w, enemy_x=x)
        for w in (8, 10, 12) for x in (4, 6, 8)
        if x <= w - 2 and (w, x) != (10, 6)
    ]
    jumps = [JUMP0] + [
        Patch.make("obstacle", width=w, height=h, x=x)
        for w in (6, 8) for h in (1, 2, 3) for x in (2, 3, 4)
        if (w, h, x) != (6, 1, 3)
    ] + [
        Patch.make("gap", width=w, gap=g, x=x)
        for w in (6, 8) for g in (1, 2, 3) for x in (2, 3)
        if x + g <= w - 1
    ]
    exits = [EXIT0] + [
        Patch.make("obstacle", width=8, height=h, x=x)
        for h in (1, 2) for x in (5, 6, 7) if (h, x) != (1, 7)
    ] + [
        Patch.make("gap", width=8, gap=g, x=8 - 1 - g - off)
        for g in (2, 3) for off in (0, 1)
    ]
    coins = [
        Patch.make("coins", count=n, spacing=s)
        for n in range(1, 6) for s in (2, 1, 3)
    ]
    seps = [SEP] + [Patch.make("corridor", width=w) for w in (2, 6, 8)]
    tails = [TAIL] + [Patch.make("corridor", width=w) for w in (8, 2)]
    return {"exit": exits, "stomp": stomps, "jump": jumps,
            **{role: list(coins) for role in COIN_ROLES},
            "sep": seps, "tail": tails}


def contexts(role: str, patch: Patch) -> list[tuple[str, list[Patch], tuple[int, int, int]]]:
    """Composition contexts and expected (enemies, coins, jumps)."""
    if role == "exit":
        return [
            ("alone", [LEAD, patch], (0, 0, 1)),
            ("after_stomp", [LEAD, STOMP0, SEP, patch], (1, 0, 1)),
            ("after_jump", [LEAD, JUMP0, SEP, patch], (0, 0, 2)),
            ("after_two_stomps_jump", [LEAD, STOMP0, SEP, STOMP0, SEP, JUMP0, SEP, patch], (2, 0, 2)),
        ]
    if role == "stomp":
        return [
            ("alone_tail", [LEAD, patch, SEP, TAIL], (1, 0, 0)),
            ("pair_tail", [LEAD, patch, SEP, patch, SEP, TAIL], (2, 0, 0)),
            ("before_exit", [LEAD, patch, SEP, EXIT0], (1, 0, 1)),
            ("before_jump_exit", [LEAD, patch, SEP, JUMP0, SEP, EXIT0], (1, 0, 2)),
        ]
    if role == "jump":
        return [
            ("before_exit", [LEAD, patch, SEP, EXIT0], (0, 0, 2)),
            ("after_stomp", [LEAD, STOMP0, SEP, patch, SEP, EXIT0], (1, 0, 2)),
            ("repeated", [LEAD, patch, SEP, patch, SEP, EXIT0], (0, 0, 3)),
        ]
    if role in COIN_ROLES:
        # A coin block is always [coins, SEP]; its position is the role.
        n = patch.p["count"]
        block = [patch, SEP]
        if role == "coins_after_lead":
            return [
                ("tail", [LEAD, *block, TAIL], (0, n, 0)),
                ("stomp_tail", [LEAD, *block, STOMP0, SEP, TAIL], (1, n, 0)),
                ("jump_exit", [LEAD, *block, JUMP0, SEP, EXIT0], (0, n, 2)),
                ("stomp_jump_exit", [LEAD, *block, STOMP0, SEP, JUMP0, SEP, EXIT0], (1, n, 2)),
            ]
        if role == "coins_after_stomps":
            return [
                ("tail", [LEAD, STOMP0, SEP, *block, TAIL], (1, n, 0)),
                ("exit", [LEAD, STOMP0, SEP, *block, EXIT0], (1, n, 1)),
                ("jump_exit", [LEAD, STOMP0, SEP, *block, JUMP0, SEP, EXIT0], (1, n, 2)),
            ]
        return [
            ("jump", [LEAD, JUMP0, SEP, *block, EXIT0], (0, n, 2)),
            ("stomp_jump", [LEAD, STOMP0, SEP, JUMP0, SEP, *block, EXIT0], (1, n, 2)),
            ("stomp_jumps", [LEAD, STOMP0, SEP, JUMP0, SEP, JUMP0, SEP, *block, EXIT0], (1, n, 3)),
        ]
    if role == "sep":
        return [
            ("stomps_jumps", [LEAD, STOMP0, patch, STOMP0, patch, JUMP0, patch, EXIT0], (2, 0, 2)),
            ("stomps_tail", [LEAD, STOMP0, patch, STOMP0, patch, TAIL], (2, 0, 0)),
        ]
    if role == "tail":
        return [
            ("alone", [LEAD, patch], (0, 0, 0)),
            ("after_stomps", [LEAD, STOMP0, SEP, STOMP0, SEP, patch], (2, 0, 0)),
        ]
    raise ValueError(role)


def _signature(result: Any) -> str:
    trace = (result.info or {}).get("normalized_event_trace", [])
    payload = {
        "valid": bool(result.valid),
        "metrics": {k: result.metrics.get(k) for k in ("complete", "enemies", "coins", "jumps")},
        "trace": trace,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def event_kill_counts(result: Any) -> dict[str, int]:
    types = [int(e.get("event_type", e.get("type", -1)))
             for e in (result.info or {}).get("game_events", [])]
    return {"stomp_kills": sum(t == STOMP_KILL_EVENT for t in types),
            "fall_kills": sum(t == FALL_KILL_EVENT for t in types)}


def check_level(pool: WidthEvaluatorPool, plan: Sequence[Patch],
                expected: tuple[int, int, int], repeats: int = REPEATS) -> dict[str, Any]:
    level = assemble(plan)
    results = [pool.evaluate(level) for _ in range(repeats)]
    first = results[0]
    observed = tuple(int(first.metrics.get(k, -1)) for k in ("enemies", "coins", "jumps"))
    kills = event_kill_counts(first)
    deterministic = len({_signature(r) for r in results}) == 1
    passed = bool(
        first.valid
        and float(first.metrics.get("complete", 0.0)) == 1.0
        and observed == tuple(expected)
        and kills["stomp_kills"] == expected[0]
        and kills["fall_kills"] == 0
        and deterministic
    )
    return {
        "passed": passed,
        "expected": list(expected),
        "observed": list(observed),
        "valid": bool(first.valid),
        "deterministic": deterministic,
        **kills,
        "width": level.width,
        "level_sha256": level_sha256(level),
    }


def _certify_one(job: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    role, payload = job
    patch = Patch.from_dict(payload)
    pool = WidthEvaluatorPool()
    checks = []
    for name, plan, expected in contexts(role, patch):
        row = check_level(pool, plan, expected)
        row["context"] = name
        checks.append(row)
        if not row["passed"]:
            break
    return {"role": role, "patch": patch.as_dict(), "token": patch.token(),
            "certified": all(row["passed"] for row in checks) and
            len(checks) == len(contexts(role, patch)),
            "checks": checks}


def library_sha256(certified: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    return hashlib.sha256(
        json.dumps(certified, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_library(workers: int = 1) -> dict[str, Any]:
    grid = candidates()
    jobs = [(role, patch.as_dict()) for role in ROLES for patch in grid[role]]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_certify_one, jobs, chunksize=1))
    else:
        rows = [_certify_one(job) for job in jobs]
    certified: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLES}
    for row in rows:
        if row["certified"]:
            certified[row["role"]].append(row["patch"])
    canonical = {"lead": LEAD, "sep": SEP, "tail": TAIL, "stomp": STOMP0,
                 "jump": JUMP0, "exit": EXIT0}
    for role in ("sep", "tail", "stomp", "jump", "exit"):
        if canonical[role].as_dict() not in certified[role]:
            raise RuntimeError(f"canonical {role} companion failed certification")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "procedure": "fixed parameter grid x fixed composition contexts; A* exact, "
                     f"stomp-only kills, no fall kills, {REPEATS} identical repeats",
        "lead": LEAD.as_dict(),
        "certified": certified,
        "library_sha256": library_sha256(certified),
        "counts": {role: {"candidates": len(grid[role]),
                          "certified": len(certified[role])} for role in ROLES},
        "evaluator_calls": sum(len(r["checks"]) for r in rows) * REPEATS,
        "rows": rows,
    }


def load_library(path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if library_sha256(payload["certified"]) != payload["library_sha256"]:
        raise RuntimeError(f"patch library hash drift: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    library = build_library(workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(library, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "counts": library["counts"],
                      "library_sha256": library["library_sha256"],
                      "evaluator_calls": library["evaluator_calls"]}, indent=2))


if __name__ == "__main__":
    main()
