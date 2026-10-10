"""PAPER-M8b: ceiling-free patch library certified with the takeoff audit.

Same procedure as ``smb_patch_library`` (fixed parameter grid x fixed
composition contexts, no LLM, no target levels), with three differences:
patches have no ceilings and a one-row floor; every context must also have
zero hidden takeoffs (``toolusepcg.smb_takeoff_audit``); and stomps are
certified in two roles, because ceiling-free stomps depend on where they sit:

* ``stomp_mid``: followed by more level (jumps and an exit jump);
* ``stomp_end``: the last patch, also absorbing the goal jump (j may be 0).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.smb_patch_library import _signature, event_kill_counts
from toolusepcg.smb_patches import Patch, WidthEvaluatorPool, assemble, level_sha256
from toolusepcg.smb_takeoff_audit import takeoffs_clean


SCHEMA_VERSION = "smb-patch-library-v2"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks/smb/patches/patch_library_v2.json"
REPEATS = 2
PAD = "flat"

LEAD = Patch.make("flat", width=4)
SEP = Patch.make("flat", width=4)
JUMP0 = Patch.make("obstacle15", width=6, height=1, x=3)
EXIT0 = Patch.make("obstacle15", width=8, height=1, x=7)
COIN0 = Patch.make("coins15", count=2, spacing=2, row=14)

ROLES = ("exit", "jump", "coins", "sep", "stomp_mid", "stomp_end")


def candidates() -> dict[str, list[Patch]]:
    jumps = [JUMP0] + [
        Patch.make("obstacle15", width=w, height=h, x=x)
        for w in (6, 8) for h in (1, 2) for x in (2, 3, 4) if (w, h, x) != (6, 1, 3)
    ] + [
        Patch.make("gap15", width=w, gap=g, x=x)
        for w in (6, 8) for g in (2, 3) for x in (2, 3) if x + g <= w - 1
    ]
    exits = [EXIT0] + [
        Patch.make("obstacle15", width=8, height=h, x=x) for h in (1, 2) for x in (5, 6, 7) if (h, x) != (1, 7)
    ] + [Patch.make("gap15", width=8, gap=g, x=8 - 1 - g - off) for g in (2, 3) for off in (0, 1)]
    coins = [Patch.make("coins15", count=n, spacing=s, row=r)
             for n in range(1, 6) for s in (2, 1, 3) for r in (14, 13)]
    seps = [SEP] + [Patch.make("flat", width=w) for w in (2, 6, 8, 12)]
    stomps = [Patch.make("stomp_cluster", count=n, pit=p, gap=g, runway=r)
              for n in (1, 2, 3, 4) for p in (2, 3, 4) for g in (1, 2, 3) for r in (2, 4, 8, 12, 16)]
    stomps += [Patch.make("stomp_pillar", runway=r, pit=p, pillar=w, height=h)
               for r in (2, 4, 8) for p in (1, 2, 3) for w in (1, 2) for h in (1, 2)]
    return {"exit": exits, "jump": jumps, "coins": coins, "sep": seps,
            "stomp_mid": list(stomps), "stomp_end": list(stomps)}


def stomp_count(patch: Patch) -> int:
    return patch.p["count"] if patch.family == "stomp_cluster" else 1


def contexts(role: str, patch: Patch) -> list[tuple[str, list[Patch], tuple[int, int, int]]]:
    if role == "exit":
        return [("alone", [LEAD, patch], (0, 0, 1)),
                ("after_jump", [LEAD, JUMP0, SEP, patch], (0, 0, 2)),
                ("after_coins", [LEAD, COIN0, SEP, patch], (0, 2, 1))]
    if role == "jump":
        return [("before_exit", [LEAD, patch, SEP, EXIT0], (0, 0, 2)),
                ("repeated", [LEAD, patch, SEP, patch, SEP, EXIT0], (0, 0, 3)),
                ("after_coins", [LEAD, COIN0, SEP, patch, SEP, EXIT0], (0, 2, 2))]
    if role == "coins":
        n = patch.p["count"]
        return [("before_exit", [LEAD, patch, SEP, EXIT0], (0, n, 1)),
                ("before_jump", [LEAD, patch, SEP, JUMP0, SEP, EXIT0], (0, n, 2)),
                ("after_jump", [LEAD, JUMP0, SEP, patch, SEP, EXIT0], (0, n, 2))]
    if role == "sep":
        return [("jumps", [LEAD, JUMP0, patch, JUMP0, patch, EXIT0], (0, 0, 3)),
                ("coins", [LEAD, COIN0, patch, EXIT0], (0, 2, 1))]
    k = stomp_count(patch)
    if role == "stomp_mid":
        return [("before_exit", [LEAD, patch, SEP, EXIT0], (k, 0, 1)),
                ("before_jump", [LEAD, patch, SEP, JUMP0, SEP, EXIT0], (k, 0, 2)),
                ("after_jump", [LEAD, JUMP0, SEP, patch, SEP, EXIT0], (k, 0, 2)),
                ("after_coins", [LEAD, COIN0, SEP, patch, SEP, EXIT0], (k, 2, 1))]
    if role == "stomp_end":
        return [("alone", [LEAD, patch], (k, 0, 0)),
                ("after_jump", [LEAD, JUMP0, SEP, patch], (k, 0, 1)),
                ("after_two_jumps", [LEAD, JUMP0, SEP, JUMP0, SEP, patch], (k, 0, 2)),
                ("after_coins", [LEAD, COIN0, SEP, patch], (k, 2, 0))]
    raise ValueError(role)


def check_level(pool: WidthEvaluatorPool, plan: Sequence[Patch], expected: tuple[int, int, int],
                repeats: int = REPEATS) -> dict[str, Any]:
    level = assemble(plan, pad=PAD)
    results = [pool.evaluate_audited(level) for _ in range(repeats)]
    first = results[0]
    observed = tuple(int(first.metrics.get(k, -1)) for k in ("enemies", "coins", "jumps"))
    kills = event_kill_counts(first)
    audit = first.info["takeoff_audit"]
    deterministic = len({_signature(r) for r in results}) == 1
    passed = bool(first.valid and float(first.metrics.get("complete", 0.0)) == 1.0
                  and observed == tuple(expected) and kills["stomp_kills"] == expected[0]
                  and kills["fall_kills"] == 0 and takeoffs_clean(first) and deterministic)
    return {"passed": passed, "expected": list(expected), "observed": list(observed),
            "valid": bool(first.valid), "deterministic": deterministic, **kills,
            "takeoffs": audit["takeoffs"], "hidden_takeoffs": audit["hidden_takeoffs"],
            "width": level.width, "level_sha256": level_sha256(level)}


def _certify_one(job: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    role, payload = job
    patch = Patch.from_dict(payload)
    pool = WidthEvaluatorPool()
    checks = []
    planned = contexts(role, patch)
    for name, plan, expected in planned:
        row = check_level(pool, plan, expected)
        row["context"] = name
        checks.append(row)
        if not row["passed"]:
            break
    return {"role": role, "patch": patch.as_dict(), "token": patch.token(),
            "certified": len(checks) == len(planned) and all(r["passed"] for r in checks),
            "checks": checks}


def library_sha256(certified: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    return hashlib.sha256(json.dumps(certified, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
    for role, canonical in (("sep", SEP), ("jump", JUMP0), ("exit", EXIT0)):
        if canonical.as_dict() not in certified[role]:
            raise RuntimeError(f"canonical {role} companion failed certification")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "procedure": "ceiling-free fixed grid x fixed contexts; A* exact, stomp-only kills, no fall kills, "
                     f"zero hidden takeoffs, {REPEATS} identical repeats",
        "pad": PAD,
        "lead": LEAD.as_dict(),
        "certified": certified,
        "library_sha256": library_sha256(certified),
        "counts": {role: {"candidates": len(grid[role]), "certified": len(certified[role])} for role in ROLES},
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
