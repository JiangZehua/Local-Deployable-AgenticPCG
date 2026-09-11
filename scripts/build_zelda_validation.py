#!/usr/bin/env python3
"""Build the frozen 120-case Zelda validation subset for R064L."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.r061_qwen38_m1 import (
    HARNESS_COMMIT,
    MAX_OUTPUT_TOKENS,
    MAX_SEMANTIC_ACTIONS,
    MAX_VERIFIER_CALLS,
    GreedyOptimizer,
    _exact_config,
    _sha256_text,
    canonical_json_bytes,
)


MANIFEST = ROOT / "benchmarks/zelda/targets.json"
EXPECTED_MANIFEST_SHA256 = (
    "b7922c8864707c89a317737d800eb58d567b9d3163ca3ecd4d5ee5846e7f96ed"
)
SELECTION_SEED = 64120


def build() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    if manifest["manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("frozen Zelda target manifest hash drift")
    selected = [row for row in manifest["targets"] if row["split"] == "validation"]
    if len(selected) != 120 or len({row["target_id"] for row in selected}) != 120:
        raise RuntimeError("validation split must contain 120 unique targets")

    cases = []
    for index, entry in enumerate(selected):
        seed = SELECTION_SEED + index
        optimizer = GreedyOptimizer(
            _exact_config(entry["player_key"], entry["key_door"]), verbose=False
        )
        optimizer.initialize(seed=seed)
        cases.append(
            {
                "index": index,
                "target_id": entry["target_id"],
                "player_key": entry["player_key"],
                "key_door": entry["key_door"],
                "tags": entry["tags"],
                "seed": seed,
                "initialization_sha256": _sha256_text(
                    optimizer.current_level.to_string()
                ),
            }
        )

    payload = {
        "version": "r064l-frozen-validation120-v1",
        "source_manifest_sha256": manifest["manifest_sha256"],
        "selection_seed": SELECTION_SEED,
        "harness_commit": HARNESS_COMMIT,
        "init_strategy": "empty",
        "budgets": {
            "semantic_actions": MAX_SEMANTIC_ACTIONS,
            "verifier_calls": MAX_VERIFIER_CALLS,
            "max_output_tokens_per_call": MAX_OUTPUT_TOKENS,
            "max_tool_calls_per_action": 20,
        },
        "requirements": {"split": "validation", "all_frozen_targets": 120},
        "observed_counts": manifest["coverage"]["validation"],
        "cases": cases,
    }
    return {
        "subset_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        **payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
