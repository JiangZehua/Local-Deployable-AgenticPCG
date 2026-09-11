"""R100-B replay audit for within-trial duplicate SMB candidate levels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "r100-smb-duplicate-replay-v1"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def audit(sample_paths: list[Path]) -> dict[str, Any]:
    semantic_steps = 0
    duplicate_steps = 0
    affected_trials = 0
    samples: list[dict[str, Any]] = []
    for sample_path in sample_paths:
        payload = json.loads(sample_path.read_text())
        sample_duplicates = 0
        sample_semantic_steps = 0
        sample_affected = 0
        duplicate_rows: list[dict[str, Any]] = []
        for record in payload["records"]:
            seen: dict[str, int] = {}
            trial_affected = False
            for step_index, step in enumerate(record["steps"], start=1):
                if step.get("message_type") != "STEP":
                    continue
                sample_semantic_steps += 1
                level_text = step.get("level_after", "")
                digest = _sha256(level_text)
                if digest in seen:
                    sample_duplicates += 1
                    trial_affected = True
                    duplicate_rows.append(
                        {
                            "case_id": record["case_id"],
                            "step_index": step_index,
                            "first_semantic_step_index": seen[digest],
                            "level_sha256": digest,
                            "accepted": bool(step.get("accepted")),
                        }
                    )
                else:
                    seen[digest] = step_index
            if trial_affected:
                sample_affected += 1
        semantic_steps += sample_semantic_steps
        duplicate_steps += sample_duplicates
        affected_trials += sample_affected
        samples.append(
            {
                "path": str(sample_path),
                "semantic_steps": sample_semantic_steps,
                "duplicate_steps": sample_duplicates,
                "affected_trials": sample_affected,
                "duplicates": duplicate_rows,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "definition": (
            "A duplicate STEP has a nonempty recorded level_after identical to "
            "an earlier semantic STEP level_after in the same trial."
        ),
        "guard_policy": {
            "identity": "sha256(canonical Level.to_string())",
            "online_action": "reject candidate, skip evaluator, emit explicit feedback",
            "retry_bound": "duplicate consumes its normal semantic STEP; no extra LLM call",
        },
        "sample_paths": [str(path) for path in sample_paths],
        "aggregate": {
            "trials": sum(
                len(json.loads(path.read_text())["records"])
                for path in sample_paths
            ),
            "semantic_steps": semantic_steps,
            "duplicate_steps": duplicate_steps,
            "duplicate_rate": (
                duplicate_steps / semantic_steps if semantic_steps else 0.0
            ),
            "affected_trials": affected_trials,
        },
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(args.sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
