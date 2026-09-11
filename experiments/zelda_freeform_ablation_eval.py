"""Aggregate matched Zelda free-form arms and paired deltas against full."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


def _matched_records(run_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads((run_dir / "evaluation.json").read_text())
    return [row for row in payload["records"] if row["condition"] == "matched"]


def _operational_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    for record in records:
        summary_path = Path(str(record["run_dir"])) / "summary.json"
        if not summary_path.is_file():
            rows.append({"accepted": 0, "rejected": 0, "calls": 0, "output_tokens": 0})
            continue
        summary = json.loads(summary_path.read_text())
        trace_path = summary_path.parent / "trace.jsonl"
        calls = 0
        if trace_path.is_file():
            for raw in trace_path.read_text().splitlines():
                event = json.loads(raw)
                if event.get("event") == "step":
                    calls += len((event.get("parsed_message") or {}).get("tool_calls") or [])
        rows.append(
            {
                "accepted": float(summary.get("accepted_steps") or 0),
                "rejected": float(summary.get("rejected_steps") or 0),
                "calls": float(calls),
                "output_tokens": float(summary.get("total_output_tokens") or 0),
            }
        )
    return {
        f"mean_{key}": mean(row[key] for row in rows) if rows else 0.0
        for key in ("accepted", "rejected", "calls", "output_tokens")
    }


def summarize_arm(label: str, run_dir: Path) -> dict[str, Any]:
    records = _matched_records(run_dir)
    by_tier: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_tier[int(row["tier"])].append(row)

    def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(rows),
            "mean_programmatic_score": mean(float(row["programmatic_score"]) for row in rows),
            "strict_successes": sum(bool(row["programmatic_strict_success"]) for row in rows),
            "functional_successes": sum(bool(row["functional"]["success"]) for row in rows),
            "complete": sum(row["status"] == "complete" for row in rows),
        }

    arm = {
        "label": label,
        "run_dir": str(run_dir.resolve()),
        "overall": summarize(records),
        "by_tier": {str(tier): summarize(rows) for tier, rows in sorted(by_tier.items())},
        "operational": _operational_stats(records),
    }
    summary = json.loads((run_dir / "summary.json").read_text())
    arm["overall"]["mean_paired_delta_vs_null"] = summary["overall"][
        "mean_programmatic_paired_delta"
    ]
    return arm


def paired_against_full(
    full_records: Sequence[Mapping[str, Any]],
    treatment_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def key(row: Mapping[str, Any]) -> tuple[str, int]:
        return str(row["case_id"]), int(row["seed"])

    full = {key(row): row for row in full_records}
    treatment = {key(row): row for row in treatment_records}
    if full.keys() != treatment.keys():
        raise ValueError("ablation arms do not contain the same matched case-seed keys")
    rows = []
    for item in sorted(full):
        base = full[item]
        other = treatment[item]
        rows.append(
            {
                "case_id": item[0],
                "seed": item[1],
                "tier": int(base["tier"]),
                "score_delta": float(other["programmatic_score"]) - float(base["programmatic_score"]),
                "strict_full": bool(base["programmatic_strict_success"]),
                "strict_ablation": bool(other["programmatic_strict_success"]),
                "functional_full": bool(base["functional"]["success"]),
                "functional_ablation": bool(other["functional"]["success"]),
            }
        )
    return {
        "pairs": rows,
        "mean_score_delta": mean(row["score_delta"] for row in rows),
        "positive_score_pairs": sum(row["score_delta"] > 0 for row in rows),
        "negative_score_pairs": sum(row["score_delta"] < 0 for row in rows),
        "strict_gains": sum(not row["strict_full"] and row["strict_ablation"] for row in rows),
        "strict_losses": sum(row["strict_full"] and not row["strict_ablation"] for row in rows),
        "functional_gains": sum(not row["functional_full"] and row["functional_ablation"] for row in rows),
        "functional_losses": sum(row["functional_full"] and not row["functional_ablation"] for row in rows),
    }


def aggregate(arms: Mapping[str, Path], full_label: str = "full") -> dict[str, Any]:
    if full_label not in arms:
        raise ValueError(f"missing full arm {full_label!r}")
    arm_summaries = [summarize_arm(label, path) for label, path in arms.items()]
    full_records = _matched_records(arms[full_label])
    comparisons = {
        label: paired_against_full(full_records, _matched_records(path))
        for label, path in arms.items()
        if label != full_label
    }
    return {
        "schema_version": "zelda_freeform_ablation_eval.v1",
        "full_label": full_label,
        "arms": arm_summaries,
        "paired_against_full": comparisons,
    }


def _parse_arm(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("arm must be LABEL=RUN_DIR")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", type=_parse_arm, required=True)
    parser.add_argument("--full-label", default="full")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = dict(args.arm)
    payload = aggregate(arms, full_label=args.full_label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
