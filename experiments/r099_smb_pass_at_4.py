"""Aggregate four independent dynamic SMB samples into pass@4 metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.r061_qwen38_m1 import _write_json
from experiments.r080_smb_astar_probe import TARGET_METRICS


def aggregate(sample_paths: list[Path]) -> dict[str, Any]:
    if len(sample_paths) != 4:
        raise ValueError("pass@4 requires exactly four sample result files")
    samples = [json.loads(path.read_text()) for path in sample_paths]
    if not all(sample.get("execution_complete") for sample in samples):
        raise ValueError("all pass@4 samples must be complete")
    if not all(sample.get("contract_gate_pass") for sample in samples):
        raise ValueError("all pass@4 samples must pass the contract gate")

    by_sample = [
        {row["case_id"]: row for row in sample["records"]} for sample in samples
    ]
    case_ids = sorted(by_sample[0])
    if len(case_ids) != 29 or any(sorted(rows) != case_ids for rows in by_sample):
        raise ValueError("pass@4 samples must contain the same frozen 29 dev cases")

    cases = []
    for case_id in case_ids:
        rows = [sample[case_id] for sample in by_sample]
        targets = rows[0]["targets"]
        if any(row["targets"] != targets for row in rows[1:]):
            raise ValueError(f"target mismatch for {case_id}")
        exact_samples = [index for index, row in enumerate(rows) if row["exact"]]
        metric_pass = {
            metric: any(float(row["metric_gaps"][metric]) == 0.0 for row in rows)
            for metric in TARGET_METRICS
        }
        cases.append(
            {
                "case_id": case_id,
                "targets": targets,
                "pass_at_4": bool(exact_samples),
                "exact_sample_indices": exact_samples,
                "exact_count": len(exact_samples),
                "hard_valid_count": sum(bool(row["hard_valid"]) for row in rows),
                "per_metric_pass_at_4": metric_pass,
                "best_e_inf": min(float(row["e_inf"]) for row in rows),
                "best_e_sum": min(float(row["e_sum"]) for row in rows),
                "unique_final_levels": len(
                    {row["final_level_sha256"] for row in rows}
                ),
                "samples": [
                    {
                        "sample_index": index,
                        "exact": bool(row["exact"]),
                        "hard_valid": bool(row["hard_valid"]),
                        "e_inf": float(row["e_inf"]),
                        "e_sum": float(row["e_sum"]),
                        "metrics": {
                            "enemies": row["final_metrics"]["enemies"],
                            "coins": row["final_metrics"]["coins"],
                            "jumps": row["final_metrics"]["jumps"],
                        },
                    }
                    for index, row in enumerate(rows)
                ],
            }
        )

    total_trials = 4 * len(case_ids)
    total_exact = sum(
        bool(row["exact"]) for sample in by_sample for row in sample.values()
    )
    pass1 = total_exact / total_trials
    totals = {
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 0,
    }
    for sample in by_sample:
        for row in sample.values():
            totals["llm_calls"] += int(row["costs"]["llm_calls"])
            totals["input_tokens"] += int(row["costs"]["input_tokens"])
            totals["output_tokens"] += int(row["costs"]["output_tokens"])
            totals["latency_ms"] += int(row["costs"]["latency_ms"])

    aggregate_row = {
        "targets": len(case_ids),
        "samples_per_target": 4,
        "trials": total_trials,
        "single_trial_exact": total_exact,
        "pass_at_1_pooled": pass1,
        "pass_at_4": sum(row["pass_at_4"] for row in cases) / len(cases),
        "pass_at_4_targets": sum(row["pass_at_4"] for row in cases),
        "iid_plugin_pass_at_4": 1.0 - (1.0 - pass1) ** 4,
        "per_metric_pass_at_4": {
            metric: sum(row["per_metric_pass_at_4"][metric] for row in cases)
            / len(cases)
            for metric in TARGET_METRICS
        },
        "all_four_hard_valid": sum(
            row["hard_valid_count"] == 4 for row in cases
        ),
        "mean_best_e_inf": sum(row["best_e_inf"] for row in cases) / len(cases),
        "mean_best_e_sum": sum(row["best_e_sum"] for row in cases) / len(cases),
        "mean_unique_final_levels": sum(
            row["unique_final_levels"] for row in cases
        ) / len(cases),
        "costs": totals,
    }
    return {
        "execution_complete": True,
        "contract_gate_pass": True,
        "definition": (
            "A frozen dev target passes at 4 iff at least one of four independent "
            "temperature-0.7 samples is hard-valid and jointly exact."
        ),
        "sample_paths": [str(path) for path in sample_paths],
        "sample_aggregates": [sample["aggregate"] for sample in samples],
        "aggregate": aggregate_row,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = aggregate(args.sample)
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
