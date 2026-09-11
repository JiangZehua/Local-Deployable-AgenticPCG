"""Score both tool access and the paired execution-loop ablation."""
import argparse
import json
from pathlib import Path
import yaml
from experiments.paper_m7_fixed_rng36 import evaluate as evaluate_tools, _strict_reconnect_equivalent, _costs
from experiments.paper_m4_freeform_routing import evaluate_run


def evaluate_loop(run_root: Path, output: Path) -> dict:
    manifest = yaml.safe_load(Path("configs/paper_m7_fixed_rng36.yaml").read_text())
    records = []
    for task in manifest["tasks"]:
        for arm in ("all_tools", "no_evaluation_loop"):
            run_dir = run_root / task["task_id"] / arm
            row = evaluate_run(task, arm, run_dir)
            row["costs"] = _costs(run_dir, row["routing"]["calls"])
            row["routing"]["strict_reconnect_audit"] = _strict_reconnect_equivalent(task, run_dir, row)
            records.append(row)
    by_arm = {}
    for arm in ("all_tools", "no_evaluation_loop"):
        rows = [r for r in records if r["arm"] == arm]
        by_arm[arm] = {"planned": 18, "functional_valid": sum(r["structure"]["functional_valid"] for r in rows),
                       "structural_success": sum(r["proxy_outcome_success"] for r in rows),
                       "tool_calls": sum(r["costs"]["trace_tool_calls"] for r in rows),
                       "tokens": sum(r["costs"]["total_tokens"] for r in rows)}
    result = {"interpretation": "descriptive; three paraphrases per family share map and RNG seed",
              "aggregate_by_arm": by_arm, "records": records}
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["tools", "tool-loop"], required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.suite == "tools":
        evaluate_tools(Path("configs/paper_m7_fixed_rng36.yaml"), args.run_root, args.output)
    else:
        evaluate_loop(args.run_root, args.output)


if __name__ == "__main__":
    main()
