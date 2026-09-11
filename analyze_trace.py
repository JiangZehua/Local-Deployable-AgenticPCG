#!/usr/bin/env python3
"""Analyze trace.jsonl files from ToolUsePCG optimization runs."""

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRecord:
    tool_name: str
    category: str
    parameters: dict
    success: bool
    tiles_changed: int
    step_num: int
    step_accepted: bool


@dataclass
class StepRecord:
    step_num: int
    accepted: bool
    accept_reason: str
    score_before: float
    score_after: float
    num_tiles_changed: int
    change_ratio: float | None
    tool_calls: list[ToolCallRecord]


@dataclass
class RunAnalysis:
    run_dir: str
    steps: list[StepRecord]
    final_score: float | None
    grid_size: tuple[int, int] | None = None  # (rows, cols)


# ---------------------------------------------------------------------------
# Tool categorization
# ---------------------------------------------------------------------------

_LEGACY_MAP = {
    "place_single_tile": "place_tile(single)",
    "place_line": "place_tile(line)",
    "place_patch": "place_tile(rect)",
}


def categorize_tool(tool_name: str, parameters: dict) -> str:
    if tool_name in _LEGACY_MAP:
        return _LEGACY_MAP[tool_name]
    if tool_name == "place_tile":
        mode = parameters.get("mode", "single")
        # Normalize non-standard modes (player, key, door, enemy, etc.) to "single"
        if mode not in ("single", "line", "rect"):
            mode = "single"
        return f"place_tile({mode})"
    return tool_name


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_grid_size(level_str: str) -> tuple[int, int] | None:
    """Extract (rows, cols) from a level string."""
    if not level_str:
        return None
    rows = level_str.strip().split("\n")
    if not rows:
        return None
    return len(rows), len(rows[0])


def parse_trace(trace_path: Path) -> RunAnalysis:
    steps: list[StepRecord] = []
    final_score: float | None = None
    grid_size: tuple[int, int] | None = None

    with open(trace_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"  Warning: malformed JSON on line {line_num}, skipping", file=sys.stderr)
                continue

            event = entry.get("event")

            if event == "initial_level":
                level_str = entry.get("level", "")
                grid_size = _parse_grid_size(level_str)
                continue

            if event == "summary":
                final_score = entry.get("final_score")
                continue

            if event != "step":
                continue

            step_num = entry.get("step_num", 0)
            accepted = entry.get("accepted", False)
            accept_reason = entry.get("accept_reason", "")
            score_before = entry.get("score_before", 0.0) or 0.0
            score_after = entry.get("score_after", 0.0) or 0.0
            num_tiles_changed = entry.get("num_tiles_changed", 0) or 0

            metrics_after = entry.get("metrics_after") or {}
            change_ratio = metrics_after.get("change_ratio")

            tool_calls: list[ToolCallRecord] = []
            for tr in entry.get("tool_results") or []:
                t_name = tr.get("tool_name", "unknown")
                t_params = tr.get("parameters") or {}
                result = tr.get("result") or {}
                data = result.get("data") or {}
                success = bool(result.get("success")) and bool(data.get("applied", False))
                tiles = data.get("num_tiles_changed", 0) or 0

                tool_calls.append(ToolCallRecord(
                    tool_name=t_name,
                    category=categorize_tool(t_name, t_params),
                    parameters=t_params,
                    success=success,
                    tiles_changed=tiles,
                    step_num=step_num,
                    step_accepted=accepted,
                ))

            steps.append(StepRecord(
                step_num=step_num,
                accepted=accepted,
                accept_reason=accept_reason,
                score_before=score_before,
                score_after=score_after,
                num_tiles_changed=num_tiles_changed,
                change_ratio=change_ratio,
                tool_calls=tool_calls,
            ))

    return RunAnalysis(
        run_dir=str(trace_path.parent),
        steps=steps,
        final_score=final_score,
        grid_size=grid_size,
    )


# ---------------------------------------------------------------------------
# ASCII table formatter
# ---------------------------------------------------------------------------

def format_table(headers: list[str], rows: list[list[str]], aligns: list[str] | None = None) -> str:
    if not rows and not headers:
        return ""
    num_cols = len(headers)
    if aligns is None:
        aligns = ["l"] * num_cols

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < num_cols:
                col_widths[i] = max(col_widths[i], len(cell))

    def _sep():
        return "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    def _row(cells):
        parts = []
        for i, cell in enumerate(cells):
            w = col_widths[i]
            if aligns[i] == "r":
                parts.append(f" {cell:>{w}} ")
            else:
                parts.append(f" {cell:<{w}} ")
        return "|" + "|".join(parts) + "|"

    lines = [_sep(), _row(headers), _sep()]
    for row in rows:
        lines.append(_row(row))
    lines.append(_sep())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

@dataclass
class ToolStats:
    category: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    accepted_steps: set = field(default_factory=set)
    rejected_steps: set = field(default_factory=set)
    tiles_list: list[int] = field(default_factory=list)  # per successful call


def compute_tool_stats(analysis: RunAnalysis) -> dict[str, ToolStats]:
    stats: dict[str, ToolStats] = {}
    for step in analysis.steps:
        for tc in step.tool_calls:
            cat = tc.category
            if cat not in stats:
                stats[cat] = ToolStats(category=cat)
            s = stats[cat]
            s.total_calls += 1
            if tc.success:
                s.successful_calls += 1
                s.tiles_list.append(tc.tiles_changed)
            else:
                s.failed_calls += 1
            if tc.step_accepted:
                s.accepted_steps.add(tc.step_num)
            else:
                s.rejected_steps.add(tc.step_num)
    return stats


def _compute_step_change_ratios(analysis: RunAnalysis) -> dict[str, list[float]]:
    """Map tool category -> list of change_ratio values from steps using that tool."""
    result: dict[str, list[float]] = {}
    for step in analysis.steps:
        if step.change_ratio is None:
            continue
        cats_in_step = {tc.category for tc in step.tool_calls}
        for cat in cats_in_step:
            result.setdefault(cat, []).append(step.change_ratio)
    return result


# ---------------------------------------------------------------------------
# Compact tool summary for a step
# ---------------------------------------------------------------------------

def compact_tools_summary(tool_calls: list[ToolCallRecord]) -> str:
    counts = Counter(tc.category for tc in tool_calls)
    parts = []
    for cat, n in counts.most_common():
        if n > 1:
            parts.append(f"{cat} x{n}")
        else:
            parts.append(cat)
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Shorten accept reason for display
# ---------------------------------------------------------------------------

def shorten_reason(reason: str) -> str:
    if reason.startswith("improved"):
        return "improved"
    if reason.startswith("annealing"):
        # "annealing (T=4.63, p=0.65, score: X -> Y)" -> "annealing"
        return "annealing"
    if reason.startswith("exploration"):
        return "exploration"
    if reason.startswith("rejected"):
        return "rejected"
    return reason[:30]


# ---------------------------------------------------------------------------
# Output sections
# ---------------------------------------------------------------------------

def print_run_overview(analysis: RunAnalysis):
    total = len(analysis.steps)
    accepted = sum(1 for s in analysis.steps if s.accepted)
    rejected = total - accepted
    rate = f"{accepted / total * 100:.1f}%" if total > 0 else "--"
    score = f"{analysis.final_score:.2f}" if analysis.final_score is not None else "--"
    grid = f"{analysis.grid_size[0]}x{analysis.grid_size[1]}" if analysis.grid_size else "--"
    total_tiles = str(analysis.grid_size[0] * analysis.grid_size[1]) if analysis.grid_size else "--"

    rows = [
        ["Total Steps", str(total)],
        ["Accepted Steps", str(accepted)],
        ["Rejected Steps", str(rejected)],
        ["Accept Rate", rate],
        ["Final Score", score],
        ["Grid Size", grid],
        ["Total Tiles", total_tiles],
    ]
    print("\n  Run Overview")
    print(format_table(["Metric", "Value"], rows, ["l", "r"]))


def print_tool_breakdown(analysis: RunAnalysis):
    stats = compute_tool_stats(analysis)
    if not stats:
        print("\n  No tool calls found.")
        return

    # Sort by total calls descending
    sorted_cats = sorted(stats.values(), key=lambda s: s.total_calls, reverse=True)

    rows = []
    total_calls = 0
    total_success = 0
    total_failed = 0
    for s in sorted_cats:
        rows.append([
            s.category,
            str(s.total_calls),
            str(s.successful_calls),
            str(s.failed_calls),
            str(len(s.accepted_steps)),
            str(len(s.rejected_steps)),
        ])
        total_calls += s.total_calls
        total_success += s.successful_calls
        total_failed += s.failed_calls

    rows.append([
        "TOTAL",
        str(total_calls),
        str(total_success),
        str(total_failed),
        "--",
        "--",
    ])

    headers = ["Tool Category", "Calls", "Success", "Failed", "In Acc.", "In Rej."]
    print("\n  Tool Usage Breakdown")
    print(format_table(headers, rows, ["l", "r", "r", "r", "r", "r"]))


def print_tile_stats(analysis: RunAnalysis):
    stats = compute_tool_stats(analysis)
    change_ratios = _compute_step_change_ratios(analysis)

    # Only tools that had at least one successful call with tiles changed
    cats_with_tiles = {
        cat: s for cat, s in stats.items()
        if s.tiles_list and any(t > 0 for t in s.tiles_list)
    }
    if not cats_with_tiles:
        print("\n  No tile changes recorded.")
        return

    sorted_cats = sorted(cats_with_tiles.values(), key=lambda s: s.total_calls, reverse=True)

    rows = []
    for s in sorted_cats:
        non_zero = [t for t in s.tiles_list if t > 0] or [0]
        avg_t = sum(non_zero) / len(non_zero)
        min_t = min(non_zero)
        max_t = max(non_zero)

        cr_list = change_ratios.get(s.category, [])
        non_zero_cr = [c for c in cr_list if c > 0]
        if non_zero_cr:
            avg_cr = f"{sum(non_zero_cr) / len(non_zero_cr) * 100:.1f}%"
            min_cr = f"{min(non_zero_cr) * 100:.1f}%"
            max_cr = f"{max(non_zero_cr) * 100:.1f}%"
        else:
            avg_cr = min_cr = max_cr = "--"

        rows.append([
            s.category,
            f"{avg_t:.1f}",
            str(min_t),
            str(max_t),
            avg_cr,
            min_cr,
            max_cr,
        ])

    headers = ["Tool Category", "Avg Tiles", "Min", "Max", "Avg Chg%", "Min%", "Max%"]
    print("\n  Per-Tool Tile Change Statistics (successful calls with changes)")
    print(format_table(headers, rows, ["l", "r", "r", "r", "r", "r", "r"]))


def print_accepted_details(analysis: RunAnalysis, sort_by: str = "step"):
    accepted = [
        s for s in analysis.steps
        if s.accepted and "agent_stop" not in s.accept_reason
    ]
    if not accepted:
        print("\n  No accepted steps.")
        return

    total_tiles = (analysis.grid_size[0] * analysis.grid_size[1]) if analysis.grid_size else None

    if sort_by == "score_change":
        accepted.sort(key=lambda s: s.score_after - s.score_before, reverse=True)
    elif sort_by == "tiles_changed":
        accepted.sort(key=lambda s: s.num_tiles_changed, reverse=True)
    elif sort_by == "change_ratio":
        accepted.sort(key=lambda s: s.change_ratio or 0, reverse=True)

    rows = []
    for s in accepted:
        tools = compact_tools_summary(s.tool_calls)
        tiles = str(s.num_tiles_changed)
        if s.change_ratio is not None:
            chg_pct = f"{s.change_ratio * 100:.1f}%"
            if total_tiles:
                tiles_from_ratio = int(s.change_ratio * total_tiles)
                chg_pct = f"{s.change_ratio * 100:.1f}% ({tiles_from_ratio})"
        else:
            chg_pct = "--"
        score_change = f"{s.score_before:.0f} -> {s.score_after:.0f}"
        reason = shorten_reason(s.accept_reason)
        rows.append([
            str(s.step_num),
            tools,
            tiles,
            chg_pct,
            score_change,
            reason,
        ])

    headers = ["Step", "Tools Used", "Tiles", "Change %", "Score", "Reason"]
    print("\n  Accepted Steps Detail")
    print(format_table(headers, rows, ["r", "l", "r", "r", "l", "l"]))


def print_comparison(analyses: list[RunAnalysis]):
    if len(analyses) < 2:
        return

    rows = []
    for a in analyses:
        total = len(a.steps)
        accepted = sum(1 for s in a.steps if s.accepted)
        rejected = total - accepted
        rate = f"{accepted / total * 100:.1f}%" if total > 0 else "--"
        score = f"{a.final_score:.2f}" if a.final_score is not None else "--"
        # Use just the directory name for brevity
        name = Path(a.run_dir).name
        rows.append([name, str(total), str(accepted), str(rejected), rate, score])

    headers = ["Run", "Steps", "Acc.", "Rej.", "Rate", "Score"]
    print("\n" + "=" * 80)
    print("  Comparison Summary")
    print("=" * 80)
    print(format_table(headers, rows, ["l", "r", "r", "r", "r", "r"]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze trace.jsonl files from ToolUsePCG optimization runs."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="One or more run directories containing trace.jsonl",
    )
    parser.add_argument(
        "--no-step-details",
        action="store_true",
        help="Suppress per-accepted-step detail table",
    )
    parser.add_argument(
        "--sort-by",
        choices=["step", "score_change", "tiles_changed", "change_ratio"],
        default="step",
        help="Sort accepted steps table (default: step)",
    )
    args = parser.parse_args()

    analyses: list[RunAnalysis] = []

    for run_dir in args.run_dirs:
        trace_path = Path(run_dir) / "trace.jsonl"
        if not trace_path.exists():
            print(f"Warning: {trace_path} not found, skipping", file=sys.stderr)
            continue

        print("\n" + "=" * 80)
        print(f"  Run: {Path(run_dir).name}")
        print("=" * 80)

        analysis = parse_trace(trace_path)
        analyses.append(analysis)

        print_run_overview(analysis)
        print_tool_breakdown(analysis)
        print_tile_stats(analysis)
        if not args.no_step_details:
            print_accepted_details(analysis, sort_by=args.sort_by)
        print()

    print_comparison(analyses)


if __name__ == "__main__":
    main()
