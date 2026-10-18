"""HPC runner and auditable readout for the Gemma 4 Z64 transfer campaign.

The frozen R061/M2 producer remains untouched.  This module invokes its real
initializer, optimizer, tools, and evaluator one target at a time, while adding
the evidence and failure-preserving output contract requested by HPC_HANDOFF.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import socket
import statistics
import subprocess
import tarfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments import r061_qwen38_m1 as producer
from experiments.zelda_mastery_m0 import canonical_json_bytes
from toolusepcg.optimizer.greedy import GreedyOptimizer


MODEL_REPO = "google/gemma-4-31B-it"
MODEL_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
SERVED_MODEL_NAME = "gemma4-31b-pcg"
PROTOCOL_PATH = Path("configs/gemma4_z64_protocol.json")
DEV_IDS_PATH = Path("configs/gemma4_z64_dev_ids.txt")
FORMAL_IDS_PATH = Path("configs/paper_m2_z64_target_ids.txt")
PROMPT_PATH = Path("prompts/zelda_geometry_serialization_rules_v3.txt")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def json_safe(value: Any) -> Any:
    """Convert evidence-only numpy/path containers to canonical JSON values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    os.replace(temporary, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json_bytes(value).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def read_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate IDs in {path}")
    return values


def relpath(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


class EvidenceGreedyOptimizer(GreedyOptimizer):
    """Greedy optimizer with exact messages and split timing attached to steps."""

    def _capture_grpo_request_messages(self, record: Any, prompt: str) -> None:
        record.request_messages = self._build_messages(prompt)
        record.request_params = {
            "model": self.config.llm.model,
            "temperature": float(self.config.llm.temperature),
            "max_tokens": int(self.config.llm.max_tokens),
            "extra_body": self._build_openai_extra_body(),
        }

    def call_llm(self, *args: Any, **kwargs: Any) -> tuple[str, bool, int, int]:
        self._init_client()
        original_create = self._client.chat.completions.create
        self._evidence_api_attempts = []

        def captured_create(*call_args: Any, **call_kwargs: Any) -> Any:
            started_at = utc_now()
            started = time.perf_counter()
            attempt = {
                "started_at": started_at,
                "request": call_kwargs,
                "response": None,
                "exception": None,
            }
            try:
                response = original_create(*call_args, **call_kwargs)
                attempt["response"] = response.model_dump(mode="json")
                return response
            except Exception as exc:
                attempt["exception"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                attempt["ended_at"] = utc_now()
                attempt["wall_seconds"] = time.perf_counter() - started
                self._evidence_api_attempts.append(attempt)

        self._client.chat.completions.create = captured_create
        started = time.perf_counter()
        try:
            return super().call_llm(*args, **kwargs)
        finally:
            self._evidence_model_seconds = time.perf_counter() - started
            self._client.chat.completions.create = original_create

    def execute_step(self, *args: Any, **kwargs: Any) -> tuple[Any, list[dict], int]:
        started = time.perf_counter()
        try:
            return super().execute_step(*args, **kwargs)
        finally:
            self._evidence_tool_seconds = time.perf_counter() - started

    def step(self, *args: Any, **kwargs: Any) -> Any:
        self._evidence_model_seconds = 0.0
        self._evidence_tool_seconds = 0.0
        record = super().step(*args, **kwargs)
        record.model_seconds = float(self._evidence_model_seconds)
        record.tool_seconds = float(self._evidence_tool_seconds)
        record.api_attempts = list(getattr(self, "_evidence_api_attempts", []))
        return record


def full_trace(optimizer: GreedyOptimizer, case: dict[str, Any]) -> dict[str, Any]:
    steps = []
    for step in optimizer.step_records:
        candidate_level = step.level_after or None
        if step.message_type == "STEP":
            saved_level = step.level_after if step.accepted else step.level_before
        else:
            saved_level = step.level_before
        steps.append(
            {
                "step_num": int(step.step_num),
                "request": {
                    "messages": list(step.request_messages),
                    **dict(getattr(step, "request_params", {})),
                },
                "response": {
                    "content": step.raw_response,
                    "reasoning_content": step.reasoning_content,
                    "truncated": bool(step.truncated),
                    "input_tokens": int(step.input_tokens),
                    "output_tokens": int(step.output_tokens),
                    "output_tokens_include_reasoning": True,
                    "reasoning_tokens": None,
                    "message_type": step.message_type,
                    "parse_errors": list(step.errors),
                    "api_attempts": list(getattr(step, "api_attempts", [])),
                },
                "parsed_message": step.parsed_message,
                "tool_execution": step.tool_results,
                "maps": {
                    "before": step.level_before,
                    "candidate_after": candidate_level,
                    "saved_after": saved_level,
                },
                "metrics": {
                    "before": step.metrics_before,
                    "candidate_after": step.metrics_after,
                    "score_before": float(step.score_before),
                    "score_after": float(step.score_after),
                },
                "decision": {
                    "accepted": bool(step.accepted),
                    "outcome": (
                        "stop"
                        if step.message_type == "STOP"
                        else "save"
                        if step.accepted
                        else "reject"
                        if step.message_type == "STEP"
                        else "failure"
                    ),
                    "reason": step.accept_reason,
                    "acceptance_triage": step.acceptance_triage,
                },
                "timing": {
                    "wall_seconds": float(step.elapsed_time),
                    "model_seconds": float(getattr(step, "model_seconds", 0.0)),
                    "tool_seconds": float(getattr(step, "tool_seconds", 0.0)),
                },
            }
        )
    final_level = optimizer.current_level.to_string()
    return json_safe({
        "schema_version": "gemma4-z64-trace-v1",
        "target_id": case["target_id"],
        "target": {
            "player_key": int(case["player_key"]),
            "key_door": int(case["key_door"]),
        },
        "initial_level": steps[0]["maps"]["before"] if steps else final_level,
        "steps": steps,
        "final_saved_level": final_level,
        "timing": {
            "model_seconds": sum(row["timing"]["model_seconds"] for row in steps),
            "tool_seconds": sum(row["timing"]["tool_seconds"] for row in steps),
        },
    })


def classify_failure(record: dict[str, Any]) -> str | None:
    if record.get("exact") is True:
        return None
    traces = record.get("experiment_enrichment", {}).get("trace", {}).get("steps", [])
    errors = " ".join(
        str(error)
        for step in traces
        for error in step.get("response", {}).get("parse_errors", [])
    ).lower()
    if "timeout" in errors or "timed out" in errors:
        return "timeout"
    if int(record.get("transport_errors", 0)):
        return "infrastructure"
    if int(record.get("truncated_responses", 0)):
        return "truncation"
    if int(record.get("unrecovered_parser_errors", 0)):
        return "parsing"
    if int(record.get("tool_errors", 0)):
        return "invalid_action"
    if record.get("exact_candidate_not_final"):
        return "save_stop"
    return "target_miss"


def episode_status(record: dict[str, Any]) -> str:
    category = classify_failure(record)
    if category == "timeout":
        return "timeout"
    if category == "infrastructure":
        return "infrastructure_interrupted"
    if category in {"parsing", "truncation"} and not any(
        step.get("message_type") == "STEP" for step in record.get("steps", [])
    ):
        return "task_failed"
    return "completed"


def compact_episode(
    *,
    run_id: str,
    arm: str,
    target_id: str,
    attempt_id: str,
    result: dict[str, Any],
    trace_path: Path,
    final_path: Path,
    root: Path,
    started_at: str,
    ended_at: str,
) -> dict[str, Any]:
    record = result["records"][0]
    trace = record["experiment_enrichment"]["trace"]
    costs = record["costs"]
    final = record["final_metrics"]
    return {
        "run_id": run_id,
        "arm": arm,
        "target_id": target_id,
        "seed": int(record["seed"]),
        "attempt_id": attempt_id,
        "status": episode_status(record),
        "exact": bool(record["exact"]),
        "final_map_perfect": producer.final_map_perfect(record),
        "hard_valid": bool(record["hard_constraints"]["hard_valid"]),
        "target_player_key": int(record["target"]["player_key"]),
        "target_key_door": int(record["target"]["key_door"]),
        "final_player_key": int(final["player_key"]),
        "final_key_door": int(final["key_door"]),
        "absolute_error": int(record["e_sum"]),
        "failure_category": classify_failure(record),
        "steps": len(record["steps"]),
        "calls": int(costs["llm_calls"]),
        "input_tokens": int(costs["input_tokens"]),
        "output_tokens": int(costs["output_tokens"]),
        "reasoning_tokens": None,
        "output_tokens_include_reasoning": True,
        "handoff_calls": int(record["protocol_recovery"]["fallback_calls"]),
        "wall_seconds": float(costs["latency_ms"]) / 1000.0,
        "model_seconds": float(trace["timing"]["model_seconds"]),
        "tool_seconds": float(trace["timing"]["tool_seconds"]),
        "started_at": started_at,
        "ended_at": ended_at,
        "trace_path": relpath(trace_path, root),
        "final_level_path": relpath(final_path, root),
        "parser_errors": int(record["parser_errors"]),
        "unrecovered_parser_errors": int(record["unrecovered_parser_errors"]),
        "tool_errors": int(record["tool_errors"]),
        "transport_errors": int(record["transport_errors"]),
        "truncated_responses": int(record["truncated_responses"]),
        "candidate_exact_ever": bool(record["candidate_exact_ever"]),
        "exact_candidate_not_final": bool(record["exact_candidate_not_final"]),
    }


def placeholder_episode(
    run_id: str,
    arm: str,
    target_id: str,
    *,
    status: str,
    attempt_id: str | None = None,
    failure_category: str | None = None,
    trace_path: str | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
) -> dict[str, Any]:
    failed_in_denominator = status in {"task_failed", "timeout"}
    return {
        "run_id": run_id,
        "arm": arm,
        "target_id": target_id,
        "seed": None,
        "attempt_id": attempt_id,
        "status": status,
        "exact": False if failed_in_denominator else None,
        "final_map_perfect": False if failed_in_denominator else None,
        "hard_valid": False if failed_in_denominator else None,
        "target_player_key": None,
        "target_key_door": None,
        "final_player_key": None,
        "final_key_door": None,
        "absolute_error": None,
        "failure_category": failure_category,
        "steps": 0,
        "calls": 0,
        "input_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "output_tokens_include_reasoning": True,
        "handoff_calls": 0,
        "wall_seconds": None,
        "model_seconds": None,
        "tool_seconds": None,
        "started_at": started_at,
        "ended_at": ended_at,
        "trace_path": trace_path,
        "final_level_path": None,
        "parser_errors": None,
        "unrecovered_parser_errors": None,
        "tool_errors": None,
        "transport_errors": None,
        "truncated_responses": None,
        "candidate_exact_ever": None,
        "exact_candidate_not_final": None,
    }


def arm_kwargs(protocol: dict[str, Any], arm: str, source_root: Path) -> dict[str, Any]:
    arm_config = protocol["arms"][arm]
    prompt = arm_config["extra_instruction_file"]
    return {
        "arm": arm,
        "model_id": protocol["model"]["served_model_name"],
        "revision": protocol["model"]["revision"],
        "max_steps": int(protocol["shared"]["max_steps"]),
        "conversation_length": int(protocol["shared"]["conversation_length"]),
        "truncation_recovery": arm_config["truncation_recovery"],
        "max_tokens": int(arm_config["max_tokens"]),
        "thinking_max_tokens": arm_config["thinking_max_tokens"],
        "temperature": float(protocol["shared"]["temperature"]),
        "request_seed_offset": int(protocol["shared"]["request_seed_offset"]),
        "ablation_mode": arm_config["ablation_mode"],
        "extra_instruction": (
            (source_root / prompt).read_text() if prompt is not None else None
        ),
        "optimizer_factory": lambda config: EvidenceGreedyOptimizer(config, verbose=False),
        "record_enricher": lambda optimizer, case: {"trace": full_trace(optimizer, case)},
    }


def validate_inputs(source_root: Path, protocol: dict[str, Any]) -> Path:
    shared = protocol["shared"]
    target_path = source_root / shared["target_file"]
    if sha256_file(target_path) != shared["target_file_sha256"]:
        raise RuntimeError("formal target file hash drift")
    subset_path = source_root / shared["source_subset_path"]
    if sha256_file(subset_path) != shared["source_subset_sha256"]:
        raise RuntimeError("source subset hash drift")
    if len(read_ids(target_path)) != int(shared["formal_target_count_per_arm"]):
        raise RuntimeError("formal target count drift")
    formal_ids = set(read_ids(target_path))
    dev_ids = read_ids(source_root / DEV_IDS_PATH)
    if formal_ids.intersection(dev_ids):
        raise RuntimeError("development targets overlap formal Z64 targets")
    return subset_path


def planned_cells(mode: str, targets: list[str]) -> list[tuple[str, str]]:
    if mode == "formal":
        return [(arm, target) for target in targets for arm in ("open_loop", "verified")]
    return [
        ("open_loop" if index % 2 == 0 else "verified", target)
        for index, target in enumerate(targets)
    ]


def next_attempt_number(attempts_path: Path, arm: str, target_id: str) -> int:
    if not attempts_path.exists():
        return 1
    count = 0
    for line in attempts_path.read_text().splitlines():
        row = json.loads(line)
        if row.get("arm") == arm and row.get("target_id") == target_id:
            count += 1
    return count + 1


def run_campaign(args: argparse.Namespace) -> None:
    source_root = args.source_root.resolve()
    protocol = load_json(source_root / args.protocol)
    subset = validate_inputs(source_root, protocol)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    (output / "raw").mkdir(exist_ok=True)
    (output / "episodes").mkdir(exist_ok=True)
    if args.mode == "formal":
        targets = read_ids(source_root / FORMAL_IDS_PATH)
    else:
        targets = read_ids(source_root / DEV_IDS_PATH)
    cells = planned_cells(args.mode, targets)
    campaign_started = time.monotonic()
    for arm, target_id in cells:
        episode_dir = output / "episodes" / arm / target_id
        compact_path = episode_dir / "episode.json"
        if compact_path.is_file() and load_json(compact_path).get("status") in {
            "completed",
            "task_failed",
            "timeout",
        }:
            continue
        if args.budget_seconds is not None:
            remaining = float(args.budget_seconds) - (time.monotonic() - campaign_started)
            if remaining < float(args.minimum_start_seconds):
                break
        episode_dir.mkdir(parents=True, exist_ok=True)
        raw_path = output / "raw" / arm / f"{target_id}.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        attempt_number = next_attempt_number(output / "attempts.jsonl", arm, target_id)
        attempt_id = f"{arm}-{target_id}-a{attempt_number:02d}"
        started_at = utc_now()
        current = {
            "run_id": args.run_id,
            "mode": args.mode,
            "arm": arm,
            "target_id": target_id,
            "attempt_id": attempt_id,
            "started_at": started_at,
        }
        write_json(output / "current_episode.json", current)
        attempt = {
            **current,
            "status": "running",
            "ended_at": None,
            "reason": None,
            "raw_result_path": relpath(raw_path, output),
            "log_path": None,
        }
        try:
            result = producer.run_matched_arm(
                subset,
                args.base_url,
                raw_path,
                target_ids=[target_id],
                **arm_kwargs(protocol, arm, source_root),
            )
            ended_at = utc_now()
            trace = result["records"][0]["experiment_enrichment"]["trace"]
            trace_path = episode_dir / "trace.json"
            final_path = episode_dir / "final_level.txt"
            write_json(trace_path, trace)
            write_text(final_path, trace["final_saved_level"] + "\n")
            compact = compact_episode(
                run_id=args.run_id,
                arm=arm,
                target_id=target_id,
                attempt_id=attempt_id,
                result=result,
                trace_path=trace_path,
                final_path=final_path,
                root=output,
                started_at=started_at,
                ended_at=ended_at,
            )
            write_json(compact_path, compact)
            attempt.update(
                status=compact["status"],
                ended_at=ended_at,
                reason=compact["failure_category"],
                trace_path=compact["trace_path"],
            )
        except Exception as exc:
            ended_at = utc_now()
            error_path = episode_dir / f"{attempt_id}.error.log"
            write_text(error_path, traceback.format_exc())
            category = (
                "timeout"
                if "timeout" in str(exc).lower() or "timed out" in str(exc).lower()
                else "infrastructure"
            )
            status = "timeout" if category == "timeout" else "infrastructure_interrupted"
            compact = placeholder_episode(
                args.run_id,
                arm,
                target_id,
                status=status,
                attempt_id=attempt_id,
                failure_category=category,
                trace_path=relpath(error_path, output),
                started_at=started_at,
                ended_at=ended_at,
            )
            write_json(compact_path, compact)
            attempt.update(
                status=status,
                ended_at=ended_at,
                reason=f"{type(exc).__name__}: {exc}",
                log_path=relpath(error_path, output),
            )
        append_jsonl(output / "attempts.jsonl", attempt)
        if attempt["status"] == "infrastructure_interrupted":
            break
    if (output / "current_episode.json").exists():
        (output / "current_episode.json").unlink()
    analyze_campaign(output, source_root, protocol, args.mode, args.run_id)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def exact_mcnemar(b: int, c: int) -> float | None:
    discordant = b + c
    if discordant == 0:
        return None
    tail = sum(math.comb(discordant, k) for k in range(min(b, c) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_bootstrap(differences: list[float], *, n: int, seed: int) -> list[float | None]:
    if not differences:
        return [None, None]
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n, dtype=np.float64)
    chunk = 10000
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def load_episode_rows(output: Path, mode: str, run_id: str, source_root: Path) -> list[dict[str, Any]]:
    targets = read_ids(source_root / (FORMAL_IDS_PATH if mode == "formal" else DEV_IDS_PATH))
    rows = []
    for arm, target_id in planned_cells(mode, targets):
        path = output / "episodes" / arm / target_id / "episode.json"
        if path.is_file():
            rows.append(load_json(path))
        else:
            rows.append(placeholder_episode(run_id, arm, target_id, status="not_started"))
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summary_for(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_arm_target = {(row["arm"], row["target_id"]): row for row in rows}
    targets = sorted({row["target_id"] for row in rows})
    paired = []
    differences = []
    open_only = verified_only = common_success = common_failure = 0
    for target in targets:
        open_row = by_arm_target.get(("open_loop", target))
        verified_row = by_arm_target.get(("verified", target))
        if open_row is None or verified_row is None:
            continue
        evaluable = all(
            row["status"] not in {"not_started", "infrastructure_interrupted"}
            and row["exact"] is not None
            for row in (open_row, verified_row)
        )
        difference = (
            int(bool(verified_row["exact"])) - int(bool(open_row["exact"]))
            if evaluable
            else None
        )
        if difference is not None:
            differences.append(float(difference))
            open_only += int(difference == -1)
            verified_only += int(difference == 1)
            common_success += int(open_row["exact"] is True and verified_row["exact"] is True)
            common_failure += int(open_row["exact"] is False and verified_row["exact"] is False)
        paired.append(
            {
                "target_id": target,
                "open_status": open_row["status"],
                "verified_status": verified_row["status"],
                "open_exact": open_row["exact"],
                "verified_exact": verified_row["exact"],
                "open_hard_valid": open_row["hard_valid"],
                "verified_hard_valid": verified_row["hard_valid"],
                "open_absolute_error": open_row["absolute_error"],
                "verified_absolute_error": verified_row["absolute_error"],
                "exact_difference_verified_minus_open": difference,
                "open_failure_category": open_row["failure_category"],
                "verified_failure_category": verified_row["failure_category"],
            }
        )
    analysis_cfg = protocol["analysis"]
    ci = paired_bootstrap(
        differences,
        n=int(analysis_cfg["bootstrap_resamples"]),
        seed=int(analysis_cfg["bootstrap_seed"]),
    )
    # Primary paper metric: score the final saved map, ignoring episode errors.
    map_differences = [
        float(int(bool(v["final_map_perfect"])) - int(bool(o["final_map_perfect"])))
        for o, v in (
            (by_arm_target.get(("open_loop", t)), by_arm_target.get(("verified", t)))
            for t in targets
        )
        if o is not None and v is not None
        and o.get("final_map_perfect") is not None
        and v.get("final_map_perfect") is not None
    ]
    map_ci = paired_bootstrap(
        map_differences,
        n=int(analysis_cfg["bootstrap_resamples"]),
        seed=int(analysis_cfg["bootstrap_seed"]),
    )
    map_wins = sum(value == 1 for value in map_differences)
    map_losses = sum(value == -1 for value in map_differences)
    arm_summary = {}
    for arm in ("open_loop", "verified"):
        arm_rows = [row for row in rows if row["arm"] == arm]
        denominator = len(arm_rows)
        scored_rows = [
            row
            for row in arm_rows
            if row["status"] not in {"not_started", "infrastructure_interrupted"}
            and row["exact"] is not None
        ]
        missing_count = denominator - len(scored_rows)
        wall_values = [
            float(row["wall_seconds"])
            for row in scored_rows
            if row["wall_seconds"] is not None
        ]
        call_values = [
            float(row["calls"]) for row in scored_rows if row["calls"] is not None
        ]
        token_values = [
            float(row["input_tokens"]) + float(row["output_tokens"])
            for row in scored_rows
            if row["input_tokens"] is not None and row["output_tokens"] is not None
        ]
        arm_summary[arm] = {
            "expected": denominator,
            "scored_denominator": len(scored_rows),
            "completed": sum(row["status"] == "completed" for row in arm_rows),
            "task_failed": sum(row["status"] == "task_failed" for row in arm_rows),
            "timeout": sum(row["status"] == "timeout" for row in arm_rows),
            "infrastructure_interrupted": sum(
                row["status"] == "infrastructure_interrupted" for row in arm_rows
            ),
            "not_started": sum(row["status"] == "not_started" for row in arm_rows),
            "exact_count": sum(row["exact"] is True for row in arm_rows),
            "final_map_perfect_count": sum(
                row.get("final_map_perfect") is True for row in arm_rows
            ),
            "exact_rate_full_denominator": (
                sum(row["exact"] is True for row in arm_rows) / denominator
                if denominator and missing_count == 0
                else None
            ),
            "exact_rate_scored": (
                sum(row["exact"] is True for row in scored_rows) / len(scored_rows)
                if scored_rows
                else None
            ),
            "hard_valid_count": sum(row["hard_valid"] is True for row in arm_rows),
            "hard_valid_rate_full_denominator": (
                sum(row["hard_valid"] is True for row in arm_rows) / denominator
                if denominator and missing_count == 0
                else None
            ),
            "hard_valid_rate_scored": (
                sum(row["hard_valid"] is True for row in scored_rows) / len(scored_rows)
                if scored_rows
                else None
            ),
            "costs": {
                "wall_seconds_median": (
                    float(statistics.median(wall_values)) if wall_values else None
                ),
                "wall_seconds_p90": percentile(wall_values, 90),
                "calls_median": (
                    float(statistics.median(call_values)) if call_values else None
                ),
                "total_tokens_median": (
                    float(statistics.median(token_values)) if token_values else None
                ),
                "handoff_calls_total": sum(
                    int(row["handoff_calls"] or 0) for row in scored_rows
                ),
            },
            "missing_ids": [
                row["target_id"]
                for row in arm_rows
                if row["status"] in {"not_started", "infrastructure_interrupted"}
            ],
        }
    summary = {
        "schema_version": "gemma4-z64-summary-v1",
        "temporary": len(differences) != len(targets),
        "expected_episodes": len(rows),
        "paired_targets_expected": len(targets),
        "paired_targets_evaluable": len(differences),
        "arms": arm_summary,
        "paired_exact": {
            "difference_verified_minus_open": (
                statistics.fmean(differences) if differences else None
            ),
            "difference_percentage_points": (
                100.0 * statistics.fmean(differences) if differences else None
            ),
            "bootstrap_95_ci": ci,
            "bootstrap_95_ci_percentage_points": [
                100.0 * value if value is not None else None for value in ci
            ],
            "bootstrap_resamples": int(analysis_cfg["bootstrap_resamples"]),
            "bootstrap_seed": int(analysis_cfg["bootstrap_seed"]),
            "verified_only": verified_only,
            "open_loop_only": open_only,
            "common_success": common_success,
            "common_failure": common_failure,
            "mcnemar_exact_two_sided_p": exact_mcnemar(verified_only, open_only),
        },
        "paired_final_map": {
            "pairs": len(map_differences),
            "difference_percentage_points": (
                100.0 * statistics.fmean(map_differences) if map_differences else None
            ),
            "bootstrap_95_ci_percentage_points": [
                100.0 * value if value is not None else None for value in map_ci
            ],
            "verified_only": map_wins,
            "open_loop_only": map_losses,
            "mcnemar_exact_two_sided_p": exact_mcnemar(map_wins, map_losses),
        },
        "failure_categories": {
            category: sum(row["failure_category"] == category for row in rows)
            for category in sorted(
                {row["failure_category"] for row in rows if row["failure_category"]}
            )
        },
    }
    return summary, paired


def analyze_campaign(
    output: Path,
    source_root: Path,
    protocol: dict[str, Any],
    mode: str,
    run_id: str,
) -> None:
    rows = load_episode_rows(output, mode, run_id, source_root)
    write_text(
        output / "episodes.jsonl",
        "".join(canonical_json_bytes(row).decode("utf-8") + "\n" for row in rows),
    )
    summary, paired = summary_for(rows, protocol)
    write_json(output / "summary.json", summary)
    if mode == "formal":
        write_csv(
            output / "paired_results.csv",
            [
                "target_id",
                "open_status",
                "verified_status",
                "open_exact",
                "verified_exact",
                "open_hard_valid",
                "verified_hard_valid",
                "open_absolute_error",
                "verified_absolute_error",
                "exact_difference_verified_minus_open",
                "open_failure_category",
                "verified_failure_category",
            ],
            paired,
        )
    costs = []
    for row in rows:
        costs.append(
            {
                "arm": row["arm"],
                "target_id": row["target_id"],
                "status": row["status"],
                "wall_seconds": row["wall_seconds"],
                "model_seconds": row["model_seconds"],
                "tool_seconds": row["tool_seconds"],
                "calls": row["calls"],
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "reasoning_tokens": row["reasoning_tokens"],
                "output_tokens_include_reasoning": row["output_tokens_include_reasoning"],
                "handoff_calls": row["handoff_calls"],
            }
        )
    write_csv(
        output / "costs.csv",
        [
            "arm",
            "target_id",
            "status",
            "wall_seconds",
            "model_seconds",
            "tool_seconds",
            "calls",
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "output_tokens_include_reasoning",
            "handoff_calls",
        ],
        costs,
    )
    if mode == "dev":
        probe_path = output / "interface_probe.json"
        probe = load_json(probe_path) if probe_path.is_file() else {}
        checks = probe.get("checks", {})
        lines = [
            "# Gemma 4 Z64 development readout",
            "",
            f"- Run: `{run_id}`",
            f"- Generated: {utc_now()}",
            f"- Development cases: {len(rows)} total, disjoint from formal Z64.",
            f"- Completed: {sum(row['status'] == 'completed' for row in rows)}",
            f"- Task failures: {sum(row['status'] == 'task_failed' for row in rows)}",
            f"- Timeouts: {sum(row['status'] == 'timeout' for row in rows)}",
            f"- Infrastructure interruptions: {sum(row['status'] == 'infrastructure_interrupted' for row in rows)}",
            f"- Interface gate: `{probe.get('gate_pass', False)}`",
            "",
            "## Interface checks",
            "",
            "| Check | Pass |",
            "|---|---:|",
            *[
                f"| `{name}` | `{value}` |"
                for name, value in sorted(checks.items())
            ],
            "",
            "## Development cases",
            "",
            "| Arm | Target | Status | Exact | Hard-valid | Failure |",
            "|---|---|---|---:|---:|---|",
            *[
                "| {arm} | `{target_id}` | {status} | {exact} | {hard_valid} | {failure_category} |".format(
                    **row
                )
                for row in rows
            ],
            "",
            "This gate checks interface wiring and evidence completeness only; its scores are not formal results. Any allowed repair is identified by the code commit in `run_identity.json` and the execution branch history.",
        ]
        write_text(output / "dev_readout.md", "\n".join(lines) + "\n")
    else:
        write_readout(output, summary, rows, run_id)
    write_json(
        output / "analysis_validation.json",
        {
            "duplicates": len(rows) - len({(row["arm"], row["target_id"]) for row in rows}),
            "episodes_rows": len(rows),
            "paired_rows": len(paired),
            "expected_rows": 128 if mode == "formal" else 8,
            "generated_at": utc_now(),
        },
    )


def validate_dev_run(dev_run_dir: Path) -> dict[str, Any]:
    dev_run_dir = dev_run_dir.resolve()
    probe = load_json(dev_run_dir / "interface_probe.json")
    summary = load_json(dev_run_dir / "summary.json")
    validation = load_json(dev_run_dir / "analysis_validation.json")
    errors: list[str] = []
    if probe.get("gate_pass") is not True:
        errors.append("interface_probe gate_pass is not true")
    if validation.get("episodes_rows") != 8:
        errors.append("analysis_validation episodes_rows is not 8")
    if validation.get("expected_rows") != 8:
        errors.append("analysis_validation expected_rows is not 8")
    if validation.get("duplicates") != 0:
        errors.append("analysis_validation contains duplicate cells")
    if summary.get("expected_episodes") != 8:
        errors.append("summary expected_episodes is not 8")
    for arm in ("open_loop", "verified"):
        arm_summary = summary.get("arms", {}).get(arm, {})
        terminal = sum(
            int(arm_summary.get(status, 0))
            for status in ("completed", "task_failed", "timeout")
        )
        if arm_summary.get("expected") != 4:
            errors.append(f"{arm} expected is not 4")
        if terminal != 4:
            errors.append(f"{arm} does not have four non-infrastructure terminal rows")
        if int(arm_summary.get("completed", 0)) < 1:
            errors.append(f"{arm} has no completed development episode")
        if int(arm_summary.get("not_started", 0)) != 0:
            errors.append(f"{arm} contains not_started development rows")
        if int(arm_summary.get("infrastructure_interrupted", 0)) != 0:
            errors.append(f"{arm} contains infrastructure interruptions")
    result = {
        "gate_pass": not errors,
        "dev_run_dir": str(dev_run_dir),
        "errors": errors,
    }
    if errors:
        raise RuntimeError("Development gate failed: " + "; ".join(errors))
    return result


def fmt_rate(value: Any) -> str:
    return "NA" if value is None else f"{100.0 * float(value):.1f}%"


def write_readout(
    output: Path, summary: dict[str, Any], rows: list[dict[str, Any]], run_id: str
) -> None:
    open_summary = summary["arms"]["open_loop"]
    verified_summary = summary["arms"]["verified"]
    paired = summary["paired_exact"]
    ci = paired["bootstrap_95_ci_percentage_points"]
    temporary = summary["temporary"]
    conclusion = (
        "The formal campaign is incomplete; all statistics below are explicitly interim."
        if temporary
        else (
            "The paired interval is wholly above zero, supporting a positive Verified-harness effect on this frozen Gemma/Z64 suite."
            if ci[0] is not None and ci[0] > 0
            else "The paired interval does not establish a strictly positive Verified-harness effect; the result is limited or null on this frozen Gemma/Z64 suite."
        )
    )
    lines = [
        "# Gemma 4 Z64 readout",
        "",
        "## 1. Execution status and conclusion",
        "",
        f"Run `{run_id}`. {conclusion} This result cannot establish that the harness works for all local models.",
        "",
        "## 2. Reproducible identity",
        "",
        "See `run_identity.json`, `frozen_config.json`, `artifact_manifest.json`, and `analysis_command.txt` in this directory.",
        "",
        "## 3. Development checks",
        "",
        "The separate development campaign and any single allowed repair are recorded in `dev_readout.md` and the execution branch history.",
        "",
        "## 4. Coverage",
        "",
        f"- Open-loop: completed {open_summary['completed']}/64; task_failed {open_summary['task_failed']}; timeout {open_summary['timeout']}; infrastructure_interrupted {open_summary['infrastructure_interrupted']}; not_started {open_summary['not_started']}.",
        f"- Verified: completed {verified_summary['completed']}/64; task_failed {verified_summary['task_failed']}; timeout {verified_summary['timeout']}; infrastructure_interrupted {verified_summary['infrastructure_interrupted']}; not_started {verified_summary['not_started']}.",
        f"- Evaluable pairs: {summary['paired_targets_evaluable']}/64.",
        "",
        "## 5. Main results",
        "",
        "| Arm | Final-map perfect | Error-free exact | Hard-valid |",
        "|---|---:|---:|---:|",
        f"| Open-loop | {open_summary['final_map_perfect_count']}/{open_summary['scored_denominator']} | {open_summary['exact_count']}/{open_summary['scored_denominator']} ({fmt_rate(open_summary['exact_rate_scored'])}) | {open_summary['hard_valid_count']}/{open_summary['scored_denominator']} ({fmt_rate(open_summary['hard_valid_rate_scored'])}) |",
        f"| Verified | {verified_summary['final_map_perfect_count']}/{verified_summary['scored_denominator']} | {verified_summary['exact_count']}/{verified_summary['scored_denominator']} ({fmt_rate(verified_summary['exact_rate_scored'])}) | {verified_summary['hard_valid_count']}/{verified_summary['scored_denominator']} ({fmt_rate(verified_summary['hard_valid_rate_scored'])}) |",
        "",
        f"Final-map perfect match (primary): Verified minus Open-loop {summary['paired_final_map']['difference_percentage_points']} percentage points; paired bootstrap 95% CI {summary['paired_final_map']['bootstrap_95_ci_percentage_points']}; Verified-only {summary['paired_final_map']['verified_only']}, Open-loop-only {summary['paired_final_map']['open_loop_only']}.",
        "",
        f"Error-free exact: Verified-only {paired['verified_only']}; Open-loop-only {paired['open_loop_only']}; common success {paired['common_success']}; common failure {paired['common_failure']}.",
        "",
        "## 6. Paired inference",
        "",
        f"Verified minus Open-loop exact: {paired['difference_percentage_points']} percentage points; paired bootstrap 95% CI {ci} using 200,000 resamples and seed 20260831. Exact two-sided McNemar p={paired['mcnemar_exact_two_sided_p']}.",
        "",
        "## 7. Cost and failures",
        "",
        "Per-episode time, calls, tokens, handoffs, and censored failures are in `costs.csv`; resource samples and allocated GPU-hours are in `resources.csv` and `run_identity.json`. Exact reasoning-token counts are null unless emitted separately by the server; completion-token totals include reasoning.",
        "",
        f"Open-loop wall time median/p90: {open_summary['costs']['wall_seconds_median']}/{open_summary['costs']['wall_seconds_p90']} s; Verified: {verified_summary['costs']['wall_seconds_median']}/{verified_summary['costs']['wall_seconds_p90']} s.",
        "",
        f"Failure categories: `{json.dumps(summary['failure_categories'], sort_keys=True)}`.",
        "",
        "## 8. What the paper may and may not claim",
        "",
        "The paper may map any supported claim only to this frozen Gemma 4 31B IT model, the existing Z64 target suite, and the recorded backend/hardware. It may not infer universal local-model transfer, a model-size trend, or a Gemma-versus-Qwen ranking because model family, backend, date, and hardware are confounded.",
        "",
        "## 9. Follow-up",
        "",
        "Complete missing pairs first if this report is interim. M7 remains optional and cannot substitute for missing Z64 evidence.",
    ]
    write_text(output / "gemma4_z64_readout.md", "\n".join(lines) + "\n")


def interface_probe(args: argparse.Namespace) -> None:
    from openai import OpenAI

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", max_retries=0)
    cases = [
        {
            "name": "system_role_thinking_off",
            "messages": [
                {"role": "system", "content": "Follow the user instruction exactly."},
                {"role": "user", "content": "Reply with the single word READY."},
            ],
            "max_tokens": 64,
            "enable_thinking": False,
        },
        {
            "name": "system_role_thinking_on",
            "messages": [
                {"role": "system", "content": "Reason privately, then answer briefly."},
                {"role": "user", "content": "What is 17+25? Reply with the number only."},
            ],
            "max_tokens": 128,
            "enable_thinking": True,
        },
        {
            "name": "forced_truncation",
            "messages": [
                {"role": "system", "content": "Think carefully before answering."},
                {
                    "role": "user",
                    "content": (
                        "Plan a valid 16x16 Zelda map with one player, one key, "
                        "one door, and three enemies, then emit a JSON object."
                    ),
                },
            ],
            "max_tokens": 16,
            "enable_thinking": True,
        },
    ]
    records = []
    for case in cases:
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=SERVED_MODEL_NAME,
            messages=case["messages"],
            max_tokens=case["max_tokens"],
            temperature=0.0,
            timeout=300,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": case["enable_thinking"]
                }
            },
        )
        choice = response.choices[0]
        message = choice.message
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning is None:
            extra = getattr(message, "model_extra", None) or {}
            reasoning = extra.get("reasoning_content") or extra.get("reasoning")
        usage = response.usage
        records.append(
            {
                **case,
                "finish_reason": choice.finish_reason,
                "content": message.content,
                "reasoning_content": reasoning,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "latency_seconds": time.perf_counter() - started,
            }
        )
    truncated = records[-1]
    draft = str(truncated.get("reasoning_content") or truncated.get("content") or "")
    handoff_messages = [
        {
            "role": "system",
            "content": "Return a short JSON object immediately without new analysis.",
        },
        {
            "role": "user",
            "content": (
                "Serialize the last complete plan below as JSON with keys status "
                "and plan. Do not restart the analysis.\n<draft>\n"
                + draft
                + "\n</draft>"
            ),
        },
    ]
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=SERVED_MODEL_NAME,
        messages=handoff_messages,
        max_tokens=512,
        temperature=0.0,
        timeout=300,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    choice = response.choices[0]
    usage = response.usage
    records.append(
        {
            "name": "thinking_off_handoff",
            "messages": handoff_messages,
            "max_tokens": 512,
            "enable_thinking": False,
            "finish_reason": choice.finish_reason,
            "content": choice.message.content,
            "reasoning_content": getattr(choice.message, "reasoning_content", None),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "latency_seconds": time.perf_counter() - started,
        }
    )
    checks = {
        "system_role_off_returns_content": bool(records[0]["content"]),
        "thinking_on_returns_channel": bool(
            records[1]["content"] or records[1]["reasoning_content"]
        ),
        "forced_call_is_truncated": truncated["finish_reason"] == "length",
        "handoff_returns_content": bool(records[-1]["content"]),
        "handoff_stops": records[-1]["finish_reason"] == "stop",
    }
    payload = {
        "schema_version": "gemma4-interface-probe-v1",
        "generated_at": utc_now(),
        "gate_pass": all(checks.values()),
        "checks": checks,
        "records": records,
    }
    write_json(output / "interface_probe.json", payload)
    if not payload["gate_pass"]:
        raise RuntimeError(f"Gemma interface probe failed: {checks}")


def create_identity(args: argparse.Namespace) -> None:
    source_root = args.source_root.resolve()
    output = args.output.resolve()
    protocol = load_json(source_root / args.protocol)
    identity_path = output / "run_identity.json"
    prior = load_json(identity_path) if identity_path.exists() else {}
    gpu_query = []
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        gpu_query = [line.strip() for line in raw.splitlines() if line.strip()]
    except Exception:
        gpu_query = []
    payload = {
        **prior,
        "schema_version": "gemma4-z64-run-identity-v1",
        "run_id": args.run_id,
        "model_repo": protocol["model"]["repo"],
        "model_revision": protocol["model"]["revision"],
        "snapshot": str(args.snapshot.resolve()),
        "snapshot_size_bytes": sum(
            path.stat().st_size for path in args.snapshot.rglob("*") if path.is_file()
        ),
        "tokenizer_chat_template_sha256": (
            sha256_file(args.snapshot / "chat_template.jinja")
            if (args.snapshot / "chat_template.jinja").is_file()
            else None
        ),
        "dtype": protocol["model"]["dtype"],
        "quantization": protocol["model"]["quantization"],
        "code_commit": os.environ.get("GEMMA4_SOURCE_COMMIT")
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
        ).strip(),
        "protocol_sha256": sha256_file(source_root / args.protocol),
        "target_file_sha256": sha256_file(source_root / FORMAL_IDS_PATH),
        "subset_sha256": sha256_file(source_root / protocol["shared"]["source_subset_path"]),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "gpu_inventory": gpu_query,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_gpu_count": os.environ.get("SLURM_GPUS_ON_NODE"),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "vllm", "openai", "numpy")
        },
        "started_at": prior.get("started_at") or utc_now(),
        "ended_at": utc_now() if args.stage == "end" else None,
        "server_load_seconds": (
            float(os.environ["GEMMA4_SERVER_LOAD_SECONDS"])
            if os.environ.get("GEMMA4_SERVER_LOAD_SECONDS")
            else prior.get("server_load_seconds")
        ),
        "allocated_gpu_hours": (
            (time.time() - float(os.environ["GEMMA4_ALLOCATION_START_EPOCH"]))
            * float(os.environ.get("SLURM_GPUS_ON_NODE") or 1)
            / 3600.0
            if args.stage == "end"
            and os.environ.get("GEMMA4_ALLOCATION_START_EPOCH")
            else prior.get("allocated_gpu_hours")
        ),
        "effective_protocol": protocol,
    }
    write_json(identity_path, payload)
    shutil.copy2(source_root / args.protocol, output / "frozen_config.json")
    deviations = """# Protocol deviations

Frozen before formal execution.

- Model family and pinned snapshot change from Qwen to `google/gemma-4-31B-it@842da3794eaa0b77d5f08bae87a17459d91ff475`.
- The server reasoning parser changes from `qwen3` to `gemma4`; per-request `enable_thinking` remains the official boolean template switch.
- The harness continues to emit its existing JSON STEP schema through vLLM structured outputs. Native Gemma tool calling is not enabled and no new PCG guidance is introduced.
- The same producer initializer, tools, evaluation definitions, Z64 targets, temperature, four-step budget, Open-loop ablation, and Verified geometry/handoff policy are retained.
- Exact reasoning-token counts are recorded as null when vLLM does not expose them separately; `output_tokens` includes reasoning and is never added to a second reasoning count.
- Backend/hardware differences from old Qwen runs are confounders; cross-model score ranking is out of scope.
- Evidence-gate amendment v2 aligns `validate-dev` with the existing summary field `arms.<arm>.expected`; it does not change model calls, prompts, targets, seeds, producer/evaluator behavior, or formal statistics.
"""
    write_text(output / "protocol_deviations.md", deviations)


def artifact_manifest(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    archive_dir = output / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / f"{args.run_id}.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for path in sorted(output.rglob("*")):
            if not path.is_file() or path == archive or path.name == "artifact_manifest.json":
                continue
            handle.add(path, arcname=path.relative_to(output))
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append(
                {
                    "path": relpath(path, output),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(
        output / "artifact_manifest.json",
        {
            "schema_version": "gemma4-z64-artifact-manifest-v1",
            "run_id": args.run_id,
            "hpc_evidence_root": str(output),
            "archive_path": str(archive),
            "archive_sha256": sha256_file(archive),
            "files": files,
            "generated_at": utc_now(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--mode", choices=["dev", "formal"], required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--source-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--base-url", required=True)
    run.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    run.add_argument("--budget-seconds", type=float)
    run.add_argument("--minimum-start-seconds", type=float, default=600.0)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--mode", choices=["dev", "formal"], required=True)
    analyze.add_argument("--run-id", required=True)
    analyze.add_argument("--source-root", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)

    identity = subparsers.add_parser("identity")
    identity.add_argument("--stage", choices=["start", "end"], required=True)
    identity.add_argument("--run-id", required=True)
    identity.add_argument("--source-root", type=Path, required=True)
    identity.add_argument("--output", type=Path, required=True)
    identity.add_argument("--snapshot", type=Path, required=True)
    identity.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--run-id", required=True)
    manifest.add_argument("--output", type=Path, required=True)

    probe = subparsers.add_parser("interface-probe")
    probe.add_argument("--base-url", required=True)
    probe.add_argument("--output", type=Path, required=True)

    validate_dev = subparsers.add_parser("validate-dev")
    validate_dev.add_argument("--dev-run-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "run":
        run_campaign(args)
    elif args.command == "analyze":
        protocol = load_json(args.source_root.resolve() / args.protocol)
        analyze_campaign(
            args.output.resolve(),
            args.source_root.resolve(),
            protocol,
            args.mode,
            args.run_id,
        )
    elif args.command == "identity":
        create_identity(args)
    elif args.command == "manifest":
        artifact_manifest(args)
    elif args.command == "interface-probe":
        interface_probe(args)
    elif args.command == "validate-dev":
        print(json.dumps(validate_dev_run(args.dev_run_dir), sort_keys=True))


if __name__ == "__main__":
    main()
