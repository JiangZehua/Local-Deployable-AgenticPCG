"""R102 transactional verifier-gated strict mixed SMB behavior smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.r061_qwen38_m1 import _write_json
from experiments.r080_smb_astar_probe import TARGET_FIRST_ACCEPTANCE, _config
from experiments.r083_smb_baseline import (
    DEFAULT_MANIFEST,
    build_cases,
    preflight as baseline_preflight,
    run as baseline_run,
)
from experiments.r083_smb_contract import build_evaluator
from experiments.r085_smb_spawn_fate import SpawnFateSmbOptimizer
from experiments.r100_smb_corridor_stomp_atlas import (
    event_counts,
    is_exact_stomp_witness,
)
from toolusepcg.level import Level
from toolusepcg.smb_contract import JUMP_EVENT
from toolusepcg.tools.edit_smb_behavior_episode import SUPPORTED_BEHAVIOR_TARGETS


RUN_ID = "R102-B"
OPERATOR_NAME = "build_smb_behavior_episode"
RECERTIFY_REPEATS = 20
SMOKE_CASE_IDS = (
    "smb-dev-08-k1-c0-j1",
    "smb-dev-09-k1-c1-j1",
    "smb-dev-10-k1-c1-j2",
    "smb-dev-12-k1-c2-j3",
    "smb-dev-19-k2-c1-j2",
    "smb-dev-24-k2-c5-j1",
)
TEMPERATURE_ZERO_INCUMBENT_EXACT = {"smb-dev-10-k1-c1-j2"}
OPERATOR_INSTRUCTION = (
    "For this supported mixed behavior target, use exactly one "
    "build_smb_behavior_episode call with enemy_count, coin_count, and "
    "jump_count equal to the requested targets. Do not combine it with raw "
    "tile edits. Acceptance requires actual A* exact metrics, every counted "
    "kill to be a stomp, zero fall kills, and one terminal-zone ground jump "
    "plus jump_count-1 obstacle ground jumps."
)


def _event_type(event: dict[str, Any]) -> int:
    return int(event.get("event_type", event.get("type", -1)))


def classify_behavior_jumps(result, obstacle_x: tuple[int, ...]) -> dict[str, Any]:
    remaining = set(obstacle_x)
    counts = {"terminal": 0, "obstacle": 0, "unknown": 0}
    episodes = []
    for event in (result.info or {}).get("game_events", []):
        if _event_type(event) != JUMP_EVENT:
            continue
        position = event.get("mario_event_position") or {}
        x, y = int(position.get("x", -1)), int(position.get("y", -1))
        if x >= 29:
            cause, feature_x = "terminal", 31
        else:
            ahead = sorted(value for value in remaining if 0 < value - x <= 8)
            if ahead:
                cause, feature_x = "obstacle", ahead[0]
                remaining.remove(feature_x)
            else:
                cause, feature_x = "unknown", None
        counts[cause] += 1
        episodes.append(
            {
                "mario_position": {"x": x, "y": y},
                "cause": cause,
                "feature_x": feature_x,
                "ground_jump": bool(event.get("ground_jump")),
                "stomp_bounce": bool(event.get("stomp_bounce")),
            }
        )
    return {"counts": counts, "episodes": episodes}


def is_exact_behavior_witness(
    result,
    target: tuple[int, int, int],
    obstacle_x: tuple[int, ...],
) -> bool:
    causes = classify_behavior_jumps(result, obstacle_x)
    return bool(
        is_exact_stomp_witness(result, target)
        and causes["counts"] == {
            "terminal": 1,
            "obstacle": target[2] - 1,
            "unknown": 0,
        }
        and all(
            episode["ground_jump"] and not episode["stomp_bounce"]
            for episode in causes["episodes"]
        )
    )


def _configured_target(optimizer: Any) -> tuple[int, int, int]:
    return tuple(
        int(getattr(optimizer.config.targets, name).value)
        for name in ("enemies", "coins", "jumps")
    )


class VerifierGatedBehaviorOptimizer(SpawnFateSmbOptimizer):
    @staticmethod
    def _operator_results(tool_results: list[dict]) -> list[dict]:
        return [item for item in tool_results if item.get("tool_name") == OPERATOR_NAME]

    def decide_accept(self, new_result, num_changes: int, **kwargs: Any):
        tool_results = list(kwargs.get("tool_results") or [])
        operator_results = self._operator_results(tool_results)
        if not operator_results:
            return super().decide_accept(new_result, num_changes, **kwargs)
        real_calls = [item for item in tool_results if item.get("tool_name") != "_truncated"]
        if len(real_calls) != 1:
            return False, "rejected_behavior_operator_mixed_transaction"
        payload = operator_results[0].get("result", {})
        if not payload.get("success"):
            return False, "rejected_behavior_operator_tool_failure"
        data = payload.get("data") or {}
        requested = data.get("requested_behavior") or {}
        requested_target = tuple(
            int(requested.get(name, -1)) for name in ("enemies", "coins", "jumps")
        )
        target = _configured_target(self)
        obstacles = tuple(int(x) for x in data.get("obstacle_x", []))
        if requested_target != target:
            return False, (
                "rejected_behavior_operator_target_mismatch "
                f"(requested={requested_target}, target={target})"
            )
        if not is_exact_behavior_witness(new_result, target, obstacles):
            return False, (
                "rejected_behavior_operator_verifier "
                f"(metrics={new_result.metrics}, events={event_counts(new_result)}, "
                f"jumps={classify_behavior_jumps(new_result, obstacles)['counts']})"
            )
        return True, f"accepted_verified_behavior_operator (target={target})"

    def _build_environment_step_feedback(self, new_level, new_result, tool_results):
        base = super()._build_environment_step_feedback(
            new_level, new_result, tool_results
        )
        operator_results = self._operator_results(tool_results)
        if not operator_results:
            return base
        data = operator_results[0].get("result", {}).get("data") or {}
        obstacles = tuple(int(x) for x in data.get("obstacle_x", []))
        target = _configured_target(self)
        return "\n".join(
            (
                base,
                "Strict mixed behavior verifier (candidate-local):",
                f"- stomp_events={event_counts(new_result)}",
                f"- jump_causes={classify_behavior_jumps(new_result, obstacles)}",
                f"- verified={is_exact_behavior_witness(new_result, target, obstacles)}",
            )
        )


def build_behavior_optimizer(
    config: Any,
    *,
    acceptance_policy: str = TARGET_FIRST_ACCEPTANCE,
) -> VerifierGatedBehaviorOptimizer:
    if acceptance_policy != TARGET_FIRST_ACCEPTANCE:
        raise ValueError("R102-B requires target-first acceptance")
    target = tuple(
        int(getattr(config.targets, name).value)
        for name in ("enemies", "coins", "jumps")
    )
    if target not in SUPPORTED_BEHAVIOR_TARGETS:
        raise ValueError(f"R102-B unsupported target: {target}")
    config.optimizer.edit_tools = ["place_tile", OPERATOR_NAME]
    config.optimizer.stable_change_accounting = True
    base = config.optimizer.extra_instruction
    config.optimizer.extra_instruction = (
        f"{base}\n\n{OPERATOR_INSTRUCTION}" if base else OPERATOR_INSTRUCTION
    )
    optimizer = VerifierGatedBehaviorOptimizer(config, verbose=False)
    optimizer.duplicate_level_guard = True
    return optimizer


def _operator_calls(record: dict[str, Any]) -> int:
    return sum(
        call.get("tool_name") == OPERATOR_NAME
        for step in record["steps"]
        for call in (step.get("final_json") or {}).get("tool_calls", [])
    )


def _accepted_steps(record: dict[str, Any]) -> int:
    return sum(
        bool(step.get("accepted"))
        and "accepted_verified_behavior_operator" in step.get("accept_reason", "")
        for step in record["steps"]
    )


def preflight(
    manifest_path: Path,
    output: Path,
    *,
    seed: int = 102100,
) -> dict[str, Any]:
    payload = baseline_preflight(
        manifest_path,
        output,
        case_ids=list(SMOKE_CASE_IDS),
        seed=seed,
        optimizer_builder=build_behavior_optimizer,
        metadata={
            "experiment": RUN_ID,
            "semantic_operator": OPERATOR_NAME,
            "strict_mixed_contract": True,
        },
    )
    _, cases = build_cases(manifest_path, case_ids=list(SMOKE_CASE_IDS), seed=seed)
    certified = []
    evaluator, _ = build_evaluator()
    for case in cases:
        optimizer = build_behavior_optimizer(
            _config(case, model_id="preflight", base_url="http://127.0.0.1:1/v1")
        )
        optimizer.initialize(seed=seed)
        schema = optimizer._build_openai_extra_body()["structured_outputs"]["json"]
        decoded = {
            branch["properties"]["tool_name"]["const"]
            for branch in schema["properties"]["tool_calls"]["items"]["oneOf"]
        }
        if decoded != set(optimizer.registry.list_tools()):
            raise RuntimeError("runtime registry and decoding schema diverged")
        target = tuple(
            int(case["targets"][name]) for name in ("enemies", "coins", "jumps")
        )
        result = optimizer.registry.get(OPERATOR_NAME).execute(
            enemy_count=target[0], coin_count=target[1], jump_count=target[2]
        )
        if not result.success:
            raise RuntimeError(f"behavior operator failed for {case['case_id']}")
        obstacles = tuple(result.data["obstacle_x"])
        signatures = []
        for _ in range(RECERTIFY_REPEATS):
            evaluated = evaluator.evaluate(optimizer.edit_manager.level)
            if not is_exact_behavior_witness(evaluated, target, obstacles):
                raise RuntimeError(f"strict mixed verifier failed for {case['case_id']}")
            signatures.append(
                json.dumps(
                    {
                        "metrics": evaluated.metrics,
                        "events": (evaluated.info or {}).get("normalized_event_trace", []),
                    },
                    sort_keys=True,
                )
            )
        if len(set(signatures)) != 1:
            raise RuntimeError(f"unstable strict mixed witness for {case['case_id']}")
        certified.append(
            {
                "case_id": case["case_id"],
                "target": case["targets"],
                "event_counts": event_counts(evaluated),
                "jump_causes": classify_behavior_jumps(evaluated, obstacles),
                "stable_repeats": RECERTIFY_REPEATS,
            }
        )
    payload["operator_preflight"] = {
        "certified": len(certified),
        "total": len(cases),
        "cases": certified,
        "registry_schema_parity": True,
        "evaluator_calls": len(cases) * RECERTIFY_REPEATS,
        "unknown_is_not_infeasible": True,
    }
    _write_json(output, payload)
    return payload


def run(
    manifest_path: Path,
    base_url: str,
    output: Path,
    *,
    model_id: str,
    revision: str,
    max_steps: int = 4,
    max_protocol_retries: int = 4,
    seed: int = 102100,
) -> dict[str, Any]:
    payload = baseline_run(
        manifest_path,
        base_url,
        output,
        model_id=model_id,
        revision=revision,
        max_steps=max_steps,
        max_protocol_retries=max_protocol_retries,
        temperature=0.0,
        case_ids=list(SMOKE_CASE_IDS),
        seed=seed,
        optimizer_builder=build_behavior_optimizer,
        harness_overrides={
            "baseline": "R085-F2 spawn-fate + R100-B duplicate guard",
            "semantic_operator": OPERATOR_NAME,
            "operator_contract": "r102-certified-strict-mixed-episode-v1",
            "operator_verifier": "exact + stomp-only + fall=0 + causal ground jumps",
            "oracle_backed_operator_smoke": True,
            "mastery_claim": False,
            "single_changed_factor": "transactional strict mixed behavior operator",
        },
        experiment_name=RUN_ID,
    )
    verifier, _ = build_evaluator()
    verified_final = 0
    for record in payload["records"]:
        target = tuple(
            int(record["targets"][name]) for name in ("enemies", "coins", "jumps")
        )
        profile = SUPPORTED_BEHAVIOR_TARGETS[target]
        replay = verifier.evaluate(
            Level.from_string(record["final_level"], problem_type="smb")
        )
        verified = is_exact_behavior_witness(
            replay, target, tuple(profile["obstacles"])
        )
        verified_final += int(verified)
        record["operator_audit"] = {
            "operator_calls": _operator_calls(record),
            "accepted_verified_operator_steps": _accepted_steps(record),
            "final_strict_mixed_witness_verified": verified,
            "event_counts": event_counts(replay),
            "jump_causes": classify_behavior_jumps(
                replay, tuple(profile["obstacles"])
            ),
        }
    exact_ids = {record["case_id"] for record in payload["records"] if record["exact"]}
    accepted = sum(
        record["operator_audit"]["accepted_verified_operator_steps"]
        for record in payload["records"]
    )
    criteria = {
        "hard_valid_6_of_6": payload["aggregate"]["hard_valid"] == 6,
        "joint_exact_at_least_5_of_6": payload["aggregate"]["exact"] >= 5,
        "strict_final_verification_at_least_5_of_6": verified_final >= 5,
        "accepted_verified_operator_steps_at_least_5": accepted >= 5,
        "incumbent_exact_nonregression": TEMPERATURE_ZERO_INCUMBENT_EXACT <= exact_ids,
    }
    payload["operator_smoke"] = {
        "gate_pass": all(criteria.values()),
        "criteria": criteria,
        "operator_calls": sum(
            record["operator_audit"]["operator_calls"] for record in payload["records"]
        ),
        "accepted_verified_operator_steps": accepted,
        "final_strict_mixed_witnesses_verified": verified_final,
        "temperature_zero_incumbent_exact_ids": sorted(TEMPERATURE_ZERO_INCUMBENT_EXACT),
        "exact_ids": sorted(exact_ids),
        "mastery_claim": False,
    }
    _write_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run"):
        command = subs.add_parser(name)
        command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--seed", type=int, default=102100)
        if name == "run":
            command.add_argument("--base-url", required=True)
            command.add_argument("--model-id", required=True)
            command.add_argument("--revision", required=True)
            command.add_argument("--max-steps", type=int, default=4)
            command.add_argument("--max-protocol-retries", type=int, default=4)
    args = parser.parse_args()
    if args.command == "preflight":
        payload = preflight(args.manifest, args.output, seed=args.seed)
    else:
        payload = run(
            args.manifest,
            args.base_url,
            args.output,
            model_id=args.model_id,
            revision=args.revision,
            max_steps=args.max_steps,
            max_protocol_retries=args.max_protocol_retries,
            seed=args.seed,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
