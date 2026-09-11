"""R101-B transactional, verifier-gated SMB jump-operator smoke."""

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
from experiments.r101_smb_jump_episode_atlas import (
    classify_jump_causes,
    is_exact_jump_witness,
)
from toolusepcg.level import Level
from toolusepcg.tools.edit_smb_jump_episode import SUPPORTED_JUMP_TARGETS


RUN_ID = "R101-B"
OPERATOR_NAME = "build_smb_jump_episode"
SMOKE_CASE_IDS = (
    "smb-dev-00-k0-c0-j1",
    "smb-dev-01-k0-c1-j2",
    "smb-dev-02-k0-c2-j3",
    "smb-dev-03-k0-c3-j4",
    "smb-dev-05-k0-c4-j5",
    "smb-dev-06-k0-c5-j5",
)
TEMPERATURE_ZERO_INCUMBENT_EXACT = {
    "smb-dev-00-k0-c0-j1": True,
    "smb-dev-01-k0-c1-j2": False,
    "smb-dev-02-k0-c2-j3": False,
    "smb-dev-03-k0-c3-j4": True,
    "smb-dev-05-k0-c4-j5": False,
    "smb-dev-06-k0-c5-j5": False,
}
OPERATOR_INSTRUCTION = (
    "For this enemies=0 jump target, use exactly one build_smb_jump_episode "
    "call with coin_count and jump_count equal to the requested targets. Do "
    "not combine it with place_tile in the same STEP. The candidate is accepted "
    "only if A* verifies exact metrics and the causal decomposition: one "
    "terminal jump plus jump_count-1 forced obstacle jumps, with no gap, "
    "stomp-bounce, unknown jump, or kill event."
)


def _target(optimizer: Any) -> tuple[int, int, int]:
    return (
        int(optimizer.config.targets.enemies.value),
        int(optimizer.config.targets.coins.value),
        int(optimizer.config.targets.jumps.value),
    )


class VerifierGatedJumpOptimizer(SpawnFateSmbOptimizer):
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
            return False, "rejected_jump_operator_mixed_transaction"
        payload = operator_results[0].get("result", {})
        if not payload.get("success"):
            return False, "rejected_jump_operator_tool_failure"
        data = payload.get("data") or {}
        requested = data.get("requested_behavior") or {}
        requested_target = (
            int(requested.get("enemies", -1)),
            int(requested.get("coins", -1)),
            int(requested.get("jumps", -1)),
        )
        target = _target(self)
        obstacles = tuple(int(x) for x in data.get("obstacle_x", []))
        if requested_target != target:
            return False, f"rejected_jump_operator_target_mismatch (requested={requested_target}, target={target})"
        causes = classify_jump_causes(new_result, obstacles)
        if not is_exact_jump_witness(new_result, target, obstacles):
            return False, f"rejected_jump_operator_verifier (metrics={new_result.metrics}, causes={causes['counts']})"
        return True, f"accepted_verified_jump_operator (target={target}, causes={causes['counts']})"

    def _build_environment_step_feedback(self, new_level, new_result, tool_results):
        base = super()._build_environment_step_feedback(new_level, new_result, tool_results)
        operator_results = self._operator_results(tool_results)
        if not operator_results:
            return base
        data = (operator_results[0].get("result", {}).get("data") or {})
        obstacles = tuple(int(x) for x in data.get("obstacle_x", []))
        causes = classify_jump_causes(new_result, obstacles)
        verified = is_exact_jump_witness(new_result, _target(self), obstacles)
        return "\n".join((base, "Jump episode verifier (candidate-local):", f"- causes={causes['counts']}; episodes={causes['episodes']}", f"- verified={verified}; requires one terminal + jumps-1 obstacle, no gap/stomp/unknown/kill."))


def build_jump_optimizer(config: Any, *, acceptance_policy: str = TARGET_FIRST_ACCEPTANCE):
    if acceptance_policy != TARGET_FIRST_ACCEPTANCE:
        raise ValueError("R101-B requires target-first acceptance")
    pair = (int(config.targets.coins.value), int(config.targets.jumps.value))
    if int(config.targets.enemies.value) != 0 or pair not in SUPPORTED_JUMP_TARGETS:
        raise ValueError(f"R101-B unsupported target: enemies={config.targets.enemies.value}, pair={pair}")
    config.optimizer.edit_tools = ["place_tile", OPERATOR_NAME]
    config.optimizer.stable_change_accounting = True
    base = config.optimizer.extra_instruction
    config.optimizer.extra_instruction = f"{base}\n\n{OPERATOR_INSTRUCTION}" if base else OPERATOR_INSTRUCTION
    optimizer = VerifierGatedJumpOptimizer(config, verbose=False)
    optimizer.duplicate_level_guard = True
    return optimizer


def _calls(record: dict[str, Any]) -> int:
    return sum(call.get("tool_name") == OPERATOR_NAME for step in record["steps"] for call in (step.get("final_json") or {}).get("tool_calls", []))


def _accepted(record: dict[str, Any]) -> int:
    return sum(bool(step.get("accepted")) and "accepted_verified_jump_operator" in step.get("accept_reason", "") for step in record["steps"])


def preflight(manifest_path: Path, output: Path, *, seed: int = 101100):
    payload = baseline_preflight(manifest_path, output, case_ids=list(SMOKE_CASE_IDS), seed=seed, optimizer_builder=build_jump_optimizer, metadata={"experiment": RUN_ID, "semantic_operator": OPERATOR_NAME, "duplicate_level_guard": True})
    _, cases = build_cases(manifest_path, case_ids=list(SMOKE_CASE_IDS), seed=seed)
    certified = []
    for case in cases:
        optimizer = build_jump_optimizer(_config(case, model_id="preflight", base_url="http://127.0.0.1:1/v1"))
        optimizer.initialize(seed=seed)
        schema = optimizer._build_openai_extra_body()["structured_outputs"]["json"]
        decoded = {branch["properties"]["tool_name"]["const"] for branch in schema["properties"]["tool_calls"]["items"]["oneOf"]}
        if decoded != set(optimizer.registry.list_tools()):
            raise RuntimeError("runtime registry and decoding schema diverged")
        target = (0, int(case["targets"]["coins"]), int(case["targets"]["jumps"]))
        tool = optimizer.registry.get(OPERATOR_NAME)
        result = tool.execute(coin_count=target[1], jump_count=target[2])
        if not result.success:
            raise RuntimeError(f"jump operator failed for {case['case_id']}")
        obstacles = tuple(result.data["obstacle_x"])
        evaluated = optimizer.evaluator.evaluate(optimizer.edit_manager.level)
        if not is_exact_jump_witness(evaluated, target, obstacles):
            raise RuntimeError(f"jump verifier failed for {case['case_id']}")
        certified.append({"case_id": case["case_id"], "target": case["targets"], "jump_causes": classify_jump_causes(evaluated, obstacles)})
    payload["operator_preflight"] = {"certified": len(certified), "total": len(cases), "cases": certified, "registry_schema_parity": True}
    _write_json(output, payload)
    return payload


def run(manifest_path: Path, base_url: str, output: Path, *, model_id: str, revision: str, max_steps: int = 4, max_protocol_retries: int = 4, seed: int = 101100):
    payload = baseline_run(manifest_path, base_url, output, model_id=model_id, revision=revision, max_steps=max_steps, max_protocol_retries=max_protocol_retries, temperature=0.0, case_ids=list(SMOKE_CASE_IDS), seed=seed, optimizer_builder=build_jump_optimizer, harness_overrides={"baseline": "R085-F2 spawn-fate + R100-B duplicate guard", "semantic_operator": OPERATOR_NAME, "operator_contract": "r101-certified-jump-episode-v1", "operator_verifier": "exact metrics + terminal/obstacle causal counts + no kill", "stable_change_accounting": True, "duplicate_level_guard": True, "single_changed_factor": "transactional verifier-gated jump operator"}, experiment_name=RUN_ID)
    verifier, _ = build_evaluator()
    verified_final = 0
    for record in payload["records"]:
        target = (0, int(record["targets"]["coins"]), int(record["targets"]["jumps"]))
        profile = SUPPORTED_JUMP_TARGETS[(target[1], target[2])]
        replay = verifier.evaluate(Level.from_string(record["final_level"], problem_type="smb"))
        causes = classify_jump_causes(replay, tuple(profile["obstacles"]))
        verified = is_exact_jump_witness(replay, target, tuple(profile["obstacles"]))
        verified_final += int(verified)
        record["operator_audit"] = {"operator_calls": _calls(record), "accepted_verified_operator_steps": _accepted(record), "final_jump_witness_verified": verified, "final_jump_causes": causes}
    exact_ids = {record["case_id"] for record in payload["records"] if record["exact"]}
    incumbent = {case_id for case_id, exact in TEMPERATURE_ZERO_INCUMBENT_EXACT.items() if exact}
    calls = sum(record["operator_audit"]["operator_calls"] for record in payload["records"])
    accepted = sum(record["operator_audit"]["accepted_verified_operator_steps"] for record in payload["records"])
    criteria = {"hard_valid_6_of_6": payload["aggregate"]["hard_valid"] == 6, "joint_exact_at_least_5_of_6": payload["aggregate"]["exact"] >= 5, "actual_final_jump_verification_at_least_5_of_6": verified_final >= 5, "accepted_verified_operator_steps_at_least_5": accepted >= 5, "incumbent_exact_nonregression": incumbent <= exact_ids}
    payload["operator_smoke"] = {"gate_pass": all(criteria.values()), "criteria": criteria, "operator_calls": calls, "accepted_verified_operator_steps": accepted, "final_jump_witnesses_verified": verified_final, "temperature_zero_incumbent_exact": len(incumbent), "temperature_zero_incumbent_exact_ids": sorted(incumbent), "exact_ids": sorted(exact_ids), "verification_evaluator_calls": len(payload["records"])}
    _write_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run"):
        command = subs.add_parser(name)
        command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--seed", type=int, default=101100)
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
        payload = run(args.manifest, args.base_url, args.output, model_id=args.model_id, revision=args.revision, max_steps=args.max_steps, max_protocol_retries=args.max_protocol_retries, seed=args.seed)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
