"""R100-C transactional, verifier-gated SMB stomp-operator smoke."""

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
from experiments.r085_smb_spawn_fate import SpawnFateSmbOptimizer
from experiments.r100_smb_corridor_stomp_atlas import (
    event_counts,
    is_exact_stomp_witness,
)
from experiments.r083_smb_contract import build_evaluator
from toolusepcg.level import Level
from toolusepcg.tools.edit_smb_stomp_corridor import SUPPORTED_STOMP_TARGETS


RUN_ID = "R100-C"
OPERATOR_NAME = "build_smb_stomp_corridor"
SMOKE_CASE_IDS = (
    "smb-dev-07-k1-c0-j0",
    "smb-dev-16-k1-c5-j0",
    "smb-dev-22-k2-c4-j0",
    "smb-dev-23-k2-c5-j0",
    "smb-dev-25-k3-c2-j0",
    "smb-dev-26-k3-c3-j0",
)
TEMPERATURE_ZERO_INCUMBENT_EXACT = {
    case_id: False for case_id in SMOKE_CASE_IDS
}
OPERATOR_INSTRUCTION = (
    "For this jump=0 target, use exactly one build_smb_stomp_corridor call "
    "with enemy_count and coin_count equal to the requested targets. Do not "
    "combine that semantic operator with place_tile in the same STEP. The "
    "candidate is accepted only if the A* verifier observes exact completion, "
    "coins, jumps, and stomp-only counted kills with zero fall kills."
)


def _configured_target(optimizer: Any) -> tuple[int, int, int]:
    return (
        int(optimizer.config.targets.enemies.value),
        int(optimizer.config.targets.coins.value),
        int(optimizer.config.targets.jumps.value),
    )


class VerifierGatedStompOptimizer(SpawnFateSmbOptimizer):
    """Accept semantic corridor edits only after candidate-local A* proof."""

    @staticmethod
    def _operator_results(tool_results: list[dict]) -> list[dict]:
        return [
            item for item in tool_results
            if item.get("tool_name") == OPERATOR_NAME
        ]

    def decide_accept(
        self,
        new_result,
        num_changes: int,
        **kwargs: Any,
    ) -> tuple[bool, str]:
        tool_results = list(kwargs.get("tool_results") or [])
        operator_results = self._operator_results(tool_results)
        if not operator_results:
            return super().decide_accept(new_result, num_changes, **kwargs)

        real_calls = [
            item for item in tool_results if item.get("tool_name") != "_truncated"
        ]
        if len(real_calls) != 1:
            return False, "rejected_stomp_operator_mixed_transaction"
        result_payload = operator_results[0].get("result", {})
        if not result_payload.get("success"):
            return False, "rejected_stomp_operator_tool_failure"
        requested = (result_payload.get("data") or {}).get("requested_behavior") or {}
        requested_target = (
            int(requested.get("enemies", -1)),
            int(requested.get("coins", -1)),
            int(requested.get("jumps", -1)),
        )
        target = _configured_target(self)
        if requested_target != target:
            return False, (
                "rejected_stomp_operator_target_mismatch "
                f"(requested={requested_target}, target={target})"
            )
        counts = event_counts(new_result)
        if not is_exact_stomp_witness(new_result, target):
            return False, (
                "rejected_stomp_operator_verifier "
                f"(metrics={new_result.metrics}, events={counts})"
            )
        return True, (
            "accepted_verified_stomp_operator "
            f"(target={target}, events={counts})"
        )

    def _build_environment_step_feedback(self, new_level, new_result, tool_results):
        base = super()._build_environment_step_feedback(
            new_level, new_result, tool_results
        )
        if not self._operator_results(tool_results):
            return base
        target = _configured_target(self)
        counts = event_counts(new_result)
        verified = is_exact_stomp_witness(new_result, target)
        return "\n".join(
            (
                base,
                "Stomp operator verifier (candidate-local):",
                f"- target={target}; event_counts={counts}; verified={verified}",
                "- Verification requires exact metrics, counted_kills==stomp_kills, and fall_kills=0.",
            )
        )


def build_operator_optimizer(
    config: Any,
    *,
    acceptance_policy: str = TARGET_FIRST_ACCEPTANCE,
) -> VerifierGatedStompOptimizer:
    if acceptance_policy != TARGET_FIRST_ACCEPTANCE:
        raise ValueError("R100-C requires target-first acceptance")
    target = (
        int(config.targets.enemies.value),
        int(config.targets.coins.value),
    )
    jumps = int(config.targets.jumps.value)
    if target not in SUPPORTED_STOMP_TARGETS or jumps != 0:
        raise ValueError(
            f"R100-C only supports certified jump-zero targets; got {target + (jumps,)}"
        )
    config.optimizer.edit_tools = ["place_tile", OPERATOR_NAME]
    config.optimizer.stable_change_accounting = True
    base_instruction = config.optimizer.extra_instruction
    config.optimizer.extra_instruction = (
        f"{base_instruction}\n\n{OPERATOR_INSTRUCTION}"
        if base_instruction else OPERATOR_INSTRUCTION
    )
    optimizer = VerifierGatedStompOptimizer(config, verbose=False)
    optimizer.duplicate_level_guard = True
    return optimizer


def _operator_calls(record: dict[str, Any]) -> int:
    return sum(
        call.get("tool_name") == OPERATOR_NAME
        for step in record["steps"]
        for call in (step.get("final_json") or {}).get("tool_calls", [])
    )


def _accepted_verified_steps(record: dict[str, Any]) -> int:
    return sum(
        bool(step.get("accepted"))
        and "accepted_verified_stomp_operator" in step.get("accept_reason", "")
        for step in record["steps"]
    )


def preflight(
    manifest_path: Path,
    output: Path,
    *,
    seed: int = 100100,
) -> dict[str, Any]:
    payload = baseline_preflight(
        manifest_path,
        output,
        case_ids=list(SMOKE_CASE_IDS),
        seed=seed,
        optimizer_builder=build_operator_optimizer,
        metadata={
            "experiment": RUN_ID,
            "semantic_operator": OPERATOR_NAME,
            "duplicate_level_guard": True,
        },
    )
    _, cases = build_cases(
        manifest_path, case_ids=list(SMOKE_CASE_IDS), seed=seed
    )
    certified = []
    for case in cases:
        optimizer = build_operator_optimizer(
            _config(case, model_id="preflight", base_url="http://127.0.0.1:1/v1")
        )
        optimizer.initialize(seed=seed)
        if OPERATOR_NAME not in optimizer.registry.list_tools():
            raise RuntimeError("semantic operator missing from runtime registry")
        schema = optimizer._build_openai_extra_body()["structured_outputs"]["json"]
        tool_names = {
            branch["properties"]["tool_name"]["const"]
            for branch in schema["properties"]["tool_calls"]["items"]["oneOf"]
        }
        if tool_names != set(optimizer.registry.list_tools()):
            raise RuntimeError("runtime registry and decoding schema diverged")
        target = (
            int(case["targets"]["enemies"]),
            int(case["targets"]["coins"]),
            int(case["targets"]["jumps"]),
        )
        tool = optimizer.registry.get(OPERATOR_NAME)
        result = tool.execute(enemy_count=target[0], coin_count=target[1])
        if not result.success:
            raise RuntimeError(f"operator preflight failed for {case['case_id']}")
        evaluated = optimizer.evaluator.evaluate(optimizer.edit_manager.level)
        if not is_exact_stomp_witness(evaluated, target):
            raise RuntimeError(f"operator verifier failed for {case['case_id']}")
        certified.append(
            {
                "case_id": case["case_id"],
                "target": case["targets"],
                "event_counts": event_counts(evaluated),
            }
        )
    payload["operator_preflight"] = {
        "certified": len(certified),
        "total": len(cases),
        "cases": certified,
        "registry_schema_parity": True,
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
    seed: int = 100100,
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
        optimizer_builder=build_operator_optimizer,
        harness_overrides={
            "baseline": "R085-F2 spawn-fate + R100-B duplicate guard",
            "semantic_operator": OPERATOR_NAME,
            "operator_contract": "r100-certified-stomp-corridor-v1",
            "operator_verifier": "exact metrics + stomp-only counted kills + fall=0",
            "stable_change_accounting": True,
            "duplicate_level_guard": True,
            "single_changed_factor": "transactional verifier-gated stomp operator",
        },
        experiment_name=RUN_ID,
    )

    verifier, _ = build_evaluator()
    verified_final = 0
    for record in payload["records"]:
        target = (
            int(record["targets"]["enemies"]),
            int(record["targets"]["coins"]),
            int(record["targets"]["jumps"]),
        )
        replay = verifier.evaluate(
            Level.from_string(record["final_level"], problem_type="smb")
        )
        verified = is_exact_stomp_witness(replay, target)
        verified_final += int(verified)
        record["operator_audit"] = {
            "operator_calls": _operator_calls(record),
            "accepted_verified_operator_steps": _accepted_verified_steps(record),
            "final_stomp_witness_verified": verified,
            "final_event_counts": event_counts(replay),
        }

    exact_ids = {
        record["case_id"] for record in payload["records"] if record["exact"]
    }
    incumbent_exact_ids = {
        case_id for case_id, exact in TEMPERATURE_ZERO_INCUMBENT_EXACT.items()
        if exact
    }
    operator_calls = sum(
        record["operator_audit"]["operator_calls"]
        for record in payload["records"]
    )
    accepted_verified = sum(
        record["operator_audit"]["accepted_verified_operator_steps"]
        for record in payload["records"]
    )
    scientific_gate = {
        "hard_valid_6_of_6": payload["aggregate"]["hard_valid"] == 6,
        "joint_exact_at_least_4_of_6": payload["aggregate"]["exact"] >= 4,
        "actual_final_stomp_verification_at_least_4_of_6": verified_final >= 4,
        "accepted_verified_operator_steps_at_least_4": accepted_verified >= 4,
        "incumbent_exact_nonregression": incumbent_exact_ids <= exact_ids,
    }
    payload["operator_smoke"] = {
        "gate_pass": all(scientific_gate.values()),
        "criteria": scientific_gate,
        "operator_calls": operator_calls,
        "accepted_verified_operator_steps": accepted_verified,
        "final_stomp_witnesses_verified": verified_final,
        "temperature_zero_incumbent_exact": len(incumbent_exact_ids),
        "temperature_zero_incumbent_exact_ids": sorted(incumbent_exact_ids),
        "exact_ids": sorted(exact_ids),
        "verification_evaluator_calls": len(payload["records"]),
    }
    _write_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--seed", type=int, default=100100)
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
