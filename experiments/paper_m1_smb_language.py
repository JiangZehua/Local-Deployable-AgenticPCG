"""Paper M1: deterministic language contract and held-out SMB evaluation.

R108 showed that Qwen3.8 extracted the correct typed fields for every valid
request, but rejected eight complete specifications as ``needs_clarification``.
The parser now emits typed constraints plus explicit ambiguity evidence.  A
deterministic harness derives the execution disposition from the domain,
ambiguities, and domain validator; the model no longer emits or reviews status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from experiments.r061_qwen38_m1 import _write_json
from experiments.r108_smb_functional_harness import (
    ALL_LANGUAGE_METRICS,
    DEFAULT_LANGUAGE_MANIFEST,
    LANGUAGE_SCHEMA,
    LANGUAGE_SYSTEM_PROMPT,
    _canonical_parser_payload,
    build_resolver,
    is_fail_closed,
    derive_parser_disposition,
    parser_payload_to_spec,
)


RUN_ID = "PAPER-M1-SMB-LANG-DETERMINISTIC-CONTRACT-VAL-SEAL"
SEED = 109100
REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = REPO_ROOT / "benchmarks/smb/splits.json"
MANIFEST_ROOT = REPO_ROOT / "benchmarks/smb"
VALIDATION_MANIFEST = MANIFEST_ROOT / "template_validation.json"
SEALED_MANIFEST = MANIFEST_ROOT / "template_test.json"


VALID_TEMPLATES = (
    "In the simulated playthrough, require exactly {k} defeated enemies, {c} collected coins, and {j} voluntary ground jumps.",
    "Make a playable Mario stage whose actual run ends with precisely {k} kills, {c} coins picked up, and {j} jumps initiated from the ground.",
    "Use gameplay events rather than tile counts: Mario must eliminate {k} enemies, collect {c} coins, and perform {j} counted ground jumps.",
    "For the A*-played outcome, set enemy defeats to {k}, collected coins to {c}, and voluntary ground jumps to {j}, all exactly.",
    "I need the completed run—not the visible inventory—to record {k} enemy kills, {c} coin collections, and {j} ground jumps.",
    "During actual play, let Mario finish after defeating exactly {k} enemies, taking exactly {c} coins, and making exactly {j} ground jumps.",
    "以模拟游玩轨迹为准：敌人击杀数必须正好为{k}，实际收集金币数为{c}，主动地面跳跃数为{j}。",
    "请生成可通关的马里奥关卡，实际运行结果需要精确击杀{k}个敌人、收集{c}枚金币，并完成{j}次地面起跳。",
    "不要按地图上的 tile 数量计算；Mario 通关时应记录{k}次击杀、{c}枚已收集金币和{j}次 counted ground jump。",
    "动态目标如下：通关轨迹中 kills={k}、collected coins={c}、voluntary jumps={j}，三个数都必须精确匹配。",
    "我关心的是玩法事件：正好消灭{k}个敌人，拿到{c}枚金币，从地面主动跳起{j}次。",
    "让模拟器验证最终游玩数据：{k}个敌人被击杀、{c}枚金币被取得、发生{j}次地面跳跃。",
)

VALIDATION_GUARDS = (
    ("val-guard-vague-count", "During play, collect a handful of coins and jump a reasonable number of times.", "needs_clarification", "smb_dynamic"),
    ("val-guard-conflict", "The completed run must contain exactly one kill and exactly three kills.", "needs_clarification", "smb_dynamic"),
    ("val-guard-domain-ambiguous", "Put two enemies and three coins in the level.", "needs_clarification", "smb_dynamic"),
    ("val-guard-no-target", "Make a lively and enjoyable Mario course.", "needs_clarification", "smb_dynamic"),
    ("val-guard-static-vague", "The static grid should contain several pipes and lots of blocks.", "needs_clarification", "smb_static"),
    ("val-guard-outside", "In one playthrough Mario must defeat nine enemies, collect one coin, and jump twice.", "unsupported", "smb_dynamic"),
    ("val-guard-other-game", "Create a Sokoban board with three crates and three goals.", "unsupported", "unknown"),
    ("val-guard-mixed-contract", "Use exactly two enemy tiles, and make sure Mario kills exactly those enemies, with no indication whether static or dynamic control is primary.", "needs_clarification", "smb_dynamic"),
)

SEALED_GUARDS = (
    ("test-guard-vague-count", "Have Mario grab some coins while making a sensible amount of jumps.", "needs_clarification", "smb_dynamic"),
    ("test-guard-conflict", "The same run should collect exactly two coins and exactly five coins.", "needs_clarification", "smb_dynamic"),
    ("test-guard-domain-ambiguous", "Give the stage four enemies and one coin.", "needs_clarification", "smb_dynamic"),
    ("test-guard-no-target", "Design a challenging but fair Mario level.", "needs_clarification", "smb_dynamic"),
    ("test-guard-static-vague", "在静态地图里放一些水管和很多方块。", "needs_clarification", "smb_static"),
    ("test-guard-outside-kill", "通关轨迹必须正好击杀十二个敌人、收集两枚金币并跳一次。", "unsupported", "smb_dynamic"),
    ("test-guard-outside-jump", "During play require one kill, one coin, and exactly eleven ground jumps.", "unsupported", "smb_dynamic"),
    ("test-guard-other-game", "Build a Zelda room with a key behind four guards.", "unsupported", "unknown"),
    ("test-guard-mixed-contract", "Place three coin tiles and also collect three coins; choose whichever interpretation is easier.", "needs_clarification", "smb_dynamic"),
    ("test-guard-approximate", "Aim for roughly three defeated enemies and about two collected coins.", "needs_clarification", "smb_dynamic"),
)


def _expected_payload(target: Mapping[str, Any]) -> dict[str, Any]:
    constraints = {
        name: {"relation": "unspecified", "value": -1}
        for name in ALL_LANGUAGE_METRICS
    }
    for name in ("enemies", "coins", "jumps"):
        constraints[name] = {"relation": "exact", "value": int(target[name])}
    return {
        "status": "resolved",
        "domain": "smb_dynamic",
        "constraints": constraints,
        "objectives": [],
        "clarification": "",
    }


def build_manifest(split_name: str, guards: tuple[tuple[str, str, str, str], ...]) -> dict[str, Any]:
    split_payload = json.loads(SPLIT_PATH.read_text())
    rows = split_payload["splits"][split_name]
    cases: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        target = row["target"]
        template_index = int(
            hashlib.sha256(row["case_id"].encode()).hexdigest(), 16
        ) % len(VALID_TEMPLATES)
        instruction = VALID_TEMPLATES[template_index].format(
            k=int(target["enemies"]), c=int(target["coins"]), j=int(target["jumps"])
        )
        cases.append({
            "case_id": f"lang-{row['case_id']}",
            "role": "valid_target",
            "source_case_id": row["case_id"],
            "instruction": instruction,
            "expected": _expected_payload(target),
        })
    for case_id, instruction, status, domain in guards:
        cases.append({
            "case_id": case_id,
            "role": "guard",
            "instruction": instruction,
            "expected": {"status": status, "domain": domain},
        })
    return {
        "manifest_id": f"PAPER-M1-{split_name.upper()}-LANGUAGE-v1",
        "schema_version": "paper-m1-language-v1",
        "frozen": True,
        "split": split_name,
        "source_split_semantic_sha256": split_payload["semantic_sha256"],
        "valid_target_cases": len(rows),
        "guard_cases": len(guards),
        "template_selection": "sha256(case_id) modulo fixed bilingual template bank",
        "witness_levels_or_traces_loaded": False,
        "cases": cases,
    }


def write_manifests(output: Path) -> dict[str, Any]:
    validation = build_manifest("validation", VALIDATION_GUARDS)
    sealed = build_manifest("sealed_test", SEALED_GUARDS)
    validation_path = output / "template_validation.json"
    sealed_path = output / "template_test.json"
    _write_json(validation_path, validation)
    _write_json(sealed_path, sealed)
    return {
        "validation": {
            "path": str(validation_path),
            "sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
            "cases": len(validation["cases"]),
        },
        "sealed": {
            "path": str(sealed_path),
            "sha256": hashlib.sha256(sealed_path.read_bytes()).hexdigest(),
            "cases": len(sealed["cases"]),
        },
    }


def _chat(client: Any, *, model_id: str, messages: list[dict[str, str]], schema: dict[str, Any], schema_name: str, seed: int) -> tuple[dict[str, Any], dict[str, int]]:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.0,
        seed=seed,
        max_tokens=1024,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        },
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    usage = response.usage
    return json.loads(response.choices[0].message.content or "{}"), {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "latency_ms": int(round((time.perf_counter() - started) * 1000)),
    }


BUILDERS = ("legacy_r108", "stitch_v2", "stitch_then_search_v2")


def _language_resolver(builder: str, case: Mapping[str, Any]):
    if builder == "legacy_r108":
        return build_resolver(disabled_backbones=set(case.get("disabled_backbones", [])))
    if builder not in BUILDERS:
        raise ValueError(f"Unknown SMB builder: {builder}")
    from experiments.smb_dynamic_builders import build_resolver as dynamic_resolver
    return dynamic_resolver(builder.removesuffix("_v2"), contract="v2")


def structured_ceiling(manifest: Path, output: Path, *, builder: str = "legacy_r108") -> dict[str, Any]:
    frozen = json.loads(manifest.read_text())
    records = []
    for case in frozen["cases"]:
        if case.get("role") != "valid_target":
            continue
        spec = parser_payload_to_spec(case["expected"])
        result = _language_resolver(builder, case).resolve(spec, allow_search=True, max_evaluator_calls=64)
        payload = result.as_dict(include_level=False)
        records.append({
            "case_id": case["case_id"],
            "status": payload["status"],
            "exact": payload["status"] == "exact",
            "selected_target": payload["selected_target"],
            "evaluator_calls": payload["evaluator_calls"],
            "level_sha256": payload["level_sha256"],
            "takeoff_audit": (result.evaluation_info or {}).get("takeoff_audit"),
        })
    result = {
        "run_id": RUN_ID,
        "builder": builder,
        "arm": "structured-no-llm-resolver",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "cases": len(records),
        "exact": sum(row["exact"] for row in records),
        "gate_pass": bool(records) and all(row["exact"] for row in records),
        "records": records,
    }
    _write_json(output, result)
    return result


def resolve_language_spec(
    parsed: Mapping[str, Any], case: Mapping[str, Any], *, builder: str = "legacy_r108"
):
    """Resolve one admitted parse while preserving case-level search controls."""

    return _language_resolver(builder, case).resolve(
        parser_payload_to_spec(parsed),
        allow_search=True,
        max_evaluator_calls=int(case.get("max_evaluator_calls", 64)),
    )


def run_language(manifest: Path, base_url: str, output: Path, *, model_id: str, revision: str, seed: int = SEED, builder: str = "legacy_r108") -> dict[str, Any]:
    from openai import OpenAI

    frozen = json.loads(manifest.read_text())
    client = OpenAI(base_url=base_url, api_key="EMPTY", max_retries=0, timeout=600.0)
    records = []
    for index, case in enumerate(frozen["cases"]):
        parsed, cost = _chat(
            client,
            model_id=model_id,
            messages=[
                {"role": "system", "content": LANGUAGE_SYSTEM_PROMPT},
                {"role": "user", "content": case["instruction"]},
            ],
            schema=LANGUAGE_SCHEMA,
            schema_name="smb_functional_spec",
            seed=seed + index,
        )
        disposition = derive_parser_disposition(parsed)
        canonical = _canonical_parser_payload(parsed, status=disposition["status"])
        expected = _canonical_parser_payload(case["expected"])
        expected_status = case["expected"]["status"]
        parse_exact = bool(
            disposition["status"] == expected_status
            and parsed["domain"] == case["expected"]["domain"]
            and (
                expected_status != "resolved"
                or (
                    canonical["constraints"] == expected["constraints"]
                    and canonical["objectives"] == expected["objectives"]
                    and not canonical["ambiguities"]
                )
            )
        )
        resolution = None
        if disposition["status"] == "resolved":
            try:
                resolution = resolve_language_spec(parsed, case, builder=builder)
            except (KeyError, TypeError, ValueError) as exc:
                resolution = {"status": "invalid_request", "errors": [str(exc)]}
        resolution_payload = resolution.as_dict(include_level=True) if hasattr(resolution, "as_dict") else resolution
        valid_target = bool(
            case.get("role") == "valid_target"
            or ("role" not in case and expected_status == "resolved")
        )
        functional_exact = bool(
            valid_target and parse_exact and resolution_payload
            and resolution_payload["status"] == "exact"
        )
        fail_closed = bool(
            not valid_target
            and is_fail_closed(expected_status, disposition["status"], resolution)
        )
        false_resolve = bool(not valid_target and disposition["status"] == "resolved")
        costs = {"llm_calls": 1, **cost}
        records.append({
            "case_id": case["case_id"],
            "role": case.get("role", "legacy"),
            "instruction": case["instruction"],
            "expected": expected,
            "raw_parser_payload": parsed,
            "parsed": canonical,
            "disposition": disposition,
            "parse_exact": parse_exact,
            "functional_exact": functional_exact,
            "fail_closed": fail_closed,
            "false_resolve": false_resolve,
            "resolution": resolution_payload,
            "costs": costs,
        })
        _write_json(output, {"execution_complete": False, "records": records})

    valid = [row for row in records if row["role"] == "valid_target" or (row["role"] == "legacy" and row["expected"]["status"] == "resolved")]
    guards = [row for row in records if row not in valid]
    aggregate = {
        "cases": len(records),
        "valid_cases": len(valid),
        "guard_cases": len(guards),
        "parse_exact": sum(row["parse_exact"] for row in records),
        "valid_parse_exact": sum(row["parse_exact"] for row in valid),
        "functional_exact": sum(row["functional_exact"] for row in valid),
        "fail_closed": sum(row["fail_closed"] for row in guards),
        "false_resolve": sum(row["false_resolve"] for row in guards),
        "llm_calls": sum(row["costs"]["llm_calls"] for row in records),
        "input_tokens": sum(row["costs"]["input_tokens"] for row in records),
        "output_tokens": sum(row["costs"]["output_tokens"] for row in records),
        "latency_ms": sum(row["costs"]["latency_ms"] for row in records),
    }
    min_valid = len(valid) if frozen.get("split") in {None, "exposed_regression"} else max(0, len(valid) - 2)
    criteria = {
        "execution_complete": len(records) == len(frozen["cases"]),
        "valid_parse_exact_gate": aggregate["valid_parse_exact"] >= min_valid,
        "valid_functional_exact_gate": aggregate["functional_exact"] >= min_valid,
        "guards_fail_closed_all": aggregate["fail_closed"] == len(guards),
        "zero_false_resolve": aggregate["false_resolve"] == 0,
    }
    result = {
        "run_id": RUN_ID,
        "builder": builder,
        "split": frozen.get("split", "exposed_regression"),
        "execution_complete": criteria["execution_complete"],
        "scientific_gate_pass": all(criteria.values()),
        "criteria": criteria,
        "model_id": model_id,
        "revision": revision,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "status_policy": "deterministic harness decision from domain, ambiguities, and adapter validation",
        "aggregate": aggregate,
        "records": records,
    }
    _write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-manifests")
    build.add_argument("--output", type=Path, required=True)
    ceiling = sub.add_parser("structured-ceiling")
    ceiling.add_argument("--manifest", type=Path, required=True)
    ceiling.add_argument("--output", type=Path, required=True)
    ceiling.add_argument("--builder", choices=BUILDERS, default="stitch_v2")
    run = sub.add_parser("run-language")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--base-url", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--model-id", required=True)
    run.add_argument("--revision", required=True)
    run.add_argument("--seed", type=int, default=SEED)
    run.add_argument("--builder", choices=BUILDERS, default="stitch_v2")
    args = parser.parse_args()
    if args.command == "build-manifests":
        payload = write_manifests(args.output)
    elif args.command == "structured-ceiling":
        payload = structured_ceiling(args.manifest, args.output, builder=args.builder)
    else:
        payload = run_language(
            args.manifest, args.base_url, args.output,
            model_id=args.model_id, revision=args.revision, seed=args.seed, builder=args.builder,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
