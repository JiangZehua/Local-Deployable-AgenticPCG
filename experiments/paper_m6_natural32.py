"""PAPER-M6 Natural-32 evaluation and transparent CPU parser baseline.

The model-facing parser and certified SMB resolver live in the frozen SMB
mastery worktree.  This module owns only the new manifest checks, a deliberately
simple lexicon/nearest-number baseline, and the combined preregistered readout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "paper-m6-natural32.v1"
ALL_METRICS = ("enemies", "coins", "jumps")
RANGES = {"enemies": (0, 3), "coins": (0, 5), "jumps": (0, 5)}

NUMBER_WORDS = {
    "no": 0,
    "zero": 0,
    "one": 1,
    "single": 1,
    "once": 1,
    "two": 2,
    "pair": 2,
    "twice": 2,
    "three": 3,
    "trio": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "没有": 0,
    "不要": 0,
    "不用": 0,
    "零": 0,
    "一": 1,
    "一次": 1,
    "一回": 1,
    "一个": 1,
    "一只": 1,
    "一枚": 1,
    "两": 2,
    "二": 2,
    "两个": 2,
    "两次": 2,
    "两回": 2,
    "两枚": 2,
    "三": 3,
    "三个": 3,
    "三次": 3,
    "三回": 3,
    "三枚": 3,
    "四": 4,
    "四个": 4,
    "四次": 4,
    "四枚": 4,
    "五": 5,
    "五个": 5,
    "五次": 5,
    "五枚": 5,
}

ALIASES = {
    "enemies": (
        "enemy", "enemies", "foe", "foes", "kill", "kills", "defeat",
        "defeated", "eliminated", "combat", "stomping", "敌人", "敌", "怪",
        "击杀", "消灭", "解决", "打掉", "清掉",
    ),
    "coins": (
        "coin", "coins", "collected", "gathered", "pickups", "pocket",
        "金币", "收集", "实收", "捡", "拿", "收入囊中",
    ),
    "jumps": (
        "jump", "jumps", "takeoff", "takeoffs", "hop", "ground",
        "跳", "起跳", "跳跃", "脚离开地面",
    ),
}

AMBIGUOUS_MARKERS = (
    "few", "some ", "whenever", "feels natural", "whichever", "approximately",
    "roughly", "大概", "三四", "别太多", "如果", "否则", "适中", "有趣就行",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token_spans(text: str) -> list[tuple[int, int, int, str]]:
    lowered = text.lower()
    spans: list[tuple[int, int, int, str]] = []
    for match in re.finditer(r"(?<!\d)\d+(?!\d)", lowered):
        spans.append((match.start(), match.end(), int(match.group()), match.group()))
    # Longest first prevents 一 from being separately counted inside 一次.
    occupied: list[tuple[int, int]] = [(start, end) for start, end, _, _ in spans]
    for token in sorted(NUMBER_WORDS, key=len, reverse=True):
        for match in re.finditer(re.escape(token), lowered):
            if any(match.start() < end and match.end() > start for start, end in occupied):
                continue
            spans.append((match.start(), match.end(), NUMBER_WORDS[token], token))
            occupied.append((match.start(), match.end()))
    return sorted(spans)


def _alias_positions(text: str, aliases: Iterable[str]) -> list[int]:
    lowered = text.lower()
    positions: list[int] = []
    for alias in aliases:
        positions.extend(match.start() for match in re.finditer(re.escape(alias), lowered))
    return positions


def _nearest_metric_values(text: str) -> dict[str, list[int]]:
    tokens = _token_spans(text)
    positions = {metric: _alias_positions(text, aliases) for metric, aliases in ALIASES.items()}
    assigned: dict[str, list[int]] = {metric: [] for metric in ALL_METRICS}
    for start, end, value, _ in tokens:
        center = (start + end) / 2
        distances = {
            metric: min((abs(center - pos) for pos in metric_positions), default=10_000)
            for metric, metric_positions in positions.items()
        }
        metric, distance = min(distances.items(), key=lambda item: item[1])
        if distance <= 42:
            assigned[metric].append(value)
    return assigned


def rule_parse(instruction: str) -> dict[str, Any]:
    lowered = instruction.lower()
    if any(name in lowered for name in ("zelda", "sokoban", "locked door", "地牢")):
        return {"status": "unsupported", "domain": "unknown", "constraints": {}}
    if any(marker in lowered for marker in AMBIGUOUS_MARKERS):
        return {"status": "needs_clarification", "domain": "smb_dynamic", "constraints": {}}
    if "tile" in lowered and any(word in lowered for word in ("collect", "have mario", "实际")):
        return {"status": "needs_clarification", "domain": "smb_dynamic", "constraints": {}}

    # The supported dynamic envelope has no count above five.  Reject before
    # nearest-alias assignment so incidental phrases such as "one run" cannot
    # hide an explicit out-of-envelope request.
    if any(value > 5 for _, _, value, _ in _token_spans(instruction)):
        return {"status": "unsupported", "domain": "smb_dynamic", "constraints": {}}

    values = _nearest_metric_values(instruction)
    if any(len(set(metric_values)) > 1 for metric_values in values.values()):
        return {"status": "needs_clarification", "domain": "smb_dynamic", "constraints": {}}
    if any(not metric_values for metric_values in values.values()):
        return {"status": "needs_clarification", "domain": "smb_dynamic", "constraints": {}}

    constraints = {
        metric: {"relation": "exact", "value": metric_values[0]}
        for metric, metric_values in values.items()
    }
    if any(
        not RANGES[metric][0] <= constraint["value"] <= RANGES[metric][1]
        for metric, constraint in constraints.items()
    ):
        return {"status": "unsupported", "domain": "smb_dynamic", "constraints": constraints}
    return {
        "status": "resolved",
        "domain": "smb_dynamic",
        "constraints": constraints,
        "objectives": [],
    }


def _expected_core(case: Mapping[str, Any]) -> dict[str, Any]:
    expected = case["expected"]
    payload = {"status": expected["status"], "domain": expected["domain"]}
    if expected["status"] == "resolved":
        payload["constraints"] = expected["constraints"]
        payload["objectives"] = expected.get("objectives", [])
    return payload


def baseline_evaluation(manifest: Mapping[str, Any]) -> dict[str, Any]:
    records = []
    for case in manifest["cases"]:
        parsed = rule_parse(case["instruction"])
        expected = _expected_core(case)
        exact = (
            parsed == expected
            if expected["status"] == "resolved"
            else parsed["status"] == expected["status"] and parsed["domain"] == expected["domain"]
        )
        is_guard = case["role"] == "guard"
        records.append({
            "case_id": case["case_id"],
            "role": case["role"],
            "expected": expected,
            "parsed": parsed,
            "parse_exact": exact,
            "false_resolve": bool(is_guard and parsed["status"] == "resolved"),
            "fail_closed": bool(is_guard and parsed["status"] != "resolved"),
        })
    valid = [row for row in records if row["role"] == "valid_target"]
    guards = [row for row in records if row["role"] == "guard"]
    return {
        "baseline": "manifest-authored-lexical-guard-sanity-v1",
        "independence_boundary": (
            "The heuristic was authored with the manifest categories and is not an "
            "independent guard benchmark. Its valid-request score is a transparent "
            "rule-parser sanity comparison; guard performance is descriptive only."
        ),
        "records": records,
        "aggregate": {
            "cases": len(records),
            "valid_cases": len(valid),
            "guard_cases": len(guards),
            "valid_parse_exact": sum(row["parse_exact"] for row in valid),
            "guard_parse_exact": sum(row["parse_exact"] for row in guards),
            "fail_closed": sum(row["fail_closed"] for row in guards),
            "false_resolve": sum(row["false_resolve"] for row in guards),
        },
    }


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


def _longest_contiguous_overlap(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    best = 0
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            value = previous[index - 1] + 1 if left_token == right_token else 0
            current.append(value)
            best = max(best, value)
        previous = current
    return best


def prior_template_audit(manifest: Mapping[str, Any], templates: Sequence[str]) -> dict[str, Any]:
    rows = []
    for case in manifest["cases"]:
        if case["role"] != "valid_target":
            continue
        case_tokens = _word_tokens(case["instruction"])
        overlaps = [
            _longest_contiguous_overlap(case_tokens, _word_tokens(template))
            for template in templates
        ]
        rows.append({
            "case_id": case["case_id"],
            "max_contiguous_token_overlap": max(overlaps, default=0),
            "exact_string_reuse": any(case["instruction"].strip() == template.strip() for template in templates),
        })
    return {
        "prior_template_count": len(templates),
        "valid_case_count": len(rows),
        "max_contiguous_token_overlap": max((row["max_contiguous_token_overlap"] for row in rows), default=0),
        "exact_string_reuse": sum(row["exact_string_reuse"] for row in rows),
        "gate_pass": bool(rows) and not any(row["exact_string_reuse"] for row in rows)
        and all(row["max_contiguous_token_overlap"] < 12 for row in rows),
        "records": rows,
    }


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Natural-32 schema")
    cases = payload.get("cases") or []
    if len(cases) != 32:
        raise ValueError("Natural-32 requires exactly 32 cases")
    if len({case["case_id"] for case in cases}) != 32:
        raise ValueError("duplicate case_id")
    valid = [case for case in cases if case.get("role") == "valid_target"]
    guards = [case for case in cases if case.get("role") == "guard"]
    if len(valid) != 24 or len(guards) != 8:
        raise ValueError("Natural-32 requires 24 valid cases and 8 guards")
    for case in valid:
        if case["expected"]["status"] != "resolved":
            raise ValueError(f"{case['case_id']}: valid case must resolve")
        if set(case["expected"]["constraints"]) != set(ALL_METRICS):
            raise ValueError(f"{case['case_id']}: incomplete ground truth")
    return payload


def preflight(
    manifest_path: Path,
    output: Path | None = None,
    *,
    require_template_audit: bool = False,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    baseline = baseline_evaluation(manifest)
    template_audit = None
    try:
        from experiments.paper_m1_smb_language import VALID_TEMPLATES
    except ImportError:
        if require_template_audit:
            raise
    else:
        template_audit = prior_template_audit(manifest, VALID_TEMPLATES)
        if require_template_audit and not template_audit["gate_pass"]:
            raise ValueError("Natural-32 reuses a prior template string or >=12-token span")
    payload = {
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": _sha256(manifest_path),
        "case_count": len(manifest["cases"]),
        "language_counts": {
            language: sum(case.get("language") == language for case in manifest["cases"])
            for language in ("en", "zh", "mixed")
        },
        "ground_truth_source": "frozen manifest fields; never model output",
        "prior_template_reuse_audit": template_audit,
        "rule_baseline": baseline,
        "gate_pass": True,
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def combine(manifest_path: Path, model_path: Path, ceiling_path: Path, output: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    model = json.loads(model_path.read_text())
    ceiling = json.loads(ceiling_path.read_text())
    baseline = baseline_evaluation(manifest)
    aggregate = model["aggregate"]
    manifest_sha = _sha256(manifest_path)
    provenance_checks = {
        "model_manifest_sha_matches": model.get("manifest_sha256") == manifest_sha,
        "ceiling_manifest_sha_matches": ceiling.get("manifest_sha256") == manifest_sha,
        "model_id_matches": model.get("model_id") == manifest.get("model"),
        "model_revision_matches": model.get("revision") == manifest.get("model_revision"),
    }
    criteria = {
        "provenance_matches_frozen_manifest": all(provenance_checks.values()),
        "all_32_executed": bool(model.get("execution_complete") and aggregate["cases"] == 32),
        "valid_typed_exact_at_least_20_of_24": aggregate["valid_parse_exact"] >= 20,
        "valid_end_to_end_exact_at_least_20_of_24": aggregate["functional_exact"] >= 20,
        "all_8_guards_fail_closed": aggregate["fail_closed"] == 8,
        "zero_false_resolve": aggregate["false_resolve"] == 0,
        "structured_ceiling_24_of_24": ceiling.get("exact") == 24,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_sha,
        "model_id": model.get("model_id"),
        "model_revision": model.get("revision"),
        "claim_boundary": manifest["claim_boundary"],
        "criteria": criteria,
        "scientific_gate_pass": all(criteria.values()),
        "provenance_checks": provenance_checks,
        "model_parser_and_resolver": model,
        "structured_ceiling": ceiling,
        "rule_baseline": baseline,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--output", type=Path)
    check.add_argument("--require-template-audit", action="store_true")
    merge = sub.add_parser("combine")
    merge.add_argument("--manifest", type=Path, required=True)
    merge.add_argument("--model-result", type=Path, required=True)
    merge.add_argument("--structured-ceiling", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        payload = preflight(
            args.manifest,
            args.output,
            require_template_audit=args.require_template_audit,
        )
    else:
        payload = combine(args.manifest, args.model_result, args.structured_ceiling, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
