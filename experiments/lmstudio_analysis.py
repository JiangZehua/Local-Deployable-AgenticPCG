"""Read-only native campaign analysis, including every attempt and wire audit.

An execution interruption is reported separately from a completed task failure.
Server comparisons require explicit newly generated reference files.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
from experiments.paper_m7_fixed_rng36 import evaluate as evaluate_tools

ROOT = Path(__file__).resolve().parents[1]

def read(path, default=None):
    if not path.exists():
        return {} if default is None else default
    # A concurrent writer may be between truncation and completion. Do not
    # silently replace a corrupt/truncated result with a negative observation.
    return json.loads(path.read_text())

def distribution(values):
    values = sorted(v for v in values if v is not None)
    return {'n': len(values), 'total': sum(values) if values else None,
            'median': statistics.median(values) if values else None,
            'p90_nearest_rank': values[math.ceil(.9 * len(values))-1] if values else None,
            'max': max(values) if values else None}

def trace_cost_counts(steps):
    """Count saved tool outcomes, keeping attempted, successful and accepted distinct."""
    if steps is None:
        return {key: None for key in ('proposed_tool_calls', 'recorded_tool_results',
                'successful_tool_results', 'failed_tool_results', 'accepted_trace_steps', 'rejected_trace_steps')}
    results = [result for step in steps for result in step.get('tool_results', [])
               if result.get('tool_name') != '_truncated']
    return {
        'proposed_tool_calls': sum(len((step.get('parsed_message') or {}).get('tool_calls') or []) for step in steps),
        'recorded_tool_results': len(results),
        'successful_tool_results': sum(result.get('result', {}).get('success') is True for result in results),
        'failed_tool_results': sum(result.get('result', {}).get('success') is False for result in results),
        'accepted_trace_steps': sum(step.get('accepted') is True for step in steps),
        'rejected_trace_steps': sum(step.get('accepted') is False and step.get('message_type') == 'STEP' for step in steps),
    }

def metrics(suite, record):
    if not record:
        return {}
    if suite == 'm6':
        return {k: record.get(k) for k in ('parse_exact', 'functional_exact', 'fail_closed', 'false_resolve')}
    if suite == 'z64':
        return {'exact': record.get('exact'), 'hard_valid': record.get('hard_constraints', {}).get('hard_valid')}
    return {'functional_valid': record.get('structure', {}).get('functional_valid'),
            'structural_proxy': record.get('proxy_outcome_success'),
            'strict_route': record.get('routing', {}).get('expected_or_equivalent_route')}

def indexed_records(suite, payload):
    rows = {}
    for record in payload.get('records', []):
        key = (record['task_id'] + '__' + record['arm'] if suite == 'm7'
               else record['case_id'] if suite == 'm6' else record['target_id'])
        if key in rows:
            raise ValueError('Duplicate reference case: ' + key)
        rows[key] = record
    return rows

def config_audit(call_dir, job):
    request, wire = read(call_dir/'request.json'), read(call_dir/'wire_success.json')
    if not wire:
        return {'verified': None, 'reason': 'no_success_wire_record'}
    requested = {f['key']: f['value'] for layer in request['predictionConfigStack']['layers']
                 for f in layer['config']['fields']}
    effective = {f['key']: f['value'] for f in wire['predictionConfig']['fields']}
    suffixes = ('temperature', 'maxPredictedTokens', 'contextOverflowPolicy', 'seed',
                'topKSampling', 'topPSampling', 'minPSampling', 'repeatPenalty', 'enableThinking', 'structured')
    mismatches = [key for key, value in requested.items()
                  if key.endswith(suffixes) and effective.get(key)!=value]
    model = wire.get('modelInfo', {})
    expected_model = 'qwen/qwen3.8-27b@lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf'
    if model.get('indexedModelIdentifier')!=expected_model or model.get('contextLength')!=32768:
        mismatches.append('model_identity_or_context')
    cap = requested.get('llm.prediction.maxPredictedTokens', {}).get('value')
    allowed = {'m6': {1024}, 'z64': {6144, 8192}, 'm7': {4096}}[job['suite']]
    if cap not in allowed:
        mismatches.append('suite_output_budget')
    if requested.get('llm.prediction.temperature')!=0:
        mismatches.append('nonzero_temperature')
    seed = job.get('seed', job.get('case', {}).get('seed'))
    if requested.get('llm.prediction.seed')!={'checked': True, 'value': seed}:
        mismatches.append('case_seed')
    return {'verified': not mismatches, 'mismatches': mismatches,
            'actual_model_identifier': model.get('indexedModelIdentifier'), 'context_length': model.get('contextLength')}


def analyze(run: Path, output: Path, references: dict | None = None) -> dict:
    run = run.resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs = read(run / "jobs.json", [])
    if not jobs or any(j.get("execution_mode") == "cpu_only" for j in jobs):
        raise ValueError("Expected a native inference campaign, not a CPU-only preflight")
    if len({(j["suite"], j["id"]) for j in jobs}) != len(jobs):
        raise ValueError("Duplicate campaign cases")
    ledger = [json.loads(line) for line in (run / "attempts.jsonl").read_text().splitlines()] if (run / "attempts.jsonl").exists() else []
    selected = {}
    for job in jobs:
        key = job["suite"] + "/" + job["id"]
        attempts = [row for row in ledger if row["key"] == key]
        # The first successful process is final even if its scientific outcome failed.
        finished = next((row for row in attempts if row["returncode"] == 0), None)
        selected[key] = finished
    run_paths = {j["id"]: run / j["suite"] / j["id"] / f"attempt_{selected[j['suite']+'/'+j['id']]['attempt']:02}" / "pcg"
                 for j in jobs if j["suite"] == "m7" and selected[j["suite"]+"/"+j["id"]]}
    m7 = evaluate_tools(ROOT / "configs/paper_m7_fixed_rng36.yaml", run / "unobserved", output / "m7_evaluation.json", run_paths=run_paths)
    m7_rows = indexed_records("m7", m7)
    episodes, calls = [], []
    for job in jobs:
        key = job["suite"] + "/" + job["id"]
        entry = selected[key]
        suite, case = job["suite"], job["id"]
        directory = run / suite / case
        record = {}
        if entry:
            chosen = directory / f"attempt_{entry['attempt']:02}"
            if suite == "m7":
                candidate = m7_rows[case]
                if candidate["artifact_present"]:
                    record = candidate
            else:
                records = read(chosen / "result.json").get("records", [])
                if len(records) > 1:
                    raise ValueError(f"Expected one case result: {chosen}")
                record = records[0] if records else {}
                if record and (record.get("case_id") if suite == "m6" else record.get("target_id")) != case:
                    raise ValueError(f"Case identity mismatch: {chosen}")
        attempts = [row for row in ledger if row["key"] == key]
        state = "complete" if record else "missing_artifact" if entry else "interrupted" if attempts else "unobserved"
        episodes.append({"suite": suite, "case_id": case, "state": state, "attempt": entry["attempt"] if entry else None,
                         "arm": job.get("arm"), "role": job.get("cases", [{}])[0].get("role"),
                         "metrics": metrics(suite, record), "record": record})
        # Include interrupted/orphan attempts; observed inference cost must not disappear on resume.
        for attempt_dir in sorted(directory.glob("attempt_*")):
            for call_dir in sorted(attempt_dir.glob("call_*")):
                response, error = read(call_dir / "response.json"), read(call_dir / "error.json")
                stats = response.get("stats", {})
                calls.append({"suite": suite, "case_id": case, "attempt": attempt_dir.name, "call": call_dir.name,
                              "elapsed_seconds": response.get("elapsed_seconds", error.get("elapsed_seconds")),
                              "input_tokens": stats.get("promptTokensCount"), "output_tokens": stats.get("predictedTokensCount"),
                              "stop_reason": stats.get("stopReason"), "error": error.get("error"),
                              "configuration_audit": config_audit(call_dir, job)})
    summaries = {}
    for suite in ("m6", "z64", "m7"):
        rows = [e for e in episodes if e["suite"] == suite]
        suite_calls = [c for c in calls if c["suite"] == suite]
        summaries[suite] = {"planned": len(rows), "states": dict(Counter(r["state"] for r in rows)),
                            "completed": sum(r["state"] == "complete" for r in rows),
                            "output_tokens_observed": sum(c["output_tokens"] or 0 for c in suite_calls),
                            "calls_missing_token_counts": sum(c["output_tokens"] is None for c in suite_calls),
                            "call_seconds": distribution(c["elapsed_seconds"] for c in suite_calls)}
    summaries["z64"]["exact"] = sum(e["metrics"].get("exact") is True for e in episodes if e["suite"] == "z64")
    valid = [e for e in episodes if e["suite"] == "m6" and e["role"] == "valid_target"]
    guards = [e for e in episodes if e["suite"] == "m6" and e["role"] != "valid_target"]
    summaries["m6"].update(valid_requests=len(valid), guard_requests=len(guards),
                           typed_exact=sum(e["metrics"].get("parse_exact") is True for e in valid),
                           functional_exact=sum(e["metrics"].get("functional_exact") is True for e in valid),
                           guards_fail_closed=sum(e["metrics"].get("fail_closed") is True for e in guards),
                           false_resolve=sum(e["metrics"].get("false_resolve") is True for e in valid + guards))
    summaries["m7"].update(aggregate_by_arm=m7["aggregate_by_arm"], scientific_gate_pass=m7["scientific_gate_pass"])
    paired = []
    for suite, path in (references or {}).items():
        reference = indexed_records(suite, read(path))
        for episode in [e for e in episodes if e["suite"] == suite]:
            if episode["case_id"] not in reference:
                raise ValueError("Missing reference case: " + episode["case_id"])
            for metric, server in metrics(suite, reference[episode["case_id"]]).items():
                native = episode["metrics"].get(metric)
                paired.append({"suite": suite, "case_id": episode["case_id"], "metric": metric,
                               "server": server, "native": native,
                               "native_minus_server": int(native)-int(server) if native is not None and server is not None else None})
    result = {"schema_version": "portable-native-analysis-v1", "planned": len(jobs),
              "complete": all(e["state"] == "complete" for e in episodes), "recorded_attempts": len(ledger),
              "orphan_attempt_directories": sum(1 for j in jobs for p in (run/j['suite']/j['id']).glob('attempt_*')) - len(ledger),
              "recorded_worker_seconds": sum(row["seconds"] for row in ledger),
              "timing_definition": "Sum of recorded worker attempts; excludes model loading and unrecorded interrupted-process time.",
              "configuration_audit": dict(Counter(str(c["configuration_audit"]["verified"]) for c in calls)),
              "suites": summaries, "episodes": episodes, "calls": calls, "paired": paired,
              "interpretation": "Descriptive exact-ID comparisons only. Interrupted and missing observations remain distinct from completed task failures."}
    (output / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for name in ("m6", "z64", "m7"):
        parser.add_argument("--server-" + name, type=Path)
    args = parser.parse_args()
    result = analyze(args.run, args.output, {s: getattr(args, "server_" + s) for s in ("m6", "z64", "m7") if getattr(args, "server_" + s)})
    print(json.dumps({"planned": result["planned"], "complete": result["complete"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
