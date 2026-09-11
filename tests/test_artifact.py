"""Runner argument handling and failure accounting tests."""
import argparse
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import pytest
import yaml
from experiments.reproduce import ROOT, SUITES, build_plan



def args_for(suite, tmp_path):
    return argparse.Namespace(suite=suite, output=tmp_path / suite, model="qwen38", model_id=None,
                              revision=None, base_url="http://127.0.0.1:8000/v1", gemma_mode="formal",
                              gguf=tmp_path / "model.gguf")


def test_every_launch_plan_is_accepted_by_its_actual_cli(tmp_path):
    class Parsed(Exception):
        pass
    original_parse = argparse.ArgumentParser.parse_args
    def parse_only(parser, *args, **kwargs):
        original_parse(parser, *args, **kwargs)
        raise Parsed
    # Parse complete commands; do not execute producers or contact a server.
    for suite in SUITES:
        plan = build_plan(args_for(suite, tmp_path))
        for command in plan:
            with patch.object(sys, "argv", [command[2] if command[1] == "-m" else command[1],
                                            *command[3 if command[1] == "-m" else 2:]]), \
                 patch.object(argparse.ArgumentParser, "parse_args", parse_only):
                with pytest.raises(Parsed):
                    if command[1] == "-m":
                        runpy.run_module(command[2], run_name="__main__")
                    else:
                        runpy.run_path(command[1], run_name="__main__")


def test_tools_keep_shared_seeds_and_explicit_profile(tmp_path):
    plan = build_plan(args_for("tool-loop", tmp_path))
    assert len(plan) == 37
    for left, right in zip(plan[:-1:2], plan[1:-1:2], strict=True):
        for flag in ("--process-seed", "--seed", "--init", "--config", "--extra-instruction"):
            assert left[left.index(flag)+1] == right[right.index(flag)+1]
        assert left[left.index("--prompt-profile")+1] == "freeform"
        assert right[right.index("--ablation")+1] == "no_evaluation_loop"


def test_missing_tool_runs_stay_in_denominator(tmp_path):
    from experiments.tool_evaluation import evaluate_loop
    result = evaluate_loop(tmp_path, tmp_path / "evaluation.json")
    assert len(result["records"]) == 36
    for arm in result["aggregate_by_arm"].values():
        assert arm["planned"] == 18 and arm["functional_valid"] == 0
    assert all(not row["artifact_present"] for row in result["records"])


def test_lmstudio_plan_contains_all_132_original_cases():
    from experiments.lmstudio_campaign import jobs
    rows = jobs("test-model")
    assert len({(r["suite"], r["id"]) for r in rows}) == 132
    assert {name: sum(r["suite"] == name for r in rows) for name in ("z64", "m6", "m7")} == {"z64": 64, "m6": 32, "m7": 36}


def test_paired_statistics_reject_missing_observations_and_check_direction(tmp_path):
    from experiments.paired_stats import compare
    left = tmp_path / 'before.json'
    right = tmp_path / 'after.json'
    rows = [{"target_id": f"case-{i}", "target": {"player_key": i}, "seed": i,
             "initialization_sha256": "same-initial", "exact": False} for i in range(4)]
    left.write_text(json.dumps({"records": rows}))
    right.write_text(json.dumps({"records": [dict(r, exact=True) for r in rows]}))
    result = compare(left, right)
    assert result['difference_percentage_points'] == 100
    assert result['paired_bootstrap_95_percent_interval'] == [100, 100]
    assert result['mcnemar_two_sided_exact_p'] == 0.125
    broken = [dict(r, exact=True) for r in rows]
    del broken[0]['exact']
    right.write_text(json.dumps({"records": broken}))
    with pytest.raises(ValueError, match='Missing or invalid'):
        compare(left, right)
    right.write_text(json.dumps({"records": rows[:-1]}))
    with pytest.raises(ValueError, match='same nonempty target set'):
        compare(left, right)


def test_native_analysis_keeps_completed_task_failure_and_all_attempt_costs(tmp_path):
    from experiments.lmstudio_analysis import analyze
    job = {'suite': 'm6', 'id': 'example', 'seed': 1, 'execution_mode': 'native_inference',
           'cases': [{'role': 'valid_target'}]}
    (tmp_path / 'jobs.json').write_text(json.dumps([job]))
    attempts = []
    for attempt in range(1, 4):
        path = tmp_path / 'm6' / 'example' / f'attempt_{attempt:02}'
        call = path / 'call_01'
        call.mkdir(parents=True)
        (call / 'response.json').write_text(json.dumps({'stats': {'predictedTokensCount': 10}, 'elapsed_seconds': 2}))
        attempts.append({'key': 'm6/example', 'attempt': attempt, 'returncode': 1 if attempt == 1 else 0, 'seconds': 3})
        if attempt > 1:
            (path / 'result.json').write_text(json.dumps({'records': [{'case_id': 'example', 'role': 'valid_target',
                'parse_exact': attempt == 3, 'functional_exact': attempt == 3}]}))
    (tmp_path / 'attempts.jsonl').write_text('\n'.join(map(json.dumps, attempts)) + '\n')
    report = analyze(tmp_path, tmp_path / 'analysis')
    assert report['complete']
    assert report['episodes'][0]['attempt'] == 2
    assert report['suites']['m6']['functional_exact'] == 0
    assert report['recorded_worker_seconds'] == 9
    assert report['suites']['m6']['output_tokens_observed'] == 30
