"""Sequential native LM Studio reproduction with an immutable attempt ledger.

Load the registered Q4_K_M model yourself before running this command. Resume
requires an explicit --resume and never replaces a failed attempt's evidence.
"""
import argparse
import hashlib
import fcntl
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "e00082f779fa385cee8c68a3ec8833a75778cc87272240b942f74e0b8243e520"


def jobs(model_id: str) -> list[dict]:
    result = []
    m6 = json.loads((ROOT / "configs/paper_m6_natural32.json").read_text())
    for i, case in enumerate(m6["cases"]):
        result.append({"suite": "m6", "id": case["case_id"], "seed": 116000 + i,
                       "cases": [case], "split": "mac_formal_natural32"})
    cases = {c["target_id"]: c for c in json.loads((ROOT / "benchmarks/zelda/test721.json").read_text())["cases"]}
    for target in (ROOT / "configs/paper_m2_z64_target_ids.txt").read_text().splitlines():
        result.append({"suite": "z64", "id": target, "case": cases[target]})
    m7 = yaml.safe_load((ROOT / "configs/paper_m7_fixed_rng36.yaml").read_text())
    for task in m7["tasks"]:
        for arm, config in m7["arms"].items():
            result.append({"suite": "m7", "id": task["task_id"] + "__" + arm,
                           "task_id": task["task_id"], "arm": arm, "seed": task["seed"],
                           "instruction": task["instruction"], "config": config["config"],
                           "initial": task.get("initial", m7["default_initial"])})
    for row in result:
        row.update(model_id=model_id, model_sha256=MODEL_SHA256)
    assert len(result) == 132
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="qwen38-mac-preflight")
    parser.add_argument("--model-file", type=Path)
    parser.add_argument("--resume", action="store_true", help="Retry only interrupted processes; retain prior attempts")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--case-limit", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    if not args.cpu_only:
        if args.model_file is None or not args.model_file.is_file():
            parser.error("--model-file must point to the registered Q4_K_M GGUF")
        with args.model_file.open("rb") as model_file:
            digest = hashlib.file_digest(model_file, "sha256").hexdigest()
        if digest != MODEL_SHA256:
            parser.error("GGUF bytes do not match the registered model")
        from scripts.lmstudio_worker import verify_loaded_studio_model
        loaded = json.loads(subprocess.check_output(["lms", "ps", "--json"], text=True))
        verify_loaded_studio_model(loaded, args.model_id)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a") as run_lock:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = jobs(args.model_id)
        for job in frozen:
            job["execution_mode"] = "cpu_only" if args.cpu_only else "native_inference"
        if args.case_limit:
            frozen = frozen[:args.case_limit]
        frozen_path = output / "jobs.json"
        if frozen_path.exists():
            if not args.resume or json.loads(frozen_path.read_text()) != frozen:
                parser.error("Existing campaign: --resume requires the same model and case selection")
        else:
            frozen_path.write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n")
        ledger_path = output / "attempts.jsonl"
        prior = [json.loads(line) for line in ledger_path.read_text().splitlines()] if ledger_path.exists() else []
        completed = {row["key"] for row in prior if row["returncode"] == 0}
        failures = 0
        for job in frozen:
            key = job["suite"] + "/" + job["id"]
            if key in completed:
                continue
            attempt = 1 + sum(r["key"] == key for r in prior)
            # A process killed before ledger write also has an immutable attempt directory.
            while (output / key / f"attempt_{attempt:02}").exists():
                attempt += 1
            run_dir = output / key / f"attempt_{attempt:02}"
            run_dir.mkdir(parents=True, exist_ok=False)
            jobfile = run_dir / "job.json"
            jobfile.write_text(json.dumps(job, indent=2, ensure_ascii=False) + "\n")
            cmd = [sys.executable, str(ROOT / "scripts/lmstudio_worker.py"), "--job", str(jobfile), "--output", str(run_dir)]
            if args.cpu_only:
                cmd.append("--cpu-only")
            started = time.time()
            print(f"{key}: attempt {attempt}", flush=True)
            with (run_dir / "worker.log").open("w") as log:
                proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                      env={**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "vendor"))),
                                           "RPE_PCG_COORD_GRAMMAR": "on", "RPE_PCG_OPENAI_SEED": "on"})
            row = {"key": key, "attempt": attempt, "returncode": proc.returncode, "directory": str(run_dir),
                   "started_epoch": started, "seconds": time.time() - started, "ended_at": datetime.now(timezone.utc).isoformat()}
            with ledger_path.open("a") as ledger:
                ledger.write(json.dumps(row) + "\n")
            prior.append(row)
            failures += proc.returncode != 0
        print(f"Campaign artifacts: {output}; process failures this session: {failures}")
        if not args.cpu_only:
            from experiments.lmstudio_analysis import analyze
            analyze(output, output / "analysis")
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
