"""Portable launch plans for the paper's fixed experiments.

Use --dry-run to inspect commands without contacting a model server.
Every invocation uses a new output directory for experiment output.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "qwen38": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
    "qwen36": ("Qwen/Qwen3.6-35B-A3B-FP8", "95a723d08a9490559dae23d0cff1d9466213d989"),
    "gemma4": ("gemma4-31b-pcg", "842da3794eaa0b77d5f08bae87a17459d91ff475"),
}
SUITES = ("smoke", "zelda-oracle", "z64", "zelda-validation", "zelda-test", "ladder", "commit-stop",
          "binarydoor", "natural32", "language-validation", "language-test",
          "tools", "tool-loop", "scenes", "smb-baseline", "smb-pass4",
          "smb-stomp", "smb-jump", "smb-behavior", "smb-library",
          "smb-dynamic", "smb-static", "smb-static-baseline", "smb-static-language", "gemma4", "lmstudio")


def command(module: str, *args: object) -> list[str]:
    return [sys.executable, "-m", f"experiments.{module}", *map(str, args)]


def build_plan(args: argparse.Namespace) -> list[list[str]]:
    out = args.output.resolve()
    model, revision = MODELS[args.model]
    if args.model_id:
        model = args.model_id
    if args.revision:
        revision = args.revision
    common = ["--base-url", args.base_url, "--model-id", model, "--revision", revision]
    plan: list[list[str]] = []
    if args.suite == "smoke":
        return [command("smoke", "--output", out)]
    if args.suite == "zelda-oracle":
        return [command("zelda_mastery_m0", "--output", out)]
    if args.suite in {"z64", "ladder", "commit-stop", "zelda-validation", "zelda-test"}:
        if args.model == "gemma4":
            raise ValueError("Use the gemma4 suite, which preserves its model-specific reasoning interface.")
        subset = ROOT / "benchmarks/zelda/test721.json"
        count = 721 if args.suite == "zelda-test" else 64
        if args.suite == "zelda-validation":
            subset = out / "validation120.json"
            count = 120
            plan.append([sys.executable, str(ROOT / "scripts/build_zelda_validation.py"), "--output", str(subset)])
        cells = [("open_loop", "no_evaluation_loop", False, False, []),
                 ("full", "full", True, True, [])]
        if args.suite in {"zelda-validation", "zelda-test"}:
            cells = cells[1:]
        elif args.suite == "ladder":
            cells = [cells[0],
                     ("recovery", "no_evaluation_loop", True, False, []),
                     ("accept_stop", "no_metric_observation", True, False, []),
                     ("feedback", "no_historical_feedback", True, False, []),
                     ("history", "full", True, False, []), cells[-1]]
        elif args.suite == "commit-stop":
            cells = [(f"{policy}_{stop}", "no_metric_observation", True, False,
                      ["--commit-policy", policy, "--evaluator-exact-stop", stop])
                     for policy in ("force_apply", "verified") for stop in ("off", "on")]
        targets = []
        if args.suite in {"z64", "ladder", "commit-stop"}:
            for target in (ROOT / "configs/paper_m2_z64_target_ids.txt").read_text().splitlines():
                targets += ["--target-id", target]
        for name, ablation, recovery, geometry, switches in cells:
            flags = ["--thinking-max-tokens", "6144"] if recovery else []
            if geometry:
                flags += ["--extra-instruction-file", str(ROOT / "prompts/zelda_geometry_serialization_rules_v3.txt")]
            plan.append(command("r061_qwen38_m1", "prompt-probe", *common,
                                "--subset", subset, "--arm", name,
                                "--case-limit", count, "--max-steps", 4,
                                "--conversation-length", 2, "--temperature", 0,
                                "--request-seed-offset", 0, "--max-tokens", 8192,
                                "--ablation-mode", ablation, "--truncation-recovery",
                                "thinking_off_handoff" if recovery else "none",
                                *flags, *switches, *targets, "--output", out / f"{name}.json"))
        return plan
    if args.suite == "binarydoor":
        return [command("paper_m3_binarydoor", "run", *common,
                        "--manifest", ROOT / "configs/paper_m3_binarydoor_manifest.json",
                        "--arm", arm, "--output", out / f"{arm}.json") for arm in ("h0", "h3")]
    if args.suite in {"natural32", "language-validation", "language-test", "smb-static-language"}:
        manifest = ROOT / {
            "natural32": "configs/paper_m6_natural32.json",
            "language-validation": "benchmarks/smb/template_validation.json",
            "language-test": "benchmarks/smb/template_test.json",
            "smb-static-language": "benchmarks/smb/static_language.json",
        }[args.suite]
        if args.suite == "natural32":
            plan.append(command("paper_m6_natural32", "preflight", "--manifest", manifest,
                                "--require-template-audit", "--output", out / "preflight.json"))
        plan += [command("paper_m1_smb_language", "structured-ceiling", "--manifest", manifest,
                         "--builder", "stitch_v2", "--output", out / "ceiling.json"),
                 command("paper_m1_smb_language", "run-language", *common, "--manifest", manifest,
                         "--builder", "stitch_v2", "--output", out / "model.json")]
        if args.suite == "natural32":
            plan.append(command("paper_m6_natural32", "combine", "--manifest", manifest,
                                "--model-result", out / "model.json", "--structured-ceiling", out / "ceiling.json",
                                "--output", out / "evaluation.json"))
        return plan
    if args.suite in {"tools", "tool-loop"}:
        import yaml
        manifest = yaml.safe_load((ROOT / "configs/paper_m7_fixed_rng36.yaml").read_text())
        arms = [("all_tools", "full", "all"), ("manual_only", "full", "manual")]
        if args.suite == "tool-loop":
            arms = [("all_tools", "full", "all"), ("no_evaluation_loop", "no_evaluation_loop", "all")]
        for task in manifest["tasks"]:
            for arm, ablation, tools in arms:
                plan.append(command("run_seeded_entrypoint", "--process-seed", task["seed"],
                    "--entrypoint", ROOT / "main.py", "--config", ROOT / f"configs/zelda_pcg_tools_32x32_{tools}.yaml",
                    "--init", ROOT / task.get("initial", manifest["default_initial"]),
                    "--seed", task["seed"], "--max-steps", 8, "--max-tokens", 4096,
                    "--provider", "openai", "--model", model, "--base-url", args.base_url,
                    "--temperature", 0, "--extra-instruction", task["instruction"],
                    "--ablation", ablation, "--prompt-profile", "freeform", "--no-render-stats",
                    "--no-render-sim", "--quiet", "--run-dir", out / task["task_id"] / arm))
        plan.append(command("tool_evaluation", "--suite", args.suite, "--run-root", out,
                            "--output", out / "evaluation.json"))
        return plan
    if args.suite == "scenes":
        for arm in ("full", "no_eval", "no_history", "no_metrics", "always_accept"):
            plan.append(command("high_level_freeform_eval", "run",
                "--suite", ROOT / "configs/high_level_freeform_zelda_t1_t4_32x32.yaml",
                "--config", ROOT / f"configs/zelda_freeform_32x32_{arm}.yaml",
                "--run-dir", out / arm, "--model", model, "--base-url", args.base_url,
                "--provider", "openai", "--max-tokens", 4096, "--temperature", 0))
        plan.append(command("zelda_freeform_ablation_eval", *sum(
            (["--arm", f"{arm}={out / arm}"] for arm in
             ("full", "no_eval", "no_history", "no_metrics", "always_accept")), []),
            "--output", out / "evaluation.json"))
        return plan
    smb_modules = {"smb-baseline": "r083_smb_baseline", "smb-stomp": "r100_smb_stomp_operator",
                   "smb-jump": "r101_smb_jump_operator", "smb-behavior": "r102_smb_behavior_operator",
                   "smb-library": "r107_smb_mechanic_library", "smb-static-baseline": "r098_smb_static_baseline"}
    if args.suite in smb_modules:
        module = smb_modules[args.suite]
        return [command(module, "preflight", "--output", out / "preflight.json"),
                command(module, "run", *common, "--output", out / "result.json")]
    if args.suite == "smb-pass4":
        for i in range(4):
            plan.append(command("r083_smb_baseline", "run", *common, "--seed", 99000 + i, "--temperature", 0.7,
                                "--output", out / f"sample_{i}.json"))
        plan.append(command("r099_smb_pass_at_4", *sum(
            (["--sample", str(out / f"sample_{i}.json")] for i in range(4)), []),
            "--output", out / "evaluation.json"))
        return plan
    if args.suite == "smb-dynamic":
        return [command("paper_m8_smb_dynamic_builders", "grid", "--arms", "stitch_v2",
                        "--workers", getattr(args, "workers", 1), "--output", out / "grid.json"),
                command("paper_m8_smb_dynamic_builders", "audit", "--grid", out / "grid.json",
                        "--workers", getattr(args, "workers", 1), "--output", out / "takeoff_audit.json"),
                command("paper_m8_smb_dynamic_builders", "summary", "--grid", out / "grid.json",
                        "--output", out / "summary.json")]
    if args.suite == "smb-static":
        return [command("r109_smb_static_playability", "--output", out / "gate.json")]
    if args.suite == "gemma4":
        return [command("gemma4_z64_hpc", "interface-probe", "--base-url", args.base_url,
                        "--output", out / "interface_probe.json"),
                command("gemma4_z64_hpc", "run", "--mode", args.gemma_mode, "--run-id", out.name,
                        "--source-root", ROOT, "--output", out, "--base-url", args.base_url),
                command("gemma4_z64_hpc", "analyze", "--mode", args.gemma_mode, "--run-id", out.name,
                        "--source-root", ROOT, "--output", out)]
    if args.suite == "lmstudio":
        if not args.gguf:
            raise ValueError("The native LM Studio suite requires --gguf /path/to/Qwen3.8-27B-Q4_K_M.gguf")
        return [command("lmstudio_campaign", "--output", out, "--model-id", args.model_id or "qwen38-mac-preflight",
                        "--model-file", args.gguf.resolve())]
    raise ValueError(args.suite)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=SUITES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", choices=MODELS, default="qwen38")
    parser.add_argument("--model-id", help="Override the server's model identifier")
    parser.add_argument("--revision", help="Record the exact checkpoint or GGUF revision")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--gemma-mode", choices=["dev", "formal"], default="formal")
    parser.add_argument("--gguf", type=Path, help="Registered Q4_K_M file, hashed before the native LM Studio suite")
    parser.add_argument("--workers", type=int, default=1, help="CPU processes for the SMB dynamic grid")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.model_id and not args.revision and args.suite != "lmstudio":
        parser.error("--model-id requires --revision to keep model provenance explicit")
    if args.suite == "gemma4" and (args.model_id or args.revision):
        parser.error("The gemma4 suite reads model identity from configs/gemma4_z64_protocol.json")
    args.output = (args.output or Path.cwd() / "runs" / f"{args.suite}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}").resolve()
    try:
        plan = build_plan(args)
    except ValueError as error:
        parser.error(str(error))
    thinking = "on" if args.suite in {"z64", "ladder", "commit-stop", "zelda-validation", "zelda-test", "binarydoor", "gemma4"} else "off"
    overrides = {"RPE_PCG_COORD_GRAMMAR": "on", "RPE_PCG_OPENAI_SEED": "on", "RPE_PCG_ENABLE_THINKING": thinking,
                 "RPE_PCG_INTERACTION_PROTOCOL": "v1", "RPE_PCG_DUPLICATE_LEVEL_GUARD": "off"}
    if args.dry_run:
        print(json.dumps({"cwd": str(ROOT), "env": overrides, "commands": plan}, indent=2, ensure_ascii=False))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    failures = []
    env = {**os.environ, **overrides, "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "vendor")))}
    # Preserve all attempted cells. A failed process does not silently disappear from the run.
    for index, cmd in enumerate(plan, 1):
        print(f"[{index}/{len(plan)}] {shlex.join(cmd)}", flush=True)
        with (args.output / f"command_{index:03}.log").open("w") as log:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            failures.append(index)
            print(f"Command {index} failed; see {args.output / f'command_{index:03}.log'}", file=sys.stderr)
            # Tool/scene episodes must still reach the denominator-preserving evaluator.
            if args.suite not in {"tools", "tool-loop", "scenes"}:
                raise SystemExit(result.returncode)
    print(f"Artifacts: {args.output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
