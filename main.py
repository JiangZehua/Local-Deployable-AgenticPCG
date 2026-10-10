#!/usr/bin/env python3
"""Main entry point for single-run ToolUsePCG optimization.

For multi-trial experiments, use run_experiment.py instead.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from toolusepcg.config import load_config, config_from_dict, deep_merge
from toolusepcg.optimizer import GreedyOptimizer
from toolusepcg.logging import TraceLogger
from toolusepcg.resume import load_resume_state


def main():
    parser = argparse.ArgumentParser(
        description="Run single LLM agent optimization on a game level. "
                    "For multi-trial experiments, use run_experiment.py instead."
    )
    parser.add_argument("--prompt-profile", choices=["current", "freeform"], default="current")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for level generation (overrides config)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum optimization steps (default: 100)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Output directory for run (default: runs/{timestamp}_{problem_type})",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["portkey", "openai", "deepseek", "google", "anthropic", "transformers"],
        default=None,
        help="LLM provider: 'openai' (default, for local vLLM/TGI/etc.), 'portkey', "
             "'deepseek' (api.deepseek.com), 'google' (Google Genai SDK), "
             "'anthropic' (Anthropic SDK), or 'transformers' (local HF ckpt).",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="For --provider transformers: base model path/id that a LoRA/QLoRA "
             "adapter was trained on top of. Ignored if --model is a full-FT ckpt.",
    )
    parser.add_argument(
        "--chat-template-path",
        type=str,
        default=None,
        help="For --provider transformers: path to a .jinja chat template that "
             "overrides the tokenizer's default (e.g. drop Qwen3.5's <think>).",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="For --provider transformers: 4-bit NF4 quantize the base weights.",
    )
    parser.add_argument(
        "--grammar-domain",
        type=str,
        default=None,
        help="For --provider transformers: enable xgrammar token-level schema "
             "constraint built from this domain (e.g. 'binarydoor'). Forces the "
             "model to emit a valid STEP JSON.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model to use (overrides config)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="LLM API base URL (overrides config, e.g., http://localhost:8000/v1 for vLLM)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="LLM API key (overrides config; for vLLM, defaults to 'EMPTY' if not set)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max completion tokens (overrides config; lower for vLLM where max_tokens + prompt must fit in context)",
    )
    parser.add_argument(
        "--init",
        type=str,
        default=None,
        help="Level initialization strategy: 'random' (default), 'empty' (all passable tiles), "
             "'filled' (all blocking tiles), 'weighted' (weighted random: common tiles favored, "
             "special tiles rare), or a file path to an ASCII level file",
    )
    parser.add_argument(
        "--initial-level",
        type=str,
        default=None,
        help="(Deprecated: use --init instead) Path to file with initial level (ASCII format)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Enable verbose step output (default: True)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Disable verbose step output",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: save full LLM prompts/responses to debug.log and show raw responses in terminal",
    )
    parser.add_argument(
        "--target",
        nargs=2,
        action="append",
        metavar=("METRIC", "VALUE"),
        help="Override target metric value (can be repeated, e.g., --target player_key 15 --target key_door 12)",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Optional experiment name for output directory naming",
    )
    parser.add_argument(
        "--render-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay metrics/targets on rendered level images (default: True, use --no-render-stats to disable)",
    )
    parser.add_argument(
        "--render-sim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render simulation GIF + trajectory image for SMB, simulation GIF for Zelda (default: True, use --no-render-sim to disable)",
    )
    parser.add_argument(
        "--edit-tools",
        nargs="*",
        default=None,
        metavar="TOOL",
        help="Edit tools to provide to the agent (e.g., place_single_tile place_line place_patch)",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=None,
        help="Max tool calls per optimization step (overrides config optimizer.max_tool_calls_per_step)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Designer sampling temperature (overrides config llm.temperature)",
    )
    parser.add_argument(
        "--init-strategy",
        default=None,
        choices=["random", "weighted_random", "empty", "filled"],
        help="Initial level strategy (overrides config optimizer.init_strategy)",
    )
    parser.add_argument(
        "--acceptance-strategy",
        choices=["greedy", "annealing", "llm_triage"],
        default=None,
        help="Optimizer acceptance strategy (overrides config optimizer.strategy)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Probability of accepting a worse solution in greedy strategy (overrides config optimizer.epsilon_accept_worse)",
    )
    parser.add_argument(
        "--sa-temp",
        type=float,
        default=None,
        help="Simulated annealing initial temperature (enables SA strategy when set; e.g. 10.0)",
    )
    parser.add_argument(
        "--sa-decay",
        type=float,
        default=None,
        help="Simulated annealing decay rate per step (default: 0.95); only used when --sa-temp is set",
    )
    parser.add_argument(
        "--triage-model",
        default=None,
        help="llm_triage acceptance: independent triage model (default: main LLM model)",
    )
    parser.add_argument(
        "--triage-base-url",
        default=None,
        help="llm_triage acceptance: OpenAI-compatible endpoint (default: main LLM base URL)",
    )
    parser.add_argument(
        "--triage-max-tokens",
        type=int,
        default=None,
        help="llm_triage acceptance: max tokens for the triage decision",
    )
    parser.add_argument(
        "--triage-min-confidence",
        type=float,
        default=None,
        help="llm_triage acceptance: minimum confidence required for decision=accept",
    )
    parser.add_argument(
        "--conv-length",
        type=int,
        default=None,
        help="Number of conversation turns to keep (1 = no history, default from config)",
    )
    parser.add_argument(
        "--conv-filter",
        type=str,
        choices=["all", "accepted"],
        default=None,
        help="Which steps to include in conversation history: 'all' or 'accepted' (default from config)",
    )
    parser.add_argument(
        "--change-percentage",
        type=float,
        default=None,
        help="Change budget multiplier for termination (budget = multiplier * height * width; overrides config termination.change_budget_multiplier)",
    )
    parser.add_argument(
        "--smb-solver",
        type=str,
        choices=["auto", "astar"],
        default=None,
        help="SMB simulation solver: 'auto' (default: heuristic then A*) or 'astar' (A* only)",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        choices=["full", "no_historical_feedback", "no_metric_observation", "no_evaluation_loop"],
        default=None,
        help="Optimizer ablation mode (default: from config, usually full)",
    )
    parser.add_argument(
        "--extra-instruction",
        type=str,
        default=None,
        help="Extra high-level instruction appended to the LLM prompt (e.g., 'make the layout look like the letter A')",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        help="Inner-loop block size (overrides config optimizer.rollout_steps). 1 = per-step accept/reject (default); "
             "k>1 = roll out k tentative steps then make ONE block accept/reject decision.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to previous run directory to resume from",
    )
    parser.add_argument(
        "--resume-step",
        type=int,
        default=None,
        help="Step to resume from (default: last step). Requires --resume-from",
    )

    metric_input = parser.add_mutually_exclusive_group()
    metric_input.add_argument("--request", help="Natural-language metric targets for the configured game")
    metric_input.add_argument("--request-spec", type=Path, help="Metric-request JSON file (no parser call)")
    parser.add_argument("--request-dry-run", action="store_true", help="Print resolved targets and exit before generation")
    args = parser.parse_args()
    if args.request_dry_run and not (args.request or args.request_spec):
        parser.error("--request-dry-run requires --request or --request-spec")
    if (args.request or args.request_spec) and (args.target or args.resume_from):
        parser.error("Metric requests cannot be combined with --target or --resume-from")

    # Validate resume args
    if args.resume_step is not None and args.resume_from is None:
        parser.error("--resume-step requires --resume-from")

    # Build config overrides from CLI arguments
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed

    # LLM overrides (provider, model, base_url, api_key)
    llm_overrides = {}
    if args.provider is not None:
        llm_overrides["provider"] = args.provider
    if args.model is not None:
        llm_overrides["model"] = args.model
    if args.base_url is not None:
        llm_overrides["base_url"] = args.base_url
    if args.api_key is not None:
        llm_overrides["api_key"] = args.api_key
    if args.max_tokens is not None:
        llm_overrides["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        llm_overrides["temperature"] = args.temperature
    if args.base_model is not None:
        llm_overrides["base_model"] = args.base_model
    if args.chat_template_path is not None:
        llm_overrides["chat_template_path"] = args.chat_template_path
    if args.load_in_4bit:
        llm_overrides["load_in_4bit"] = True
    if args.grammar_domain is not None:
        llm_overrides["grammar_domain"] = args.grammar_domain
    if llm_overrides:
        overrides["llm"] = llm_overrides

    # Handle target overrides from --target arguments
    if args.target:
        targets_overrides = {}
        for metric, value in args.target:
            try:
                parsed_value = int(value) if '.' not in value else float(value)
                targets_overrides[metric] = {"mode": "target", "value": parsed_value}
            except ValueError:
                print(f"Warning: Could not parse target value '{value}' for metric '{metric}'")

        if targets_overrides:
            overrides["targets"] = targets_overrides

    # Handle edit tools override
    if args.edit_tools is not None:
        overrides.setdefault("optimizer", {})["edit_tools"] = args.edit_tools

    # Handle max tool calls override
    if args.max_tool_calls is not None:
        overrides.setdefault("optimizer", {})["max_tool_calls_per_step"] = args.max_tool_calls

    # Handle init strategy override
    if args.init_strategy is not None:
        overrides.setdefault("optimizer", {})["init_strategy"] = args.init_strategy

    # Handle epsilon accept worse override
    if args.epsilon is not None:
        overrides.setdefault("optimizer", {})["epsilon_accept_worse"] = args.epsilon

    # Handle acceptance strategy override
    if args.acceptance_strategy is not None:
        overrides.setdefault("optimizer", {})["strategy"] = args.acceptance_strategy

    # Handle simulated annealing overrides
    if args.sa_temp is not None:
        overrides.setdefault("optimizer", {})["strategy"] = "annealing"
        overrides.setdefault("optimizer", {})["annealing_initial_temp"] = args.sa_temp
    if args.sa_decay is not None:
        overrides.setdefault("optimizer", {})["annealing_decay_rate"] = args.sa_decay
    if args.triage_model is not None:
        overrides.setdefault("optimizer", {})["triage_model"] = args.triage_model
    if args.triage_base_url is not None:
        overrides.setdefault("optimizer", {})["triage_base_url"] = args.triage_base_url
    if args.triage_max_tokens is not None:
        overrides.setdefault("optimizer", {})["triage_max_tokens"] = args.triage_max_tokens
    if args.triage_min_confidence is not None:
        overrides.setdefault("optimizer", {})["triage_min_confidence"] = args.triage_min_confidence

    # Handle conversation history overrides
    if args.conv_length is not None:
        overrides.setdefault("optimizer", {})["conversation_length"] = args.conv_length
    if args.conv_filter is not None:
        overrides.setdefault("optimizer", {})["conversation_filter"] = args.conv_filter
    if args.ablation is not None:
        overrides.setdefault("optimizer", {})["ablation_mode"] = args.ablation

    # Handle change percentage override
    if args.change_percentage is not None:
        overrides.setdefault("termination", {})["change_budget_multiplier"] = args.change_percentage

    # Handle SMB solver override
    if args.smb_solver is not None:
        overrides.setdefault("env", {})["smb_solver"] = args.smb_solver

    # Handle extra instruction override
    if args.extra_instruction is not None:
        overrides.setdefault("optimizer", {})["extra_instruction"] = args.extra_instruction

    # Handle rollout-steps (inner-loop block size) override
    if args.rollout_steps is not None:
        overrides.setdefault("optimizer", {})["rollout_steps"] = args.rollout_steps

    # --- Resume or fresh config loading ---
    resume_state = None
    if args.resume_from:
        try:
            resume_state = load_resume_state(args.resume_from, args.resume_step)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading resume state: {e}")
            return 1
        print(f"Resuming from: {args.resume_from} (step {resume_state.resume_step})")

        # Build config from saved run, then apply CLI overrides on top
        config_dict = resume_state.config_dict
        if overrides:
            config_dict = deep_merge(config_dict, overrides)
        try:
            config = config_from_dict(config_dict)
        except Exception as e:
            print(f"Error loading config from resume state: {e}")
            return 1
    else:
        # Normal config loading
        try:
            config = load_config(args.config, overrides)
        except Exception as e:
            print(f"Error loading config: {e}")
            return 1

    metric_request_payload = None
    if args.request or args.request_spec:
        from toolusepcg.metric_input import apply_metric_request, parse_metric_request
        from toolusepcg.config import config_to_dict
        try:
            if args.request_spec:
                metric_request_payload = json.loads(args.request_spec.read_text())
                config = apply_metric_request(config, metric_request_payload)
            else:
                if config.llm.provider != "openai":
                    raise ValueError("--request uses an OpenAI-compatible endpoint; use --provider openai or --request-spec")
                from openai import OpenAI
                with OpenAI(base_url=config.llm.base_url,
                            api_key=config.llm.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY",
                            max_retries=0, timeout=120.0) as client:
                    config, metric_request_payload = parse_metric_request(args.request, config, client)
            print(json.dumps({"metric_request": metric_request_payload,
                              "resolved_targets": config_to_dict(config)["targets"]},
                             ensure_ascii=False, indent=2))
        except Exception as exc:
            print(f"Error resolving metric request: {exc}", file=sys.stderr)
            return 1
        if args.request_dry_run:
            return 0

    # Use seed from config if not specified on command line
    seed = args.seed if args.seed is not None else config.seed

    # Collect runtime args for saving alongside resolved config
    # Only include non-null, non-default values to keep output clean
    runtime_args = {
        "max_steps": args.max_steps,
    }
    if metric_request_payload is not None:
        runtime_args["metric_request"] = metric_request_payload
        runtime_args["metric_request_text"] = args.request
    if args.config:
        runtime_args["config_path"] = args.config
    if args.verbose and not args.quiet:
        runtime_args["verbose"] = True
    if args.debug:
        runtime_args["debug"] = True
    if args.render_stats is not True:
        runtime_args["render_stats"] = args.render_stats
    if not args.render_sim:
        runtime_args["render_sim"] = False
    # Resolve init strategy: --init takes precedence over --initial-level
    init_strategy = args.init or (args.initial_level if args.initial_level else "random")
    if init_strategy != "random":
        runtime_args["init"] = init_strategy
    if args.conv_length is not None:
        runtime_args["conv_length"] = args.conv_length
    if args.conv_filter is not None:
        runtime_args["conv_filter"] = args.conv_filter
    if args.acceptance_strategy is not None:
        runtime_args["acceptance_strategy"] = args.acceptance_strategy
    if args.triage_model is not None:
        runtime_args["triage_model"] = args.triage_model
    if args.triage_base_url is not None:
        runtime_args["triage_base_url"] = args.triage_base_url
    if args.triage_max_tokens is not None:
        runtime_args["triage_max_tokens"] = args.triage_max_tokens
    if args.triage_min_confidence is not None:
        runtime_args["triage_min_confidence"] = args.triage_min_confidence
    if args.ablation is not None:
        runtime_args["ablation"] = args.ablation
    if args.extra_instruction:
        runtime_args["extra_instruction"] = args.extra_instruction
    if args.exp_name:
        runtime_args["exp_name"] = args.exp_name
    if resume_state is not None:
        runtime_args["resume_from"] = str(args.resume_from)
        runtime_args["resume_step"] = resume_state.resume_step

    # Initialize logger
    try:
        logger = TraceLogger(config, run_dir=args.run_dir, exp_name=args.exp_name,
                             render_stats=args.render_stats, render_sim=args.render_sim,
                             runtime_args=runtime_args)
    except Exception as e:
        print(f"Error initializing logger: {e}")
        return 1

    print(f"Run directory: {logger.get_run_dir()}")

    # Show target overrides if any were provided
    if args.target:
        print("\nTarget overrides:")
        for metric, value in args.target:
            print(f"  {metric}: {value}")

    config.optimizer.prompt_profile = args.prompt_profile

    # Initialize optimizer
    verbose = args.verbose and not args.quiet
    try:
        optimizer = GreedyOptimizer(config, logger=logger, verbose=verbose, debug=args.debug)
    except Exception as e:
        print(f"Error initializing optimizer: {e}")
        return 1

    # Set environment for rendering
    logger.set_env(optimizer.factory.env)

    if resume_state is not None:
        # --- Resume path: use level from trace ---
        initial_level = optimizer.factory.create_from_string(resume_state.level_str)
        print(f"Resuming with level from step {resume_state.resume_step}")

        # Initialize optimizer with the resumed level
        try:
            optimizer.initialize(seed=seed, initial_level=initial_level)
        except Exception as e:
            print(f"Error during initialization: {e}")
            return 1

        # Restore state from resume point
        optimizer.step_count = resume_state.resume_step
        optimizer.termination.set_total_changes(resume_state.total_changes)

        # Re-evaluate to get fresh current_result and current_score
        optimizer.current_result = optimizer.evaluator.evaluate(optimizer.current_level)
        optimizer.current_score = optimizer.scorer.score(optimizer.current_result)

    else:
        # --- Normal path: resolve initial level from init strategy ---
        initial_level = None
        if init_strategy == "empty":
            initial_level = optimizer.factory.create_empty()
            print(f"Using empty initialization (all passable tiles)")
        elif init_strategy == "filled":
            initial_level = optimizer.factory.create_filled()
            print(f"Using filled initialization (all blocking tiles)")
        elif init_strategy == "weighted":
            initial_level = optimizer.factory.create_weighted_random(seed)
            print(f"Using weighted random initialization (common tiles favored)")
        elif init_strategy != "random":
            # Treat as file path
            initial_level_path = Path(init_strategy)
            if initial_level_path.exists():
                try:
                    with open(initial_level_path) as f:
                        initial_level = optimizer.factory.create_from_string(f.read())
                    print(f"Loaded initial level from: {initial_level_path}")
                except Exception as e:
                    print(f"Error loading initial level: {e}")
                    return 1
            else:
                print(f"Error: Initial level file not found: {initial_level_path}")
                return 1

        # Initialize optimizer
        print(f"Initializing with seed={seed}")
        try:
            optimizer.initialize(seed=seed, initial_level=initial_level)
        except Exception as e:
            print(f"Error during initialization: {e}")
            return 1

    # Show initial state
    print("\nInitial Level:")
    print(optimizer.current_level.to_string())
    print(f"\nInitial Score: {optimizer.current_score:.2f}")
    print(f"Initial Metrics: {optimizer.current_result.metrics}")
    print(f"\nChange Budget: {optimizer.termination.change_budget} tiles")
    if resume_state is not None:
        print(f"Resumed Changes: {resume_state.total_changes} tiles (from prior run)")
        print(f"Remaining Budget: {optimizer.termination.remaining_budget} tiles")

    # Run optimization
    remaining = args.max_steps - optimizer.step_count
    print(f"\nStarting optimization (max {args.max_steps} total steps, {remaining} remaining)...")
    print("-" * 50)

    try:
        summary = optimizer.run(max_steps=args.max_steps)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        total_input_tokens = sum(r.input_tokens for r in optimizer.step_records)
        total_output_tokens = sum(r.output_tokens for r in optimizer.step_records)
        summary = {
            "total_steps": optimizer.step_count,
            "final_score": optimizer.current_score,
            "final_metrics": optimizer.current_result.metrics if optimizer.current_result else {},
            "termination_status": optimizer.termination.status(),
            "interrupted": True,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        }
        logger.log_final_level(optimizer.current_level, optimizer.current_result)
        logger.log_summary(summary)
    except Exception as e:
        import traceback
        print(f"\nError during optimization: {e}")
        traceback.print_exc()
        return 1

    # Show final state
    print("-" * 50)
    print("\nFinal Level:")
    print(optimizer.current_level.to_string())
    print(f"\nFinal Score: {summary['final_score']:.2f}")
    print(f"Final Metrics: {summary['final_metrics']}")
    print(f"\nTotal Steps: {summary['total_steps']}")
    print(f"Termination: {summary['termination_status']}")
    if summary.get('total_tokens', 0) > 0:
        print(f"Tokens: {summary['total_input_tokens']:,} input / {summary['total_output_tokens']:,} output / {summary['total_tokens']:,} total")

    print(f"\nResults saved to: {logger.get_run_dir()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
