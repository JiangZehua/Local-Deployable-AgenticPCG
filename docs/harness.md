# Harness and configuration

[← README](../README.md) · [Experiment guide](reproduction.md)

The execution loop lives in `toolusepcg/optimizer/greedy.py`. It builds a prompt from the saved level and the selected observation profile, parses a structured response, executes candidate tools, evaluates the candidate, and either commits it or retains the previous saved level. The trace records this decision alongside the proposed actions and tool outcomes.

## Components

| Component | Location | Responsibility |
| --- | --- | --- |
| Configuration and profiles | `toolusepcg/config.py`, `configs/` | Environment, targets, scoring, tools, model settings and ablations |
| Action protocol | `toolusepcg/protocol/` | Typed messages, argument validation and coordinate-bounded output schemas |
| Execution controller | `toolusepcg/optimizer/greedy.py` | Prompt construction, bounded action recovery, evaluation, commit and stop |
| Level editing and PCG tools | `toolusepcg/tools/`, `toolusepcg/edit_manager.py` | Manual changes, constructive algorithms and higher-level operations |
| Metrics and validity | `toolusepcg/evaluator.py`, `toolusepcg/factory.py` | Domain metrics, hard validity and simulator integration |
| Language contracts | `toolusepcg/functional_resolver.py`, `experiments/paper_m1_smb_language.py` | Typed target fields, deterministic disposition and execution |
| Gameplay operations | `experiments/r100_*`, `r101_*`, `r102_*`, `r107_*` | Stomps, jumps, combined behavior and the compact mechanic library |
| Trace and visualization | `toolusepcg/logging/trace.py`, `render_trajectory.py`, `analyze_trace.py` | Saved action evidence and inspection tools |

Use `python main.py --help` for the base agent and `python run_experiment.py --help` for configurable framework experiments. The portable suites layer the paper's fixed conditions on top of these entry points.

## Preserve the protocol profile

Two explicit prompt profiles retain the interfaces used by the source experiments:

- `current` preserves the exact-control Zelda interface used by Z64 and the related component studies.
- `freeform` preserves the earlier Zelda structure/tool-use interface used by M7 and the scene study.

The suite launcher chooses the profile. For a standalone freeform run, pass `--prompt-profile freeform` explicitly. SMB's runtime tool-registry schema is separate from Zelda's frozen schema.

The environment names `RPE_PCG_COORD_GRAMMAR`, `RPE_PCG_OPENAI_SEED`, `RPE_PCG_ENABLE_THINKING` and `RPE_PCG_INTERACTION_PROTOCOL` remain for compatibility. The package itself is `experiments`, and no prompt-evolution manager is active. The registered schemas retain legacy wire fields where changing them would alter a frozen prompt.

## Acceptance, recovery and ablation

The full loop distinguishes candidate state from saved state. Validity and target error can take priority over the ordinary edit-cost objective, so a target-satisfying candidate is not discarded merely because it changed many tiles. Verified commitment, forced application and evaluator-exact stopping are selectable controls. Exact stopping checks the saved level, rather than trusting a model declaration of success.

The exact-control runner implements the registered truncation recovery: after an incomplete thinking response, a bounded follow-up call requests the actionable output with thinking disabled. This is not an unbounded retry loop. Call budgets, finish reasons, recovery counts and costs are retained in the per-case records.

Ablation modes include `full`, `no_evaluation_loop`, `no_metric_observation`, `no_historical_feedback` and `always_accept`. Their exact effects are implemented in the controller and tested by the component suites. Use `pcg-reproduce ladder --dry-run` and `pcg-reproduce commit-stop --dry-run` to inspect the controlled combinations.

## Domain coverage

The base framework supports Binary, BinaryDoor, Zelda, Sokoban, Lode Runner and SMB. Their configurations, tile encodings, environments and render assets are included. The shared generator package includes random filling, cellular automata, BSP, connectivity repair, digging, mazes, WFC and Zelda entity placement.

The SMB static extension uses the existing tile environment while adding a separate target contract and mandatory playability checks. Dynamic control measures A* simulation events. Both the environment's engine and the higher-level constructive operations are bundled, so exact outcomes are verified locally.

## Extending the project

Copy a domain configuration to start an exploratory task. Register a tool's typed arguments, execution behavior and evaluator-visible effects together. Update its domain schema when needed. Keep a candidate separate from saved state until the selected policy commits it, and retain actual tool failures in the trace.

Place new tasks outside the frozen paper manifests. If you change a prompt, tool inventory, simulator, target representation, seed policy or stopping rule, record that as a new protocol. The original suite runners and compact benchmark assets remain available for comparison.
