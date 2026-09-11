# Reproducing the experiments

[← README](../README.md) · [Deployment](deployment.md) · [Harness](harness.md)

Run commands from the repository root after the installation in the README. All examples below use a local OpenAI-compatible server at `http://127.0.0.1:8000/v1`, unless they explicitly select the native LM Studio backend. CPU gates do not contact a server.

## Choose a protocol, then inspect its plan

```bash
pcg-reproduce --help
pcg-reproduce z64 --model qwen38 --dry-run
pcg-reproduce tools --output runs/tools --dry-run
```

`--dry-run` prints the subprocess commands, working directory and protocol environment without creating an output directory or making model calls. Actual runs require a new output directory. Reusing a completed directory is an error; choose a new name for a new repetition.

The default model is `qwen38`. `--model qwen36` selects the registered FP8 model. For another checkpoint, provide both `--model-id` and `--revision`; those results are a new configuration rather than the paper's registered comparison. Gemma uses the separate `gemma4` suite and its model-specific interface probe and protocol file.

## Exact geometry and harness components

| Command after `pcg-reproduce` | What runs | Output files |
| --- | --- | --- |
| `zelda-oracle` | Deterministic construction of all 31 × 31 path pairs; contract, split and record-schema checks | `r060_*` oracle and gate files |
| `z64 --model qwen38` | Open-loop and full harness on the same 64 fixed targets | `open_loop.json`, `full.json` |
| `z64 --model qwen36` | The same paired protocol with Qwen3.6 | `open_loop.json`, `full.json` |
| `gemma4` | Interface qualification, then both arms on Z64, then analysis | Model-specific raw records and analysis |
| `ladder` | Open-loop → recovery → acceptance/stop → current feedback → history → geometry guidance | One JSON file per cell |
| `commit-stop` | `force_apply` / `verified` × exact stop `off` / `on`, with metric observations withheld | Four cell JSON files |
| `zelda-validation` | All 120 fixed validation targets, full harness | `validation120.json`, `full.json` |
| `zelda-test` | All 721 fixed test targets, full harness | `full.json` |
| `binarydoor` | 32 targets with the registered h0 and h3 conditions | `h0.json`, `h3.json` |

Zelda's supported target range is 10–40 for each of player→key and key→door distance. A valid level also has exactly one player, key and door, three enemies, one connected passable region, and reachable required paths. A perfect match requires validity **and** both requested distances. The oracle establishes feasibility; it is not a model baseline with inference costs.

Z64 IDs are listed in `configs/paper_m2_z64_target_ids.txt`. Per-case seeds and initialization hashes come from `benchmarks/zelda/test721.json`. The validation materializer uses seeds `64120 + index`. Both arms of a paired comparison use the same IDs, maps and request seeds.

The paired Qwen protocol allows four semantic actions, up to 20 tool calls per action, two retained conversation turns and temperature 0. The open-loop arm has an 8,192-token call ceiling with no recovery. The full arm uses a 6,144-token thinking call and a bounded 8,192-token thinking-off recovery when needed, plus `zelda_geometry_serialization_rules_v3.txt`. Recovery calls consume inference resources and are included in the recorded costs. The ladder and commit/stop suite select these switches explicitly in `experiments/reproduce.py`.

For the matched Qwen comparison:

```bash
python -m experiments.paired_stats \
  --before runs/z64-qwen38/open_loop.json \
  --after runs/z64-qwen38/full.json \
  --output runs/z64-qwen38/paired_statistics.json
```

This computes the difference in exact-success rate, 200,000 paired bootstrap resamples with seed `20260831`, and a two-sided exact McNemar test. IDs, targets, seeds and initialization hashes must match; missing observations and duplicate IDs are rejected. Confirm the reported pair count is 64 for Z64. Gemma retains its original model-specific completeness and paired analysis in `experiments/gemma4_z64_hpc.py`.

For a smaller Gemma qualification run, use `pcg-reproduce gemma4 --gemma-mode dev --output runs/gemma-dev`. The eight development IDs are separate from the formal 64-target comparison.

## Language requests and simulated gameplay

| Suite | Fixed cases / purpose |
| --- | --- |
| `natural32` | 24 valid requests + 8 ambiguous/unsupported guards; template-reuse audit, structured ceiling, model parsing and combination |
| `language-validation` | 29 valid requests + 8 guards from the frozen validation partition |
| `language-test` | 38 valid requests + 10 guards from the frozen test partition |
| `smb-baseline` | Primitive-action agent on 29 development targets |
| `smb-pass4` | Four primitive-action samples; seeds 99000–99003, temperature 0.7; pass-at-four aggregation |
| `smb-stomp` | Six development controls for the stomp operation |
| `smb-jump` | Six development controls for the ground-jump operation |
| `smb-behavior` | Six development controls for the combined behavior operation |
| `smb-library` | The 16-backbone mechanic library, including its preflight and model interface |
| `smb-static` | 15 deterministic validity/playability cases, no LLM |
| `smb-static-baseline` | Eight static primitive-editing cases |
| `smb-static-language` | Seven valid static requests + one guard |

For language suites, `ceiling.json` measures execution from the manifest's ground-truth structured fields. `model.json` measures the complete language-to-fields-to-level path. Natural-32 additionally writes `preflight.json` and `evaluation.json`, including a rule baseline and fail-closed checks. Its original scientific gate includes the registered model identity; a substituted model can produce a useful report without satisfying that provenance gate.

Dynamic SMB targets count **enemies defeated, coins collected, and voluntary ground jumps during the A* playthrough**. Stomp bounces are not ground jumps. Static targets count **pipe components, enemy tiles, coin tiles and block tiles**, while also requiring the level to pass the static structural checks and the simulated completion/coin-reachability gate. These contracts are intentionally separate.

The resolver tries supported deterministic construction and the compact mechanic library, with bounded search for supported fallback cases. Its fixed backbones and search seeds are development inputs. Neither language test targets nor completed test trajectories are used as runtime templates.

To regenerate a structured ceiling alone:

```bash
python -m experiments.paper_m1_smb_language structured-ceiling \
  --manifest benchmarks/smb/template_test.json \
  --output runs/test-ceiling.json
```

The original specialized modules retain their `--help` interfaces for atlas generation, engine diagnostics and individual preflights. Use the named suites for the paper conditions, rather than inferring settings from historical module numbers.

## High-level tools and scene requests

```bash
pcg-reproduce tools --output runs/tools
pcg-reproduce tool-loop --output runs/tool-loop
pcg-reproduce scenes --output runs/scenes
```

`tools` compares `all_tools` with `manual_only`: six task families × three paraphrases × two arms = 36 episodes. `tool-loop` keeps the tools fixed and compares the full loop with `no_evaluation_loop`, again 36 episodes. Both use the fixed M7 manifest, initial maps and family seeds, eight actions, 4,096 output tokens per call and temperature 0. Python and NumPy RNG state is initialized through the preserved seeded entry point. Three paraphrases within a family share a map and seed; they are not independent seed replications.

Both commands write one run directory per task/arm and an `evaluation.json`. Evaluation distinguishes functional validity, structural proxy success, accepted macro routing and verified equivalent manual repairs. Missing artifacts stay in the planned denominator and are reported separately. The keyword-router baseline is included in the M7 evaluation.

`scenes` runs 12 Zelda descriptions at seed 42, with ten actions and a 4,096-token call limit. It preserves the original shared null-instruction control: **13 generation episodes per arm**, with each of the 12 descriptions compared to the shared null result. It runs `full`, `no_eval`, `no_history`, `no_metrics` and `always_accept`, then produces the cross-arm report. The paper's main scene comparison uses `full` versus `no_eval`.

Scene atoms measure visible tile relations. They do not establish a general semantic or narrative judgment. The original human-rating sheet generator remains available through `experiments.high_level_freeform_eval --help`.

## Native laptop campaign

The native path runs 32 Natural-32 requests, 64 full-harness Zelda targets and 36 M7 tool episodes through the LM Studio SDK. Follow [native deployment](deployment.md#native-lm-studio-on-apple-silicon) for model loading, exact GGUF identity and resume instructions.

The runner records immutable attempts, raw requests, responses, fragments and effective wire configuration. It generates `analysis/summary.json` and `analysis/m7_evaluation.json` after completion. An execution interruption remains distinct from a completed scientific failure; resume does not retry an already completed case just because its level missed the target.

## Reading a run

The launcher prints each command and saves its console output in `command_*.log`. Use `--dry-run` to inspect the commands without executing them. Suite producers add case records, aggregate counts and model/protocol identity. Core agent episodes save `trace.jsonl`, `summary.json` and `final_level.txt`. Inference cost, proposed tools, executed tools, accepted edits and final success are different quantities; use the evaluator's named fields rather than counting log lines.

An interrupted run is incomplete. Do not compute a headline success rate over only its saved successes. Keep the original target denominator, report completeness, and retain failed attempts. A process exit failure makes the portable launcher return nonzero. Tool and scene campaigns continue to their aggregate evaluator so missing cells are visible.

The release validates code and deterministic checks; it does not include historical run outputs. Actual model reruns, latency and hardware comparisons must be measured on your deployment.
