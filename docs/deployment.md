# Local deployment

[← README](../README.md) · [Experiment guide](reproduction.md)

The Python client and the model server are separate environments. `requirements/client-lock.txt` records the tested client dependencies, including transitive versions. The client runs the harness, simulators and evaluators. The server owns model weights and inference. Install the client as described in the README before choosing a backend.

## Linux GPU server with vLLM

The retained serving configuration uses vLLM 0.19.1, PyTorch 2.10.0, Transformers 5.5.4 and XGrammar 0.1.33. These are the recorded experiment versions; they are separate from the artifact's client dependency pins.

```bash
python3.12 -m venv .venv-server
source .venv-server/bin/activate
python -m pip install -r requirements/serve-qwen.txt
bash scripts/serve_vllm.sh qwen38
```

Run the client in another terminal with `.venv` activated. The first server launch may download the pinned model snapshot. For a snapshot already available locally, pass its directory as the second argument:

```bash
bash scripts/serve_vllm.sh qwen38 /path/to/pinned-snapshot
```

| Preset | Model | Frozen revision | Representation |
| --- | --- | --- | --- |
| `qwen38` | `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | BF16 |
| `qwen36` | `Qwen/Qwen3.6-35B-A3B-FP8` | `95a723d08a9490559dae23d0cff1d9466213d989` | FP8 |
| `gemma4` | `google/gemma-4-31B-it` | `842da3794eaa0b77d5f08bae87a17459d91ff475` | BF16 |

All presets bind to loopback, use one inference slot, tensor parallelism 1, a 32,768-token context, temperature supplied per request and server seed 0. Qwen uses the `qwen3` reasoning parser; Gemma uses `gemma4` and the served identifier `gemma4-31b-pcg`. The language-only switch, generation configuration and model revisions are explicit in the launch script. `PORT=8001 bash scripts/serve_vllm.sh qwen38` selects another port; use the same port in the client's `--base-url`.

The large BF16 experiments used server-class GPU memory. Choose hardware that can hold the frozen weights plus the configured context; lowering precision, changing parallelism or reducing context creates a different deployment. The client does not need a GPU for simulation or tool execution.

The client uses a placeholder API key (`EMPTY`) for a local server unless `OPENAI_API_KEY` is set. If your endpoint requires authentication, set it in your shell or a local `.env`. Do not put credentials in the configuration files.

## Native LM Studio on Apple Silicon

The paper's laptop experiment used an Apple M1 Max with 32 GiB unified memory and a Q4_K_M model. Its native SDK path preserves request and effective-configuration records that are not available through every OpenAI-compatible route.

```bash
source .venv/bin/activate
python -m pip install -e ".[lmstudio]"
```

Import the following exact GGUF into LM Studio, load it under identifier **`qwen38-mac-preflight`**, configure **32,768 context tokens** and **one prediction slot**, and start the local native server on **127.0.0.1:1234**. The `lms` command must be on `PATH`; `lms ps --json` is used to verify the selected loaded model. The runner checks the indexed model identifier, context and slot configuration before inference.

| Identity field | Value |
| --- | --- |
| Repository | `lmstudio-community/Qwen3.8-27B-GGUF` |
| Revision | `5a7da681f60570ab5b439a587e912d2e5eddb582` |
| File | `Qwen3.8-27B-Q4_K_M.gguf` |
| Bytes | `16810714336` |
| Python SDK | `lmstudio==1.5.0` |

```bash
pcg-reproduce lmstudio \
  --gguf /path/to/Qwen3.8-27B-Q4_K_M.gguf \
  --output runs/native
```

This hashes the GGUF, validates the loaded model and runs the 132 registered episodes sequentially. It does not download, load or unload a model automatically. The transport preserves per-case seeds, bounded structured output, thinking settings, neutral sampling and context-overflow behavior, and checks the effective wire configuration when returned.

To inspect inputs without loading a model, use a **separate** CPU-only preflight directory:

```bash
python -m experiments.lmstudio_campaign \
  --cpu-only --case-limit 1 --output runs/native-preflight
```

`--case-limit` is a debugging subset, not the full paper campaign. CPU-only runs have a distinct recorded execution mode and cannot be resumed as inference results.

For an interrupted native inference campaign, resume its original directory explicitly:

```bash
python -m experiments.lmstudio_campaign \
  --model-file /path/to/Qwen3.8-27B-Q4_K_M.gguf \
  --output runs/native --resume
```

Each interrupted process gets a new attempt directory. Existing attempts remain intact, and completed task failures are not rerun. A run lock prevents two campaigns from writing to the same directory concurrently.

### Analyze outputs and observe resources

Analysis runs automatically at the end of a native campaign. It can also be regenerated without inference:

```bash
python -m experiments.lmstudio_analysis \
  --run runs/native --output runs/native/analysis
```

To compare newly generated server and native results by exact case ID, add `--server-m6 runs/natural32/model.json`, `--server-z64 runs/z64-qwen38/full.json` and/or `--server-m7 runs/tools/evaluation.json`. These are descriptive comparisons across hardware, precision and inference backends; the native campaign has no open-loop arm.

Two optional read-only observers can run in separate terminals while the campaign holds its lock:

```bash
python scripts/mac_lmstudio_resource_sidecar.py --run runs/native
python scripts/mac_lmstudio_background_sidecar.py --run runs/native
```

They record timestamped swap/memory-pressure observations and background process load. Sampling begins when the observer starts, and gaps remain explicit. Native analysis records total worker-attempt time and observed token/call costs. It does not relabel those measurements as model-load time, whole-session wall time or peak unified-memory usage.

## Optional framework backends

The base framework also retains inference adapters for Transformers, Anthropic, Gemini and Portkey. Install `.[transformers]` or `.[cloud]` only when using those adapters; the paper reproduction presets use vLLM or native LM Studio. Optional plotting uses `.[plots]`. These extras are not required by the offline smoke example.
