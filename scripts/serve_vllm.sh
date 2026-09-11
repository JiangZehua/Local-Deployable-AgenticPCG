#!/usr/bin/env bash
# Run in a separate serving environment. Does not alter the client environment.
set -euo pipefail
MODEL_PRESET=${1:-qwen38}
PORT=${PORT:-8000}
case "$MODEL_PRESET" in
  qwen38)
    MODEL=Qwen/Qwen3.8-27B
    REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
    SERVED_MODEL=$MODEL
    PARSER=qwen3
    DTYPE=(--dtype bfloat16)
    TEMPLATE='{"enable_thinking": false}'
    ;;
  qwen36)
    MODEL=Qwen/Qwen3.6-35B-A3B-FP8
    REVISION=95a723d08a9490559dae23d0cff1d9466213d989
    SERVED_MODEL=$MODEL
    PARSER=qwen3
    DTYPE=()
    TEMPLATE='{"enable_thinking": false}'
    ;;
  gemma4)
    MODEL=google/gemma-4-31B-it
    REVISION=842da3794eaa0b77d5f08bae87a17459d91ff475
    SERVED_MODEL=gemma4-31b-pcg
    PARSER=gemma4
    DTYPE=(--dtype bfloat16)
    TEMPLATE='{"enable_thinking": true}'
    ;;
  *) echo "Usage: bash scripts/serve_vllm.sh {qwen38|qwen36|gemma4} [local-snapshot-directory]" >&2; exit 2 ;;
esac
MODEL_SOURCE=${2:-$MODEL}
export TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1
exec vllm serve "$MODEL_SOURCE" --revision "$REVISION" \
  --served-model-name "$SERVED_MODEL" --host 127.0.0.1 --port "$PORT" \
  "${DTYPE[@]}" --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
  --max-model-len 32768 --max-num-seqs 1 --max-num-batched-tokens 32768 \
  --seed 0 --generation-config vllm --reasoning-parser "$PARSER" \
  --language-model-only --default-chat-template-kwargs "$TEMPLATE"
