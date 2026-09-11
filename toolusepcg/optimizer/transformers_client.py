"""
OpenAI-compatible HuggingFace-transformers client for the ToolUsePCG harness.

Lets ``GreedyOptimizer`` run local SFT / GRPO checkpoints without standing up
a vLLM server. Exposes just enough of the OpenAI SDK surface that
``greedy.py::_call_llm`` keeps working:

  client.chat.completions.create(
      model=..., messages=[...], max_tokens=..., temperature=..., timeout=...,
  ) -> response
  response.choices[0].message.content -> str
  response.choices[0].finish_reason    -> "stop" | "length"
  response.usage.prompt_tokens         -> int
  response.usage.completion_tokens     -> int

Supported checkpoint layouts (auto-detected):

  * **Full fine-tune** — the model dir contains ``model.safetensors`` (or
    sharded ``model-*.safetensors``). Load directly.
  * **PEFT/LoRA adapter** — the model dir contains ``adapter_model.safetensors``
    (optionally under an ``adapter/`` subdir). The base is read from
    ``adapter_config.json::base_model_name_or_path`` (or overridden via
    ``base_model`` arg).

An optional ``chat_template_path`` overrides the tokenizer's chat template —
use this to drop Qwen3.5's empty ``<think>`` block injection.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenAI-shaped response objects
# ---------------------------------------------------------------------------


@dataclass
class _Message:
    role: str
    content: str


@dataclass
class _Choice:
    index: int
    message: _Message
    finish_reason: str  # "stop" | "length"


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class _ChatCompletion:
    id: str
    choices: list[_Choice]
    usage: _Usage
    model: str


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class _ChatCompletionsNamespace:
    """Thin namespace so ``client.chat.completions.create(...)`` works."""

    def __init__(self, outer: "TransformersClient"):
        self._outer = outer

    def create(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        timeout: float | None = None,  # accepted for API parity; unused
        **_ignored: Any,
    ) -> _ChatCompletion:
        max_new = max_completion_tokens or max_tokens or 2048
        temp = 0.0 if temperature is None else float(temperature)
        return self._outer._generate(
            messages=messages,
            max_new_tokens=max_new,
            temperature=temp,
            top_p=top_p,
        )


class _ChatNamespace:
    def __init__(self, outer: "TransformersClient"):
        self.completions = _ChatCompletionsNamespace(outer)


class TransformersClient:
    """Local HuggingFace model with an OpenAI-compatible surface."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        base_model: str | Path | None = None,
        chat_template_path: str | Path | None = None,
        load_in_4bit: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str | dict = "auto",
        seed: int | None = None,
        grammar_domain: str | None = None,
        grammar_max_tool_calls: int = 20,
    ):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"model_path does not exist: {model_path}")

        is_full = (model_path / "model.safetensors").exists() or list(
            model_path.glob("model-*.safetensors")
        )
        is_adapter = (model_path / "adapter_model.safetensors").exists() or (
            model_path / "adapter" / "adapter_model.safetensors"
        ).exists()

        load_kwargs: dict = {"trust_remote_code": True, "dtype": dtype}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        load_kwargs["device_map"] = device_map

        if is_full and not is_adapter:
            log.info(f"[TransformersClient] loading full-FT model from {model_path}")
            self._tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
        elif is_adapter:
            # Resolve the base model: explicit arg wins; otherwise read
            # adapter_config.json which PEFT writes on save_pretrained.
            import json as _json
            if (model_path / "adapter" / "adapter_model.safetensors").exists():
                adapter_dir = model_path / "adapter"
            else:
                adapter_dir = model_path
            if base_model is None:
                cfg_path = adapter_dir / "adapter_config.json"
                if cfg_path.exists():
                    base_model = _json.loads(cfg_path.read_text()).get("base_model_name_or_path")
                if base_model is None:
                    raise ValueError(
                        f"Adapter at {adapter_dir} has no base_model_name_or_path; "
                        "pass base_model=... explicitly."
                    )
            base_model = Path(base_model) if Path(str(base_model)).exists() else str(base_model)
            log.info(
                f"[TransformersClient] loading base {base_model!r} + adapter at {adapter_dir}"
            )
            self._tokenizer = AutoTokenizer.from_pretrained(str(base_model), trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(str(base_model), **load_kwargs)
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, str(adapter_dir))
        else:
            raise ValueError(
                f"{model_path} is neither a full-FT ckpt (model.safetensors) "
                f"nor a PEFT adapter (adapter_model.safetensors)."
            )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        if chat_template_path is not None:
            tmpl = Path(chat_template_path).read_text()
            self._tokenizer.chat_template = tmpl
            log.info(f"[TransformersClient] applied chat template from {chat_template_path}")

        self._model.eval()
        self._model_name = str(model_path)
        # Seed ONCE at construction (not per-call). Re-seeding every generate
        # call collapses sampling to deterministic output (every call gets the
        # same RNG → same tokens regardless of temperature).
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        # Optional: pre-compile xgrammar grammar for the given domain so the
        # model is FORCED to emit a valid STEP JSON during generation. Catches
        # the "parameters as a string of dots" failure mode at token level.
        self._compiled_grammar = None
        if grammar_domain is not None:
            try:
                self._compiled_grammar = self._build_compiled_grammar(
                    grammar_domain, grammar_max_tool_calls
                )
                log.info(
                    f"[TransformersClient] xgrammar constrained decoding ENABLED "
                    f"for domain={grammar_domain!r}"
                )
            except Exception as e:
                log.warning(
                    f"[TransformersClient] failed to build grammar for {grammar_domain!r}: "
                    f"{e}; falling back to free generation"
                )
                self._compiled_grammar = None

        # OpenAI SDK compatibility
        self.chat = _ChatNamespace(self)

    def _build_compiled_grammar(self, domain: str, max_tool_calls: int):
        """Build and compile a STEP-message grammar for ``domain``."""
        import json as _json
        import sys
        # Make project root importable so we can reach toolusepcg.protocol.grammar.
        # In a typical install we run from the project cwd; this is a safety net.
        _project_root = Path(__file__).resolve().parents[2]
        if str(_project_root) not in sys.path:
            sys.path.insert(0, str(_project_root))
        from toolusepcg.protocol.grammar import build_step_grammar_pack
        from toolusepcg.protocol.prompt_builder import HarnessPromptBuilder
        import xgrammar as xgr

        builder = HarnessPromptBuilder.for_domain(domain)
        pack = build_step_grammar_pack(builder._opt.registry, max_tool_calls=max_tool_calls)
        # Compile via xgrammar from the JSON schema. The schema is bit-exact
        # with the one used by SFT prompt rendering, so the model already
        # speaks this dialect.
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(self._tokenizer)
        compiler = xgr.GrammarCompiler(tokenizer_info)
        return compiler.compile_json_schema(_json.dumps(pack.json_schema))

    # ------------------------------------------------------------------
    # Core generate
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate(
        self,
        *,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float | None,
    ) -> _ChatCompletion:
        try:
            return self._do_generate(
                messages=messages, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
            )
        except Exception:
            import traceback, sys
            sys.stderr.write("[TransformersClient] _generate failed:\n" + traceback.format_exc())
            sys.stderr.flush()
            raise

    def _do_generate(
        self,
        *,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float | None,
    ) -> _ChatCompletion:
        enc = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = enc["input_ids"].to(self._model.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._model.device)

        gen_kwargs: dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        if temperature <= 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            if top_p is not None:
                gen_kwargs["top_p"] = top_p

        # If a compiled grammar is attached, force token-level schema constraint.
        # Each generate() call needs a fresh LogitsProcessor (it carries
        # per-generation matcher state).
        if self._compiled_grammar is not None:
            import xgrammar as xgr
            gen_kwargs["logits_processor"] = [
                xgr.hf.LogitsProcessor(self._compiled_grammar)
            ]

        out_ids = self._model.generate(**gen_kwargs)

        prompt_len = input_ids.shape[1]
        new_tokens = out_ids[0, prompt_len:]
        completion_len = int(new_tokens.shape[0])
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        # "length" when we hit max_new_tokens and the last token isn't EOS
        hit_eos = bool(
            (self._tokenizer.eos_token_id is not None)
            and (new_tokens.numel() > 0)
            and (int(new_tokens[-1].item()) == self._tokenizer.eos_token_id)
        )
        finish_reason = "stop" if hit_eos else "length"

        usage = _Usage(
            prompt_tokens=int(prompt_len),
            completion_tokens=completion_len,
            total_tokens=int(prompt_len + completion_len),
        )
        choice = _Choice(
            index=0,
            message=_Message(role="assistant", content=text),
            finish_reason=finish_reason,
        )
        return _ChatCompletion(
            id="local-transformers",
            choices=[choice],
            usage=usage,
            model=self._model_name,
        )
