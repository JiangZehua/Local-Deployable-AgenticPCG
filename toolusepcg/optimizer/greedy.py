"""Greedy hill-climb optimizer with epsilon exploration."""

from __future__ import annotations

import copy
import hashlib
import math
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

from ..config import Config, get_problem_type
from ..level import Level
from ..factory import LevelFactory
from ..evaluator import StatsEvaluator, EvalResult
from ..smb_contract import (
    COUNTED_KILL_EVENTS,
    COUNTED_KILL_METHODS,
    SMB_ENGINE_PADDING_TILES,
    SMB_SPRITE_NAMES,
    STOMP_KILL_EVENT,
)
from ..edit_manager import EditManager
from ..tools import (
    ToolRegistry,
    PlaceWallSegmentTool,
    PlaceTileTool,
    PlaceSingleTileTool,
    PlaceLineTool,
    PlacePatchTool,
    BuildSmbStompCorridorTool,
    BuildSmbJumpEpisodeTool,
    BuildSmbBehaviorEpisodeTool,
    CalculateStatsTool,
    GenerateRandomTool,
    GenerateMazeTool,
    GenerateBSPTool,
    GenerateCATool,
    GenerateConnectTool,
    GenerateDiggerTool,
    GenerateWFCTool,
    GenerateZeldaTilePlacingTool,
    GenerateZeldaRandomTool,
    GenerateZeldaMazeTool,
    GenerateZeldaBSPTool,
    GenerateZeldaCATool,
    GenerateZeldaConnectTool,
    GenerateZeldaDiggerTool,
    GenerateZeldaWFCTool,
)
from ..protocol import (
    MessageType,
    StepMessage,
    ProposeSkillMessage,
    StopMessage,
    parse_message,
    validate_message,
    interaction_protocol_v2_enabled,
)
from ..protocol.repair import parse_with_repair
from .scoring import Scorer
from .termination import TerminationChecker


# Load environment variables
load_dotenv()

OPENAI_SEED_ENV = "RPE_PCG_OPENAI_SEED"
DUPLICATE_LEVEL_GUARD_ENV = "RPE_PCG_DUPLICATE_LEVEL_GUARD"


def _level_digest(level: Level) -> str:
    """Return the canonical serialized-level identity used by the guard."""
    return hashlib.sha256(level.to_string().encode("utf-8")).hexdigest()


def _zelda_goal_contract(config: Config) -> tuple[str | None, str | None, str]:
    """Return mutually consistent path, action, and enemy instructions.

    Lines are unnumbered; the prompt numbers them. A maximize-type path goal is
    omitted (``None``) when its scoring weight ``w_path`` is 0, because the
    scorer would ignore it and the prompt must not ask for it.
    """
    pk_cfg = config.targets.player_key
    kd_cfg = config.targets.key_door
    sol_cfg = config.targets.solution_length
    path_weighted = float(config.scoring.w_path) != 0.0

    component_goals = []
    exact_components = False
    for label, target_cfg in (
        ("player-to-key", pk_cfg),
        ("key-to-door", kd_cfg),
    ):
        if target_cfg and target_cfg.mode == "target":
            component_goals.append(
                f"a {label} path length of exactly {target_cfg.value}"
            )
            exact_components = True
        elif target_cfg and target_cfg.mode == "maximize" and path_weighted:
            component_goals.append(f"the longest feasible {label} path")

    if component_goals:
        path_goal = "Achieve " + " and ".join(component_goals)
        if exact_components:
            path_instruction = (
                "Adjust paths only until the specified exact component "
                "lengths are reached; stop immediately once all hard "
                "constraints and exact targets hold"
            )
        else:
            path_instruction = "Increase the specified path lengths"
    elif sol_cfg and sol_cfg.mode == "target":
        path_goal = (
            "Achieve a solution length (player-to-key plus key-to-door) "
            f"of exactly {sol_cfg.value}"
        )
        path_instruction = (
            "Adjust the two paths only until their sum reaches the exact "
            "solution-length target; stop immediately once all hard "
            "constraints and the exact target hold"
        )
    elif path_weighted:
        path_goal = "Maximize the solution length (player-to-key plus key-to-door)"
        path_instruction = "Then optimize for longer solution paths"
    else:
        path_goal = path_instruction = None

    enemy_cfg = config.targets.enemies
    if enemy_cfg and enemy_cfg.mode == "target" and enemy_cfg.value is not None:
        enemy_goal = f"Have exactly {enemy_cfg.value} enemies (E)"
    else:
        enemy_goal = "Use enemies (E) according to the configured objective"
    return path_goal, path_instruction, enemy_goal


def _numbered(lines: list[str | None]) -> str:
    return "\n".join(f"{i}. {line}" for i, line in enumerate((l for l in lines if l), start=1))


def _openai_sampling_seed(config: Config) -> int | None:
    """Return the trial seed only for the explicit R041 sampling gate.

    Historical runs intentionally omitted the OpenAI ``seed`` request field,
    so the gate is strict and default-off.  R041 turns it on together with a
    nonzero temperature to obtain reproducible, seed-paired test-time samples.
    """

    if config.llm.provider != "openai" or os.environ.get(OPENAI_SEED_ENV) != "on":
        return None
    return int(config.seed)


@dataclass
class StepRecord:
    """Record of a single optimization step."""

    step_num: int
    message_type: str
    raw_response: str = ""
    # Native model reasoning is separate from the constrained final JSON on
    # Qwen/vLLM.  Keep it for diagnosis instead of silently discarding the part
    # of the response this project is trying to improve.
    reasoning_content: str = ""
    # R042-only capture, populated solely under the explicit environment gate.
    # Default runs retain the historical empty value and trace shape.
    request_messages: list[dict[str, str]] = field(default_factory=list)
    parsed_message: dict = field(default_factory=dict)
    tool_results: list[dict] = field(default_factory=list)
    level_before: str = ""
    level_after: str = ""
    metrics_before: dict = field(default_factory=dict)
    metrics_after: dict = field(default_factory=dict)
    # Compact, domain-specific attribution for the evaluated candidate.  This
    # is deliberately a string rather than raw evaluator info (which can be
    # large and contain non-JSON objects).
    environment_feedback_after: str = ""
    score_before: float = 0.0
    score_after: float = 0.0
    accepted: bool = False
    accept_reason: str = ""
    acceptance_triage: dict = field(default_factory=dict)
    # `transaction_tiles_changed` counts successful intermediate writes from
    # tool calls. `net_tiles_changed` counts cells whose final value differs
    # from the level at the start of the step. They differ when a transaction
    # temporarily rewrites and restores cells.
    transaction_tiles_changed: int = 0
    net_tiles_changed: int = 0
    num_tiles_changed: int = 0
    duplicate_level: bool = False
    duplicate_first_evaluated_step: int | None = None
    evaluator_skipped: bool = False
    errors: list[str] = field(default_factory=list)
    truncated: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_time: float = 0.0


@dataclass
class ConversationEntry:
    """A single turn in the conversation history."""
    prompt: str
    raw_response: str
    accepted: bool
    accept_reason: str
    message_type: str  # "STEP", "PARSE_ERROR", "STOP", "PROPOSE_SKILL"
    step_record: StepRecord | None = None


# R024 (execution-capability lever): a FIXED, env-gated worked example showing a
# valid construction transcript that builds a solvable Sokoban from an empty room.
# Activated by RPE_PCG_DESIGNER_FEWSHOT=sokoban (or a comma-list containing
# "sokoban"). It is NOT subject to reflector mutation and uses ONLY legal Sokoban
# tool names/params: tool `place_tile` (toolusepcg/tools/edit_place_tile.py),
# tile_type values player/crate/target (toolusepcg/level.py SOKOBAN_NAME_TILES).
_SOKOBAN_FEWSHOT_BLOCK = (
    "## Example Construction (worked):\n"
    "Construction order to imitate when starting from an empty room: place the "
    "COUNTS FIRST (exactly 1 player, k crates, k targets, with #crates == "
    "#targets), then a SOLVABLE ARRANGEMENT (each crate gets a clear straight "
    "push-path to a distinct target), and add wall/maze COMPLEXITY LAST. Worked "
    "example on a 7x7 room with k=2 (y is the row, x is the column, 0-indexed):\n"
    "\n"
    "Empty room (before):\n"
    "```\n"
    "#######\n"
    "#     #\n"
    "#     #\n"
    "#     #\n"
    "#     #\n"
    "#     #\n"
    "#######\n"
    "```\n"
    "STEP 1 -- place the counts (1 player, 2 crates, 2 targets) in a solvable layout:\n"
    "```json\n"
    "{\n"
    '  "type": "STEP",\n'
    '  "rationale": "Establish exactly 1 player and a balanced 2 crates / 2 targets, each crate west of its own target.",\n'
    '  "plan": "Player at (1,1); crates at (2,2) and (4,2); targets at (2,4) and (4,4) so each crate is pushed straight east onto a distinct target.",\n'
    '  "tool_calls": [\n'
    '    {"tool_name": "place_tile", "parameters": {"mode": "single", "tile_type": "player", "y": 1, "x": 1}},\n'
    '    {"tool_name": "place_tile", "parameters": {"mode": "single", "tile_type": "crate", "y": 2, "x": 2}},\n'
    '    {"tool_name": "place_tile", "parameters": {"mode": "single", "tile_type": "target", "y": 2, "x": 4}},\n'
    '    {"tool_name": "place_tile", "parameters": {"mode": "single", "tile_type": "crate", "y": 4, "x": 2}},\n'
    '    {"tool_name": "place_tile", "parameters": {"mode": "single", "tile_type": "target", "y": 4, "x": 4}}\n'
    "  ],\n"
    '  "acceptance_hint": "Accept: player count is 1, crates equal targets, and both crates have a clear eastward push-path."\n'
    "}\n"
    "```\n"
    "Resulting solvable layout (after):\n"
    "```\n"
    "#######\n"
    "#@    #\n"
    "# $ . #\n"
    "#     #\n"
    "# $ . #\n"
    "#######\n"
    "```\n"
    "Solution: stand west of each crate and push east twice -- crate (2,2)->(2,4) "
    "and crate (4,2)->(4,4). Counts first, solvable arrangement second, complexity last.\n"
    "\n"
)


def _sokoban_fewshot_enabled() -> bool:
    """True when RPE_PCG_DESIGNER_FEWSHOT selects the sokoban worked example.

    Accepts an exact "sokoban" value or a comma-separated list containing it.
    """
    raw = os.environ.get("RPE_PCG_DESIGNER_FEWSHOT", "")
    return "sokoban" in {part.strip() for part in raw.split(",") if part.strip()}


def _coord_bounds_block(level: Level) -> str:
    """R028 (execution-capability lever): env-gated explicit coordinate bounds.

    When ``RPE_PCG_COORD_BOUNDS == "explicit"`` returns a single line (terminated
    by a newline) stating the TRUE grid dimensions and the valid inclusive
    coordinate range, to be spliced into the designer prompt immediately after the
    rendered ASCII grid block. The 35B designer otherwise emits ``place_tile``
    calls at out-of-bounds coordinates because the prompt never states the grid
    size, so ~86-91% of placements fail with "Position is out of bounds".

    ``y`` is the row index (height) and ``x`` is the column index (width); these
    come straight from ``level.height`` / ``level.width`` (``data.shape[0/1]``) so
    they are never transposed. Returns "" when the gate is unset or set to any
    value other than "explicit", keeping the prompt byte-identical to baseline.
    """
    if os.environ.get("RPE_PCG_COORD_BOUNDS", "") != "explicit":
        return ""
    h, w = level.height, level.width
    return (
        f"Grid size: {h} rows × {w} cols. Valid coordinates: "
        f"y in [0, {h - 1}], x in [0, {w - 1}] (0-indexed, inclusive). "
        f"Any tile placement outside this range is REJECTED as out of bounds — "
        f"read the grid size above and keep every y,x within these limits.\n"
    )


def _coord_ruler_enabled() -> bool:
    """True when RPE_PCG_COORD_RULER selects the ruler-annotated grid rendering."""
    return os.environ.get("RPE_PCG_COORD_RULER", "") == "on"


def _coord_ruler_render(level_str: str) -> str:
    """R031 (representation lever — perception-vs-capability disambiguator): wrap a
    rendered ASCII level in a coordinate ruler so the y/x indices are PERCEPTUALLY
    visible rather than something the model must infer by counting blank rows.

    Baseline rendering of an empty 8×8 Sokoban grid is 8 indistinguishable blank
    rows, so the designer cannot SEE that valid coordinates are 0–7 and defaults to
    large "big-canvas" coordinates (y=20/50/99) → ~89% out-of-bounds placements that
    neither explicit bounds text (R028) nor fed-back OOB errors (R029/R030) fix.
    R031 tests whether making the coordinate frame visible is enough: if OOB drops
    sharply, the dead-sokoban trap was a PERCEPTION failure (unreadable empty grid);
    if OOB stays ~89% even with every cell's index on screen, it is a genuine 35B
    spatial-grounding CAPABILITY ceiling.

    Returns a block with a column-index header (ones row, plus a tens row when the
    width exceeds 10) and each level row prefixed by its right-aligned y index and a
    "| " separator. Cell columns stay contiguous (one char per cell, matching
    ``Level.to_string``) and stay aligned under the header. This function is only
    invoked when the gate is on; callers pass ``level.to_string()`` straight through
    when it is off, keeping the prompt byte-identical to baseline.
    """
    rows = level_str.split("\n")
    height = len(rows)
    width = max((len(r) for r in rows), default=0)
    yw = max(len(str(height - 1)), 1) if height else 1
    prefix = " " * (yw + 2)  # aligns header digits over the first cell column
    lines: list[str] = []
    if width > 10:
        tens = prefix + "".join(
            str((c // 10) % 10) if c % 10 == 0 else " " for c in range(width)
        )
        lines.append(tens.rstrip())
    ones = prefix + "".join(str(c % 10) for c in range(width))
    lines.append(ones)
    for y, row in enumerate(rows):
        lines.append(f"{str(y).rjust(yw)}| {row}")
    return "\n".join(lines)


def _render_level(level: "Level") -> str:
    """Render a level for the designer prompt: ruler-annotated when the R031 gate is
    on, otherwise the raw ``to_string()`` (byte-identical to baseline)."""
    level_str = level.to_string()
    if _coord_ruler_enabled():
        return _coord_ruler_render(level_str)
    return level_str


def _error_feedback_block(history: list) -> str:
    """R029 (failure-feedback loop): env-gated echo of the designer's OWN recent
    failed tool calls back into its prompt.

    The level-editing designer is stateless by default (conversation_length=1) so
    it never sees its own prior-step tool errors and can repeat the SAME failing
    call many times in a row. When ``RPE_PCG_ERROR_FEEDBACK == "on"`` this returns
    a block (terminated by a newline) titled "## Errors From Your Recent
    Attempts:" listing the FAILED tool calls from the last up to 3 steps, most
    recent first, echoing ONLY the tool's own verbatim error strings -- no
    coordinate, bounds, or game-specific text is injected by this function.
    Identical ``(tool_name, parameters, error)`` lines are collapsed with a
    "(repeated N×)" note so the model can SEE it has been repeating itself, and
    the list is capped at a small number of lines. Returns "" when the gate is
    off (any value other than "on"), when there is no history yet (step 1), or
    when no failures are present (no empty header) — keeping the designer prompt
    byte-identical to baseline in all of those cases.

    ``history`` is the optimizer's ``_conversation_history`` list; each entry has
    a ``step_record`` whose ``tool_results`` are dicts shaped like
    ``{"tool_name", "parameters", "result": {"success", "data", "errors"}}``.
    """
    if os.environ.get("RPE_PCG_ERROR_FEEDBACK", "") != "on":
        return ""
    if not history:
        return ""

    max_lines = 12
    # Aggregate failing calls across the last up to 3 steps, most recent first.
    counts: dict[tuple, int] = {}
    order: list[tuple] = []
    for entry in reversed(history[-3:]):
        rec = getattr(entry, "step_record", None)
        if rec is None:
            continue
        for tr in getattr(rec, "tool_results", None) or []:
            tool_name = tr.get("tool_name")
            # Skip the synthetic truncation marker (not a real failed call).
            if tool_name == "_truncated":
                continue
            result = tr.get("result") or {}
            errors = result.get("errors") or []
            failed = result.get("success") is False or bool(errors)
            if not failed:
                continue
            params = tr.get("parameters")
            try:
                params_str = json.dumps(params, sort_keys=True)
            except (TypeError, ValueError):
                params_str = str(params)
            # One line per distinct verbatim error string from the tool.
            for err in (errors if errors else [""]):
                key = (tool_name, params_str, err)
                if key not in counts:
                    counts[key] = 0
                    order.append(key)
                counts[key] += 1

    if not order:
        return ""

    lines = [
        "## Errors From Your Recent Attempts:",
        "These tool calls from your recent steps FAILED. Read them and do NOT "
        "repeat the same failing calls — change your approach.",
    ]
    for key in order[:max_lines]:
        tool_name, params_str, err = key
        n = counts[key]
        repeat = f" (repeated {n}×)" if n > 1 else ""
        lines.append(f"- {tool_name}({params_str}) -> FAILED: {err}{repeat}")
    return "\n".join(lines) + "\n"


class GreedyOptimizer:
    """Greedy hill-climb optimizer with LLM agent.

    Uses EditManager for all level modifications, ensuring edits go through
    a central validated interface connected to the PCG benchmark.
    """

    def __init__(self, config: Config, logger: Any = None, skill_manager: Any = None, verbose: bool = True, debug: bool = False):
        """Initialize optimizer.

        Args:
            config: Full configuration.
            logger: Optional TraceLogger instance.
            skill_manager: Optional SkillGrowthManager instance.
            verbose: Whether to print step progress to terminal.
            debug: Whether to save full LLM prompts/responses to debug.log.
        """
        self.config = config
        self.logger = logger
        self.skill_manager = skill_manager
        self.verbose = verbose
        self.debug = debug

        # Initialize components
        self.factory = LevelFactory(config)
        self.evaluator = StatsEvaluator(config, self.factory.env)
        self.scorer = Scorer(config)
        self.termination = TerminationChecker(config)

        # EditManager - central controller for all level modifications
        self.edit_manager: EditManager | None = None

        # Current state (derived from EditManager)
        self.current_result: EvalResult | None = None
        self.current_score: float = float("-inf")

        # Problem type
        self.problem_type = get_problem_type(config.env.name)

        # Tool registry (will be initialized with EditManager)
        self.registry: ToolRegistry | None = None
        self._wall_tool: PlaceWallSegmentTool | None = None
        self._tile_tool: PlaceTileTool | None = None
        self._single_tile_tool: PlaceSingleTileTool | None = None
        self._line_tool: PlaceLineTool | None = None
        self._patch_tool: PlacePatchTool | None = None
        self._stats_tool: CalculateStatsTool | None = None

        # LLM client (Portkey or OpenAI, depending on model prefix)
        self._client: Any = None
        self._triage_client: Any = None
        self._openai_structured_outputs: dict[str, Any] | None = None
        self._openai_step_schema: dict[str, Any] | None = None
        self._last_acceptance_triage: dict[str, Any] | None = None
        self._last_reasoning_content = ""
        self.llm_retry_count = 0

        # R100-B is an explicit, default-off search-quality gate.  Repeated
        # candidates consume their normal semantic STEP but never another
        # evaluator call.  The existing max-steps budget therefore bounds
        # retries without introducing a second hidden call budget.
        self.duplicate_level_guard = (
            os.environ.get(DUPLICATE_LEVEL_GUARD_ENV) == "on"
        )
        self._evaluated_level_cache: dict[str, tuple[int, EvalResult]] = {}

        # Step tracking
        self.step_count = 0
        self.step_records: list[StepRecord] = []

        # Epsilon for accepting worse solutions (greedy mode)
        self.epsilon = config.optimizer.epsilon_accept_worse
        # Inner-loop block size (R025). 1 = per-step accept/reject (default,
        # unchanged behavior); k>1 = roll out k tentative steps per block then
        # make a single block-level accept/reject decision.
        self.rollout_steps = max(1, int(getattr(config.optimizer, "rollout_steps", 1) or 1))
        self.stable_change_accounting = bool(
            getattr(config.optimizer, "stable_change_accounting", False)
        )
        # Transient stash used only by the k>1 rollout path to carry the last
        # STEP message / tool results into the block decision (never read on the
        # k==1 path).
        self._block_last_message = None
        self._block_last_tool_results = None
        self.ablation_mode = config.optimizer.ablation_mode
        self.acceptance_policy = config.optimizer.acceptance_policy
        if self.acceptance_policy not in {"default", "always_accept"}:
            raise ValueError(
                f"Unknown optimizer.acceptance_policy={self.acceptance_policy!r}. "
                "Expected 'default' or 'always_accept'."
            )
        valid_ablation_modes = {
            "full",
            "no_historical_feedback",
            "no_metric_observation",
            "no_evaluation_loop",
        }
        if self.ablation_mode not in valid_ablation_modes:
            raise ValueError(
                f"Unknown optimizer.ablation_mode={self.ablation_mode!r}. "
                f"Expected one of {sorted(valid_ablation_modes)}."
            )
        commit_policy = getattr(config.optimizer, "commit_policy", "mode_default")
        if commit_policy not in {"mode_default", "force_apply", "verified"}:
            raise ValueError(
                f"Unknown optimizer.commit_policy={commit_policy!r}. "
                "Expected mode_default, force_apply, or verified."
            )
        self.commit_policy = (
            "force_apply"
            if commit_policy == "mode_default"
            and self.ablation_mode == "no_evaluation_loop"
            else "verified" if commit_policy == "mode_default" else commit_policy
        )
        exact_stop = getattr(
            config.optimizer, "evaluator_exact_stop", "mode_default"
        )
        if exact_stop not in {"mode_default", "on", "off"}:
            raise ValueError(
                f"Unknown optimizer.evaluator_exact_stop={exact_stop!r}. "
                "Expected mode_default, on, or off."
            )
        self.evaluator_exact_stop = (
            self.ablation_mode != "no_evaluation_loop"
            if exact_stop == "mode_default"
            else exact_stop == "on"
        )

        # Simulated annealing parameters
        self.sa_initial_temp = config.optimizer.annealing_initial_temp
        self.sa_decay_rate = config.optimizer.annealing_decay_rate
        self.triage_model = config.optimizer.triage_model or config.llm.model
        self.triage_base_url = config.optimizer.triage_base_url or config.llm.base_url
        self.triage_max_tokens = config.optimizer.triage_max_tokens
        self.triage_min_confidence = config.optimizer.triage_min_confidence

        # Conversation history
        self.conversation_length = config.optimizer.conversation_length
        self.conversation_filter = config.optimizer.conversation_filter
        self._conversation_history: list[ConversationEntry] = []

        # Debug log file handle (lazy init)
        self._debug_file = None

    @property
    def current_level(self) -> Level | None:
        """Get current level from EditManager."""
        if self.edit_manager is None:
            return None
        return self.edit_manager.level

    def _init_client(self) -> None:
        """Initialize LLM client based on config.llm.provider.

        Supported providers:
        - "openai": Uses the OpenAI SDK (default; works with vLLM, TGI, etc.).
        - "portkey": Uses the Portkey SDK when explicitly selected.
        - "deepseek": Uses the OpenAI SDK against api.deepseek.com.
        - "google": Uses Google Genai SDK.
        - "anthropic": Uses the Anthropic Messages API.
        - "transformers": Local HF model (full-FT or PEFT adapter). The
          ``llm.model`` config string is interpreted as a filesystem path;
          optional extras on ``llm``: ``base_model`` (required for adapters
          unless adapter_config.json has it), ``chat_template_path`` (to
          drop Qwen3.5's <think> block), ``load_in_4bit`` (default false).
        """
        if self._client is None:
            provider = self.config.llm.provider
            if provider == "portkey":
                from portkey_ai import Portkey
                self._client = Portkey(
                    base_url=self.config.llm.base_url,
                    strict_open_ai_compliance=False,
                )
            elif provider == "openai":
                from openai import OpenAI
                api_key = self.config.llm.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
                self._client = OpenAI(
                    base_url=self.config.llm.base_url,
                    api_key=api_key,
                )
            elif provider == "deepseek":
                from openai import OpenAI
                api_key = self.config.llm.api_key or os.environ.get("DEEPSEEK_API_KEY")
                if not api_key:
                    raise ValueError("DeepSeek provider requires api_key in config or DEEPSEEK_API_KEY env var.")
                base_url = self.config.llm.base_url or "https://api.deepseek.com"
                if "ai-gateway.apps.cloud.rt.nyu.edu" in (base_url or ""):
                    base_url = "https://api.deepseek.com"
                self._client = OpenAI(base_url=base_url, api_key=api_key)
            elif provider == "google":
                from google import genai
                api_key = self.config.llm.api_key or os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("Google provider requires api_key in config or GOOGLE_API_KEY env var.")
                self._client = genai.Client(api_key=api_key)
            elif provider == "anthropic":
                import anthropic
                api_key = self.config.llm.api_key or os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("Anthropic provider requires api_key in config or ANTHROPIC_API_KEY env var.")
                self._client = anthropic.Anthropic(api_key=api_key)
            elif provider == "transformers":
                from .transformers_client import TransformersClient
                llm_cfg = self.config.llm
                self._client = TransformersClient(
                    model_path=llm_cfg.model,
                    base_model=getattr(llm_cfg, "base_model", None),
                    chat_template_path=getattr(llm_cfg, "chat_template_path", None),
                    load_in_4bit=bool(getattr(llm_cfg, "load_in_4bit", False)),
                    seed=getattr(self.config, "seed", None),
                    grammar_domain=getattr(llm_cfg, "grammar_domain", None),
                    grammar_max_tool_calls=int(
                        getattr(llm_cfg, "grammar_max_tool_calls", None)
                        or self.config.optimizer.max_tool_calls_per_step
                    ),
                )
            else:
                raise ValueError(
                    f"Unknown LLM provider: {provider!r}. "
                    "Use 'portkey', 'openai', 'deepseek', 'google', 'anthropic', or 'transformers'."
                )

    def _init_triage_client(self) -> None:
        """Initialize an independent OpenAI-compatible client for acceptance triage."""

        if self._triage_client is not None:
            return
        from openai import OpenAI

        api_key = self.config.llm.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self._triage_client = OpenAI(base_url=self.triage_base_url, api_key=api_key)

    def _init_tools(self) -> None:
        """Initialize tool registry with EditManager.

        All edit tools use EditManager for modifications.
        Tools registered depend on config.optimizer.edit_tools (if set)
        or problem type (auto mode).
        """
        self.registry = ToolRegistry()

        # NOTE: calculate_stats tool disabled — LLM outputs all tool calls in
        # one shot so it cannot actually use mid-step stats feedback.
        # self._stats_tool = CalculateStatsTool(self.evaluator, self.edit_manager)
        # self.registry.register(self._stats_tool)

        edit_tools = self.config.optimizer.edit_tools

        if edit_tools is not None:
            # Explicit tool list from config
            valid_tools = {
                "place_tile", "place_wall_segment", "place_single_tile", "place_line", "place_patch",
                "build_smb_stomp_corridor",
                "build_smb_jump_episode",
                "build_smb_behavior_episode",
                "generate_random", "generate_maze", "generate_bsp", "generate_ca",
                "generate_connect", "generate_digger", "generate_wfc",
                "generate_zelda_tile_placing",
                "generate_zelda_random", "generate_zelda_maze", "generate_zelda_bsp",
                "generate_zelda_ca", "generate_zelda_connect", "generate_zelda_digger",
                "generate_zelda_wfc",
            }
            for tool_name in edit_tools:
                if tool_name not in valid_tools:
                    raise ValueError(
                        f"Unknown edit tool: {tool_name!r}. "
                        f"Valid tools: {sorted(valid_tools)}"
                    )
            if "place_tile" in edit_tools:
                self._tile_tool = PlaceTileTool(self.edit_manager)
                self.registry.register(self._tile_tool)
            if "place_wall_segment" in edit_tools:
                self._wall_tool = PlaceWallSegmentTool(self.edit_manager)
                self.registry.register(self._wall_tool)
            if "place_single_tile" in edit_tools:
                self._single_tile_tool = PlaceSingleTileTool(self.edit_manager)
                self.registry.register(self._single_tile_tool)
            if "place_line" in edit_tools:
                self._line_tool = PlaceLineTool(self.edit_manager)
                self.registry.register(self._line_tool)
            if "place_patch" in edit_tools:
                self._patch_tool = PlacePatchTool(self.edit_manager)
                self.registry.register(self._patch_tool)
            if "build_smb_stomp_corridor" in edit_tools:
                self.registry.register(BuildSmbStompCorridorTool(self.edit_manager))
            if "build_smb_jump_episode" in edit_tools:
                self.registry.register(BuildSmbJumpEpisodeTool(self.edit_manager))
            if "build_smb_behavior_episode" in edit_tools:
                self.registry.register(BuildSmbBehaviorEpisodeTool(self.edit_manager))
            if "generate_random" in edit_tools:
                self.registry.register(GenerateRandomTool(self.edit_manager))
            if "generate_maze" in edit_tools:
                self.registry.register(GenerateMazeTool(self.edit_manager))
            if "generate_bsp" in edit_tools:
                self.registry.register(GenerateBSPTool(self.edit_manager))
            if "generate_ca" in edit_tools:
                self.registry.register(GenerateCATool(self.edit_manager))
            if "generate_connect" in edit_tools:
                self.registry.register(GenerateConnectTool(self.edit_manager))
            if "generate_digger" in edit_tools:
                self.registry.register(GenerateDiggerTool(self.edit_manager))
            if "generate_wfc" in edit_tools:
                self.registry.register(GenerateWFCTool(self.edit_manager))
            if "generate_zelda_tile_placing" in edit_tools:
                self.registry.register(GenerateZeldaTilePlacingTool(self.edit_manager))
            if "generate_zelda_random" in edit_tools:
                self.registry.register(GenerateZeldaRandomTool(self.edit_manager))
            if "generate_zelda_maze" in edit_tools:
                self.registry.register(GenerateZeldaMazeTool(self.edit_manager))
            if "generate_zelda_bsp" in edit_tools:
                self.registry.register(GenerateZeldaBSPTool(self.edit_manager))
            if "generate_zelda_ca" in edit_tools:
                self.registry.register(GenerateZeldaCATool(self.edit_manager))
            if "generate_zelda_connect" in edit_tools:
                self.registry.register(GenerateZeldaConnectTool(self.edit_manager))
            if "generate_zelda_digger" in edit_tools:
                self.registry.register(GenerateZeldaDiggerTool(self.edit_manager))
            if "generate_zelda_wfc" in edit_tools:
                self.registry.register(GenerateZeldaWFCTool(self.edit_manager))
        else:
            # Auto mode: select based on problem type
            if self.problem_type in ('zelda', 'sokoban', 'loderunner', 'smb', 'smb_static'):
                self._tile_tool = PlaceTileTool(self.edit_manager)
                self.registry.register(self._tile_tool)
            else:
                self._wall_tool = PlaceWallSegmentTool(self.edit_manager)
                self.registry.register(self._wall_tool)

    def initialize(self, seed: int | None = None, initial_level: Level | None = None) -> None:
        """Initialize optimizer with starting level.

        Args:
            seed: Random seed for level generation.
            initial_level: Optional pre-specified initial level.
        """
        if initial_level is not None:
            level = initial_level.copy()
            # Ensure level has correct problem type
            level.problem_type = self.problem_type
        else:
            strategy = self.config.optimizer.init_strategy
            if strategy == "empty":
                level = self.factory.create_empty()
            elif strategy == "filled":
                level = self.factory.create_filled()
            elif strategy == "weighted_random":
                level = self.factory.create_weighted_random(seed)
            elif strategy == "random":
                level = self.factory.create_random(seed)
            else:
                raise ValueError(f"Unknown init_strategy: {strategy!r}")

        # Create EditManager as central controller
        self.edit_manager = EditManager(self.factory.env, level, problem_type=self.problem_type)

        # Initialize tools with EditManager
        self._init_tools()

        # Evaluate initial level
        self.current_result = self.evaluator.evaluate(self.current_level)
        self.current_score = self.scorer.score(self.current_result)

        self._evaluated_level_cache = {}
        if self.duplicate_level_guard:
            self._evaluated_level_cache[_level_digest(self.current_level)] = (
                0,
                copy.deepcopy(self.current_result),
            )

        # Reset tracking
        self.termination.reset()
        self.step_count = 0
        self.step_records = []
        self._conversation_history = []

        # Log initial state
        if self.logger:
            self.logger.log_initial_level(self.current_level, self.current_result)

    def _write_debug(self, text: str) -> None:
        """Write a line to the debug log file.

        Args:
            text: Text to write.
        """
        if not self.debug:
            return

        # Lazy init: open debug.log in run directory
        if self._debug_file is None and self.logger:
            debug_path = self.logger.get_run_dir() / "debug.log"
            self._debug_file = open(debug_path, "a")

        if self._debug_file:
            self._debug_file.write(text + "\n")
            self._debug_file.flush()

    def build_prompt(self) -> str:
        """Build prompt for LLM.

        Returns:
            Complete prompt string.
        """
        if self.problem_type == 'zelda':
            prompt = self._build_zelda_prompt()
        elif self.problem_type == 'sokoban':
            prompt = self._build_sokoban_prompt()
        elif self.problem_type == 'loderunner':
            prompt = self._build_loderunner_prompt()
        elif self.problem_type == 'smb_static':
            prompt = self._build_smb_static_prompt()
        elif self.problem_type == 'smb':
            prompt = self._build_smb_prompt()
        elif self.problem_type == 'binarydoor':
            prompt = self._build_binarydoor_prompt()
        else:
            prompt = self._build_binary_prompt()

        # Inform agent of tool call limit
        max_calls = self.config.optimizer.max_tool_calls_per_step
        prompt += f"\n\nNote: You may use up to {max_calls} tool calls per step. Any beyond this limit will be dropped."

        # R030 (step-feedback loop): when RPE_PCG_STEP_FEEDBACK == "on", surface the
        # previous step's ACCEPT/REJECT decision + reason AND the per-domain metric
        # deltas back to the stateless (conversation_length=1) designer. By default
        # this rich feedback is only emitted by _build_messages when
        # conversation_length > 1; this gate makes it work at conversation_length=1
        # too. It reuses _build_step_feedback as-is, so the per-domain metric
        # selection (self._FEEDBACK_METRICS keyed by self.problem_type) stays correct
        # and no cross-domain metric leaks. Emits nothing (→ byte-identical) when the
        # gate is off, on step 1 (empty history), or when the last entry has no
        # usable step_record.
        if os.environ.get("RPE_PCG_STEP_FEEDBACK", "") == "on" and self._conversation_history:
            last_entry = self._conversation_history[-1]
            if getattr(last_entry, "step_record", None) is not None:
                step_feedback = self._build_step_feedback(last_entry)
                if step_feedback:
                    prompt += (
                        "\n\n## Feedback From Your Previous Step:\n"
                        f"{step_feedback.rstrip()}"
                    )

        # R029 (failure-feedback loop): when RPE_PCG_ERROR_FEEDBACK == "on", echo
        # the designer's own recent FAILED tool calls back to it so a stateless
        # (conversation_length=1) designer can stop repeating the same failing
        # call. Returns "" (no append → byte-identical) when the gate is off, on
        # step 1, or when there are no failures.
        error_feedback = _error_feedback_block(self._conversation_history)
        if error_feedback:
            prompt += f"\n\n{error_feedback.rstrip()}"

        # Append extra instruction if configured
        extra = self.config.optimizer.extra_instruction
        if extra:
            prompt += f"\n\n## Extra Instruction:\n{extra}"

        return prompt

    def _build_binary_prompt(self) -> str:
        """Build prompt for binary maze optimization."""
        level_str = _render_level(self.current_level)
        metrics_str = self.evaluator.format_metrics_for_prompt(self.current_result)
        tools_str = self.registry.format_tools_for_prompt()
        term_status = self.termination.format_status()
        score_str = self.scorer.format_score(self.current_result)

        # Build dynamic goal based on path target mode
        path_target = self.config.targets.path
        if path_target.mode == "target" and path_target.value is not None:
            path_goal = f"achieve a longest shortest path of ~{path_target.value}"
        else:
            path_goal = "maximize the longest shortest path"

        # Build dynamic goal for regions
        region_target = self.config.targets.num_connected_regions
        if region_target.mode == "target":
            target_regions = region_target.value if region_target.value is not None else 1
            if target_regions == 1:
                region_goal = "while maintaining exactly 1 connected region"
            else:
                region_goal = f"while achieving ~{target_regions} connected regions"
        else:
            region_goal = "while maximizing the number of connected regions"

        prompt = f"""You are an AI agent optimizing a binary maze level. Your goal is to {path_goal} {region_goal}.

## Current Level (rows are y, columns are x, 0-indexed):
```
{level_str}
```
{_coord_bounds_block(self.current_level)}Legend: '.' = empty (passable), '#' = wall (blocked)

## {metrics_str}

## {score_str}

## Termination Status:
{term_status}

## {tools_str}

## Instructions:
1. Analyze the current level and its metrics
2. Plan edits to improve the score (move metrics toward goals described above)
3. Execute tool calls to modify the level

## Response Format:
Respond with a JSON object of one of these types:

### STEP - To execute tools:
```json
{{
  "type": "STEP",
  "rationale": "explanation of your reasoning",
  "plan": "high-level plan for improvement",
  "tool_calls": [
    {{"tool_name": "place_wall_segment", "parameters": {{...}}}}
  ],
  "acceptance_hint": "hint about whether to accept changes"
}}
```

### PROPOSE_SKILL - To propose a new tool:
```json
{{
  "type": "PROPOSE_SKILL",
  "rationale": "why this tool would help",
  "skill_spec": {{
    "name": "tool_name",
    "description": "what it does",
    "parameters": {{}},
    "implementation_hint": "how to implement"
  }}
}}
```

### STOP - To terminate optimization:
```json
{{
  "type": "STOP",
  "rationale": "why stopping now",
  "final_notes": "observations about the result"
}}
```

Respond with ONLY the JSON object, no additional text."""

        return prompt

    def _door_adjacent_str(self, door_pos: tuple[int, int]) -> str:
        """Return a human-readable string for the inner cell adjacent to a border door.

        Args:
            door_pos: (row, col) in the augmented/bordered grid.
        """
        r, c = door_pos
        h, w = self.current_level.height, self.current_level.width
        ah, aw = h + 2, w + 2

        if r == 0:
            # Top border: adjacent inner cell is directly below
            return f"row=0, col={c - 1}"
        elif r == ah - 1:
            # Bottom border: adjacent inner cell is directly above
            return f"row={h - 1}, col={c - 1}"
        elif c == 0:
            # Left border: adjacent inner cell is directly right
            return f"row={r - 1}, col=0"
        elif c == aw - 1:
            # Right border: adjacent inner cell is directly left
            return f"row={r - 1}, col={w - 1}"
        else:
            return f"row={r - 1}, col={c - 1}"

    def _build_bordered_level_with_doors(self) -> str:
        """Build a bordered ASCII view of the level with doors marked as 'D'.

        The inner grid is the current level. The border is walls (#) except
        at the two door positions which are marked 'D'.

        Returns:
            Multi-line string of the bordered level.
        """
        level = self.current_level
        h, w = level.height, level.width
        tile_chars = level.tile_chars

        # Get door positions from the env
        door1 = self.factory.env._problem._door1
        door2 = self.factory.env._problem._door2

        # Build augmented grid (H+2 x W+2)
        ah, aw = h + 2, w + 2
        lines = []
        for r in range(ah):
            row_chars = []
            for c in range(aw):
                if (r, c) == door1 or (r, c) == door2:
                    row_chars.append('D')
                elif r == 0 or r == ah - 1 or c == 0 or c == aw - 1:
                    row_chars.append('#')
                else:
                    # Inner cell: (r-1, c-1) in the original level
                    tile_val = int(level.data[r - 1, c - 1])
                    row_chars.append(tile_chars.get(tile_val, '?'))
            lines.append(''.join(row_chars))

        return '\n'.join(lines)

    def _build_binarydoor_prompt(self) -> str:
        """Build prompt for binary door maze optimization."""
        bordered_level_str = self._build_bordered_level_with_doors()
        inner_level_str = self.current_level.to_string()
        metrics_str = self.evaluator.format_metrics_for_prompt(self.current_result)
        tools_str = self.registry.format_tools_for_prompt()
        term_status = self.termination.format_status()
        score_str = self.scorer.format_score(self.current_result)

        # Get door positions
        door1 = self.factory.env._problem._door1
        door2 = self.factory.env._problem._door2

        # Build dynamic goal based on door_path target mode
        door_path_cfg = self.config.targets.door_path
        if door_path_cfg and door_path_cfg.mode == "target" and door_path_cfg.value is not None:
            path_goal = f"achieve a door-to-door path length of ~{door_path_cfg.value}"
        else:
            path_goal = "maximize the door-to-door path length"

        # Build dynamic goal for regions
        region_target = self.config.targets.num_connected_regions
        if region_target.mode == "target":
            target_regions = region_target.value if region_target.value is not None else 1
            if target_regions == 1:
                region_goal = "while maintaining exactly 1 connected region"
            else:
                region_goal = f"while achieving ~{target_regions} connected regions"
        else:
            region_goal = "while maximizing the number of connected regions"

        # Keep the BinaryDoor control arm aligned with the Zelda paper
        # ablation: the target contract, current grid, doors, legal tools, and
        # budget remain visible, while evaluator observations are withheld.
        # Shadow evaluation is still recorded for analysis.
        observation_section = ""
        if self.ablation_mode not in {"no_metric_observation", "no_evaluation_loop"}:
            observation_section = f"""
## {metrics_str}

## {score_str}
"""
        response_guard = (
            "The harness stops automatically at an exact, hard-valid "
            "state; do not emit a different message type."
            if getattr(
                self,
                "evaluator_exact_stop",
                self.ablation_mode != "no_evaluation_loop",
            )
            else "Use the available action budget to produce the best complete "
            "construction; do not emit a different message type."
        )

        prompt = f"""You are an AI agent optimizing a binary maze with two door openings. Your goal is to {path_goal} {region_goal}.

## How This Problem Works:
- The maze has a fixed border of walls with exactly TWO door openings (marked 'D' below).
- You edit only the inner grid (rows 0-{self.current_level.height-1}, columns 0-{self.current_level.width-1}).
- The door_path metric measures the shortest path from door1 to door2 through the maze (including passing through the border doors).
- A longer, more winding path through the maze = higher door_path.
- Doors are at FIXED positions and cannot be moved.

## Bordered Level (showing doors and border):
```
{bordered_level_str}
```
- 'D' = door opening (fixed), '#' = wall (border or inner), '.' = empty (passable)
- Door 1: row={door1[0]}, col={door1[1]} (in bordered grid) → adjacent inner cell: {self._door_adjacent_str(door1)}
- Door 2: row={door2[0]}, col={door2[1]} (in bordered grid) → adjacent inner cell: {self._door_adjacent_str(door2)}

## Editable Inner Level (rows are y, columns are x, 0-indexed):
```
{inner_level_str}
```
Legend: '.' = empty (passable), '#' = wall (blocked)
{observation_section}

## Termination Status:
{term_status}

## {tools_str}

## Key Strategy:
- The inner cells adjacent to each door MUST be empty ('.') for the path to reach the doors.
- Create a single winding path connecting both doors to maximize door_path.
- Avoid creating multiple disconnected regions — keep exactly 1 connected region.
- Place walls strategically to force a longer path between the doors.

## Instructions:
1. Analyze the current level and its metrics
2. Plan edits to improve the score (move metrics toward goals described above)
3. Execute tool calls to modify the level

## Response Format:
Respond with a JSON object of one of these types:

### STEP - To execute tools:
```json
{{
  "type": "STEP",
  "rationale": "explanation of your reasoning",
  "plan": "high-level plan for improvement",
  "tool_calls": [
    {{"tool_name": "place_wall_segment", "parameters": {{...}}}}
  ],
  "acceptance_hint": "hint about whether to accept changes"
}}
```

### PROPOSE_SKILL - To propose a new tool:
```json
{{
  "type": "PROPOSE_SKILL",
  "rationale": "why this tool would help",
  "skill_spec": {{
    "name": "tool_name",
    "description": "what it does",
    "parameters": {{}},
    "implementation_hint": "how to implement"
  }}
}}
```

### STOP - To terminate optimization:
```json
{{
  "type": "STOP",
  "rationale": "why stopping now",
  "final_notes": "observations about the result"
}}
```

{response_guard}
Respond with ONLY the JSON object, no additional text."""

        return prompt

    def _build_freeform_zelda_prompt(self) -> str:
        """Build prompt for Zelda level optimization."""
        level_str = _render_level(self.current_level)
        metrics_str = self.evaluator.format_metrics_for_prompt(self.current_result)
        tools_str = self.registry.format_tools_for_prompt()
        term_status = self.termination.format_status()
        score_str = self.scorer.format_score(self.current_result)

        path_goal, path_instruction, enemy_goal = _zelda_goal_contract(self.config)

        # Official cumulative harness ablation: no_metric_observation removes
        # evaluator metrics and scalar score from Zelda just as it already does
        # for SMB. no_evaluation_loop inherits the same observation removal and
        # additionally bypasses accept/reject in step(). The current grid,
        # termination budget, tools, base task contract, and extra instruction
        # remain visible, so this does not become a blind-action condition.
        observation_section = ""
        if self.ablation_mode not in {"no_metric_observation", "no_evaluation_loop"}:
            observation_section = f"""
## {metrics_str}

## {score_str}
"""

        goals_block = _numbered([
            "Have exactly 1 player (P), 1 key (K), and 1 door (D)",
            "Ensure the level is playable: player can reach key, key can reach door",
            path_goal,
            "Maintain exactly 1 connected region",
            enemy_goal,
        ])
        instructions_block = _numbered([
            "First ensure the level has exactly 1 player, 1 key, and 1 door",
            "Ensure paths exist: player->key and key->door must be reachable",
            path_instruction,
            "Match the configured enemy count without blocking critical paths",
        ])
        prompt = f"""You are an AI agent optimizing a Zelda dungeon level. Your goals are:
{goals_block}

## Current Level (rows are y, columns are x, 0-indexed):
```
{level_str}
```
{_coord_bounds_block(self.current_level)}Legend: '#' = wall, '.' = empty, 'P' = player, 'K' = key, 'D' = door, 'E' = enemy
{observation_section}

## Termination Status:
{term_status}

## {tools_str}

## Instructions:
{instructions_block}

## Response Format:
Respond with a JSON object of one of these types:

### STEP - To execute tools:
```json
{{
  "type": "STEP",
  "rationale": "explanation of your reasoning",
  "plan": "high-level plan for improvement",
  "tool_calls": [
    {{"tool_name": "place_tile", "parameters": {{"mode": "single", "tile_type": "player", "y": 1, "x": 1}}}}
  ],
  "expected_delta": {{"player_key": 0, "key_door": 0}},
  "acceptance_hint": "hint about whether to accept changes"
}}
```

`expected_delta` is your predicted signed change in the two measured path
lengths after this transaction. The harness stops automatically at an exact,
hard-valid state; do not emit a different message type.

Respond with ONLY the JSON object, no additional text."""

        return prompt


    def _build_zelda_prompt(self) -> str:
        """Build prompt for Zelda level optimization."""
        if getattr(self.config.optimizer, "prompt_profile", "current") == "freeform":
            return self._build_freeform_zelda_prompt()
        level_str = _render_level(self.current_level)
        metrics_str = self.evaluator.format_metrics_for_prompt(self.current_result)
        tools_str = self.registry.format_tools_for_prompt()
        term_status = self.termination.format_status()
        score_str = self.scorer.format_score(self.current_result)

        path_goal, path_instruction, enemy_goal = _zelda_goal_contract(self.config)

        # Cumulative harness ablation: remove evaluator observations from the
        # model prompt in both no-metric and no-evaluation-loop modes.  The
        # latter also bypasses accept/reject below, but still exposes the
        # current grid, target contract, legal tools, and remaining budget.
        observation_section = ""
        if self.ablation_mode not in {"no_metric_observation", "no_evaluation_loop"}:
            observation_section = f"""
## {metrics_str}

## {score_str}
"""
        response_guard = (
            "The harness stops automatically at an exact, hard-valid "
            "state; do not emit a different message type."
            if getattr(
                self,
                "evaluator_exact_stop",
                self.ablation_mode != "no_evaluation_loop",
            )
            else "Use the available action budget to produce the best complete "
            "construction; do not emit a different message type."
        )

        goals_block = _numbered([
            "Have exactly 1 player (P), 1 key (K), and 1 door (D)",
            "Ensure the level is playable: player can reach key, key can reach door",
            path_goal,
            "Maintain exactly 1 connected region",
            enemy_goal,
        ])
        instructions_block = _numbered([
            "First ensure the level has exactly 1 player, 1 key, and 1 door",
            "Ensure paths exist: player->key and key->door must be reachable",
            path_instruction,
            "Match the configured enemy count without blocking critical paths",
        ])
        prompt = f"""You are an AI agent optimizing a Zelda dungeon level. Your goals are:
{goals_block}

## Current Level (rows are y, columns are x, 0-indexed):
```
{level_str}
```
{_coord_bounds_block(self.current_level)}Legend: '#' = wall, '.' = empty, 'P' = player, 'K' = key, 'D' = door, 'E' = enemy
{observation_section}

## Termination Status:
{term_status}

## {tools_str}

## Instructions:
{instructions_block}

## Response Format:
Respond with a JSON object of one of these types:

### STEP - To execute tools:
```json
{{
  "type": "STEP",
  "rationale": "explanation of your reasoning",
  "plan": "high-level plan for improvement",
  "tool_calls": [
    {{"tool_name": "place_tile", "parameters": {{"mode": "single", "tile_type": "player", "y": 1, "x": 1}}}}
  ],
  "expected_delta": {{"player_key": 0, "key_door": 0}},
  "acceptance_hint": "hint about whether to accept changes"
}}
```

`expected_delta` is your predicted signed change in the two measured path
lengths after this transaction. {response_guard}

Respond with ONLY the JSON object, no additional text."""

        return prompt

    def _build_sokoban_prompt(self) -> str:
        """Build prompt for Sokoban level optimization."""
        level_str = _render_level(self.current_level)
        metrics_str = self.evaluator.format_metrics_for_prompt(self.current_result)
        tools_str = self.registry.format_tools_for_prompt()
        term_status = self.termination.format_status()
        score_str = self.scorer.format_score(self.current_result)

        # R024: env-gated, fixed worked construction example (not reflector-mutated).
        # Empty when the gate is unset so the prompt stays byte-identical.
        fewshot_block = _SOKOBAN_FEWSHOT_BLOCK if _sokoban_fewshot_enabled() else ""

        # Build dynamic crate goal based on target mode
        crate_cfg = self.config.targets.crates
        if crate_cfg and crate_cfg.mode == "target" and crate_cfg.value is not None:
            crate_goal = f"2. Have ~{crate_cfg.value} crates ($) that can be pushed onto target locations (.)"
        elif crate_cfg and crate_cfg.mode == "maximize":
            crate_goal = "2. Maximize the number of crates ($) that can be pushed onto target locations (.)"
        else:
            crate_goal = "2. Have crates ($) that can be pushed onto target locations (.)"

        prompt = f"""You are an AI agent optimizing a Sokoban puzzle level. Your goals are:
1. Have exactly 1 player (@)
{crate_goal}
3. Ensure the number of crates equals the number of targets
4. Ensure the level is solvable (player can push all crates onto targets)
5. Maximize the solution length for more interesting puzzles

## Current Level (rows are y, columns are x, 0-indexed):
```
{level_str}
```
{_coord_bounds_block(self.current_level)}Legend: '#' = solid wall, ' ' = empty floor, '@' = player, '$' = crate, '.' = target

## {metrics_str}

## {score_str}

## Termination Status:
{term_status}

## {tools_str}

## Instructions:
1. First ensure the level has exactly 1 player
2. Ensure crates and targets are balanced (same count)
3. Position crates so they can be pushed onto targets
4. Leave enough empty space for the player to maneuver
5. Avoid creating unsolvable situations (crate in corner, crates blocking each other)

{fewshot_block}## Response Format:
Respond with a JSON object of one of these types:

### STEP - To execute tools:
```json
{{
  "type": "STEP",
  "rationale": "explanation of your reasoning",
  "plan": "high-level plan for improvement",
  "tool_calls": [
    {{"tool_name": "place_tile", "parameters": {{"mode": "single", "tile_type": "player", "y": 1, "x": 1}}}}
  ],
  "acceptance_hint": "hint about whether to accept changes"
}}
```

### PROPOSE_SKILL - To propose a new tool:
```json
{{
  "type": "PROPOSE_SKILL",
  "rationale": "why this tool would help",
  "skill_spec": {{
    "name": "tool_name",
    "description": "what it does",
    "parameters": {{}},
    "implementation_hint": "how to implement"
  }}
}}
```

### STOP - To terminate optimization:
```json
{{
  "type": "STOP",
  "rationale": "why stopping now",
  "final_notes": "observations about the result"
}}
```

Respond with ONLY the JSON object, no additional text."""

        return prompt

    def _build_loderunner_prompt(self) -> str:
        """Build prompt for Lode Runner level optimization."""
        level_str = _render_level(self.current_level)
        metrics_str = self.evaluator.format_metrics_for_prompt(self.current_result)
        tools_str = self.registry.format_tools_for_prompt()
        term_status = self.termination.format_status()
        score_str = self.scorer.format_score(self.current_result)

        # Build dynamic ladder/rope goals based on target mode
        ladder_cfg = self.config.targets.ladder
        rope_cfg = self.config.targets.rope
        if ladder_cfg and ladder_cfg.mode == "target" and ladder_cfg.value is not None:
            ladder_goal = f"4. Use ~{ladder_cfg.value} ladders (L) for vertical traversal"
        elif ladder_cfg and ladder_cfg.mode == "maximize":
            ladder_goal = "4. Maximize ladders (L) for vertical traversal"
        else:
            ladder_goal = "4. Use ladders (L) for vertical traversal"
        if rope_cfg and rope_cfg.mode == "target" and rope_cfg.value is not None:
            rope_goal = f"and ~{rope_cfg.value} ropes (R) for horizontal traversal"
        elif rope_cfg and rope_cfg.mode == "maximize":
            rope_goal = "and maximize ropes (R) for horizontal traversal"
        else:
            rope_goal = "and ropes (R) for horizontal traversal"

        prompt = f"""You are an AI agent optimizing a Lode Runner level. Your goals are:
1. Have exactly 1 player (P)
2. Have collectible gold pieces (G) that the player can reach
3. Have enemies (E) for challenge
{ladder_goal} {rope_goal}
5. Ensure the player can collect all gold

## Current Level (rows are y, columns are x, 0-indexed):
```
{level_str}
```
{_coord_bounds_block(self.current_level)}Legend: 'S' = solid brick, '.' = empty, 'P' = player, 'G' = gold, 'E' = enemy, 'L' = ladder, 'R' = rope

## {metrics_str}

## {score_str}

## Termination Status:
{term_status}

## {tools_str}

## Instructions:
1. First ensure the level has exactly 1 player
2. Add gold pieces that are reachable by the player
3. Create paths using ladders (vertical) and ropes (horizontal)
4. Add enemies for challenge but ensure gold is still collectible

## Response Format:
Respond with a JSON object of one of these types:

### STEP - To execute tools:
```json
{{
  "type": "STEP",
  "rationale": "explanation of your reasoning",
  "plan": "high-level plan for improvement",
  "tool_calls": [
    {{"tool_name": "place_tile", "parameters": {{"mode": "single", "tile_type": "player", "y": 1, "x": 1}}}}
  ],
  "acceptance_hint": "hint about whether to accept changes"
}}
```

### PROPOSE_SKILL - To propose a new tool:
```json
{{
  "type": "PROPOSE_SKILL",
  "rationale": "why this tool would help",
  "skill_spec": {{
    "name": "tool_name",
    "description": "what it does",
    "parameters": {{}},
    "implementation_hint": "how to implement"
  }}
}}
```

### STOP - To terminate optimization:
```json
{{
  "type": "STOP",
  "rationale": "why stopping now",
  "final_notes": "observations about the result"
}}
```

Respond with ONLY the JSON object, no additional text."""

        return prompt

    def _build_smb_static_prompt(self) -> str:
        """Build the static-count prompt with a mandatory gameplay gate."""
        level_str = _render_level(self.current_level)
        metrics_str = self.evaluator.format_metrics_for_prompt(self.current_result)
        tools_str = self.registry.format_tools_for_prompt()
        term_status = self.termination.format_status()
        score_str = self.scorer.format_score(self.current_result)
        target_lines = []
        for name in (
            "pipe_count", "enemy_tile_count", "coin_tile_count", "block_tile_count"
        ):
            target = getattr(self.config.targets, name, None)
            if target and target.mode == "target" and target.value is not None:
                target_lines.append(f"- {name}: exactly {target.value}")
            elif target and target.mode == "maximize":
                target_lines.append(f"- {name}: maximize")
        goals = "\n".join(target_lines) or "- create a coherent tile layout"

        return f"""You are optimizing static SMB tile counts in a level that must also be playable.
Match these literal-grid targets:
{goals}

## Current Level (rows are y, columns are x, 0-indexed):
```
{level_str}
```
{_coord_bounds_block(self.current_level)}Legend: '.' = empty, 'X' = count-controlled solid, '#' = fixed platform terrain, 'B' = brick, '?' = question block, 'T' = tube, 'C' = coin, 'G' = goomba, 'K' = koopa, 'Y' = spiny

## Usable-Level Contract
- Keep continuous platform/ground support so Mario can traverse from start to finish.
- Every enemy must have a solid/platform/block/tube tile immediately below it.
- Each pipe must be one disconnected, filled rectangular component exactly two columns wide, with terrain below both bottom tiles.
- The A* gameplay simulation must actually run and complete the level.
- Every C tile must be collected in that playthrough.
- '#' platform terrain is fixed support infrastructure and does not count toward block_tile_count; X/B/? do count.

## {metrics_str}

## {score_str}

## Termination Status:
{term_status}

## {tools_str}

Use compact geometric edits and preserve the bottom support floor. Count enemy, coin, and block tiles literally; a pipe is counted per valid component, not per T tile.

Respond with ONLY one JSON object: either a STEP with short rationale, short plan, and tool_calls, or a STOP."""

    _KILL_EVENT_TYPES = COUNTED_KILL_EVENTS
    _KILL_TYPE_NAMES = COUNTED_KILL_METHODS
    _SPRITE_TYPE_NAMES = SMB_SPRITE_NAMES
    _ENGINE_PAD = SMB_ENGINE_PADDING_TILES

    def _build_smb_trajectory_map(self, level_str: str, info: dict | None) -> str | None:
        """Build an annotated ASCII map showing Mario's trajectory and kill locations.

        Args:
            level_str: Current level as string (from Level.to_string()).
            info: Raw info dict from PCG benchmark evaluation.

        Returns:
            Annotated map string, or None if no simulation data.
        """
        if info is None:
            return None
        locations = info.get("locations", [])
        if not locations:
            return None

        lines = level_str.split('\n')
        grid = [list(line) for line in lines if line]
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        if width == 0:
            return None

        # Collect trajectory tile positions (engine pixel → original content coords)
        trajectory_tiles: set[tuple[int, int]] = set()
        for loc in locations:
            tx = int(loc[0] / 16) - self._ENGINE_PAD
            ty = int(loc[1] / 16)
            tx = max(0, min(width - 1, tx))
            ty = max(0, min(height - 1, ty))
            trajectory_tiles.add((ty, tx))

        # Collect kill positions from game_events
        kill_tiles: set[tuple[int, int]] = set()
        kill_details: dict[tuple[int, int], list[tuple[str, str]]] = {}  # (row,col) → [(kill_method, enemy_name)]
        for ge in info.get("game_events", []):
            event_type = ge.get("event_type", ge.get("type"))
            if event_type in self._KILL_EVENT_TYPES:
                position = ge.get("mario_event_position") or {}
                tx = int(position.get("x", int(ge["mario_x"] / 16) - self._ENGINE_PAD))
                ty = int(position.get("y", int(ge["mario_y"] / 16)))
                tx = max(0, min(width - 1, tx))
                ty = max(0, min(height - 1, ty))
                kill_tiles.add((ty, tx))
                method = self._KILL_TYPE_NAMES.get(event_type, "killed")
                param = ge.get("event_param", ge.get("param"))
                enemy = self._SPRITE_TYPE_NAMES.get(param, f"enemy({param})")
                kill_details.setdefault((ty, tx), []).append((method, enemy))

        # Build annotated map
        for (row, col) in kill_tiles:
            if 0 <= row < height and 0 <= col < width:
                grid[row][col] = '!'
        for (row, col) in trajectory_tiles:
            if (row, col) not in kill_tiles and 0 <= row < height and 0 <= col < width:
                if grid[row][col] == '.':
                    grid[row][col] = '*'

        return '\n'.join(''.join(row) for row in grid)

    def _build_smb_simulation_summary(self, info: dict | None) -> str | None:
        """Build a text summary of the SMB simulation results.

        Args:
            info: Raw info dict from PCG benchmark evaluation.

        Returns:
            Simulation summary text, or None if info is unavailable.
        """
        if info is None:
            return None
        locations = info.get("locations", [])
        if not info.get("engine_executed", bool(locations)):
            reason = info.get("engine_skip_reason") or "unknown"
            return (
                "Simulation was skipped because the engine contract failed: "
                f"{reason}. Fix the reported structural gate first."
            )
        if not locations:
            return "Simulation ran but produced no trajectory; treat the level as invalid."

        complete = info.get("complete", 0.0)
        enemies_killed = info.get("enemies", 0)
        coins_collected = info.get("coins", 0)
        jumps = info.get("jumps", 0)
        width = info.get("width", 32)

        lines = ["## Simulation Results",
                 "The Mario AI agent played through the level. Here is what happened:"]

        # Completion
        if complete >= 1.0:
            lines.append("- Completion: 100% (reached the flag)")
        else:
            last_x = int(locations[-1][0] / 16) - self._ENGINE_PAD
            last_x = max(0, min(width - 1, last_x))
            lines.append(f"- Completion: {complete*100:.0f}% (Mario died or got stuck around column {last_x})")

        # Kill details
        game_events = info.get("game_events", [])
        kill_descriptions = []
        for ge in game_events:
            event_type = ge.get("event_type", ge.get("type"))
            if event_type in self._KILL_EVENT_TYPES:
                method = self._KILL_TYPE_NAMES.get(event_type, "killed")
                param = ge.get("event_param", ge.get("param"))
                enemy = self._SPRITE_TYPE_NAMES.get(param, f"enemy({param})")
                position = ge.get("mario_event_position") or {}
                tx = int(position.get("x", int(ge["mario_x"] / 16) - self._ENGINE_PAD))
                ty = int(position.get("y", int(ge["mario_y"] / 16)))
                tx = max(0, min(width - 1, tx))
                height = int(info.get("height", 16))
                ty = max(0, min(height - 1, ty))
                kill_descriptions.append(
                    f"{enemy} {method}; Mario event position [{tx}, {ty}]"
                )

        if kill_descriptions:
            lines.append(f"- Enemies killed: {enemies_killed} ({', '.join(kill_descriptions)})")
        else:
            lines.append(f"- Enemies killed: {enemies_killed}")

        lines.append(f"- Coins collected: {coins_collected}")
        # Count stomp bounces (not counted as jumps)
        stomp_bounces = sum(
            1
            for ge in game_events
            if ge.get("event_type", ge.get("type")) == STOMP_KILL_EVENT
        )
        jump_line = f"- Jumps performed: {jumps}"
        if stomp_bounces > 0:
            jump_line += f" (stomp bounces: {stomp_bounces}, not counted as jumps)"
        lines.append(jump_line)

        # Traversal range
        first_x = max(0, int(locations[0][0] / 16) - self._ENGINE_PAD)
        last_x = max(0, int(locations[-1][0] / 16) - self._ENGINE_PAD)
        last_x = min(width - 1, last_x)
        lines.append(f"- Mario traversed from column {first_x} to column {last_x}")

        # Enemy behavior guide
        lines.append("")
        lines.append("Enemy behavior during simulation:")
        lines.append("- Goomba (G): Walks horizontally, reverses on walls. Speed ~1.75 px/frame.")
        lines.append("- Koopa (K): Green koopa walks horizontally, falls off cliffs. Becomes a kickable shell when stomped.")
        lines.append("- Spiny (Y): Walks horizontally like goomba but CANNOT be stomped (hurts Mario). Must be avoided or killed with shell/fireball.")

        return '\n'.join(lines)

    def _build_smb_prompt(self) -> str:
        """Build prompt for Super Mario Bros level optimization."""
        level_str = _render_level(self.current_level)
        metrics_str = self.evaluator.format_metrics_for_prompt(self.current_result)
        tools_str = self.registry.format_tools_for_prompt()
        term_status = self.termination.format_status()
        score_str = self.scorer.format_score(self.current_result)

        # Build dynamic enemy/coin/jump goals based on target mode
        enemies_cfg = self.config.targets.enemies
        coins_cfg = self.config.targets.coins
        jumps_cfg = self.config.targets.jumps
        smb_target_parts = []
        smb_maximize_parts = []
        if enemies_cfg and enemies_cfg.mode == "target" and enemies_cfg.value is not None:
            smb_target_parts.append(f"exactly {enemies_cfg.value} enemies killed")
        elif enemies_cfg and enemies_cfg.mode == "maximize":
            smb_maximize_parts.append("enemies killed")
        if jumps_cfg and jumps_cfg.mode == "target" and jumps_cfg.value is not None:
            smb_target_parts.append(f"exactly {jumps_cfg.value} ground jumps performed")
        elif jumps_cfg and jumps_cfg.mode == "maximize":
            smb_maximize_parts.append("jumps performed")
        if coins_cfg and coins_cfg.mode == "target" and coins_cfg.value is not None:
            smb_target_parts.append(f"exactly {coins_cfg.value} coins collected")
        elif coins_cfg and coins_cfg.mode == "maximize":
            smb_maximize_parts.append("coins collected")
        smb_goal_parts = []
        if smb_target_parts:
            smb_goal_parts.append(f"achieve {', '.join(smb_target_parts)}")
        if smb_maximize_parts:
            smb_goal_parts.append(f"maximize {', '.join(smb_maximize_parts)}")
        if smb_goal_parts:
            smb_goal = "3. " + " and ".join(smb_goal_parts).capitalize() + " during gameplay simulation"
        else:
            smb_goal = "3. Place enemies on solid ground, not floating"

        # Build simulation trajectory map and summary from current evaluation
        info = self.current_result.info if self.current_result else None
        traj_map = self._build_smb_trajectory_map(level_str, info)
        sim_summary = self._build_smb_simulation_summary(info)

        # Build trajectory map section
        if traj_map is not None:
            traj_section = f"""## Simulation Trajectory Map (last evaluation)
```
{traj_map}
```
Legend: '*' = Mario's path (empty tiles only), '!' = enemy killed here
Note: Enemies move during simulation. '!' marks where Mario was when the kill happened, not the enemy's spawn position. Enemies that were avoided or not reached are still shown at their original tile positions (G/K/Y)."""
        else:
            traj_section = """## Simulation Trajectory Map (last evaluation)
No trajectory available — simulation was skipped or has not run yet."""

        # Build simulation summary section
        sim_section = sim_summary if sim_summary else ""

        if self.ablation_mode in {"no_metric_observation", "no_evaluation_loop"}:
            return f"""You are an AI agent optimizing a Super Mario Bros level. Your goals are:
1. Create a completable level (Mario can reach the end)
2. Have proper tube/pipe structures (tubes must span 2 tiles properly)
{smb_goal}
4. Minimize horizontal noise for smoother gameplay
5. Add coins and question blocks for rewards

## Current Level (rows are y, columns are x, 0-indexed):
```
{level_str}
```
{_coord_bounds_block(self.current_level)}Legend: '.' = empty, 'X' = solid floor, '#' = solid platform, 'B' = brick, '?' = question block, 'T' = tube, 'C' = coin, 'G' = goomba, 'K' = koopa, 'Y' = spiny

## Termination Status:
{term_status}

## {tools_str}

## Instructions:
1. Ensure the level is completable (Mario can traverse from start to end)
2. Place solid ground (X) for Mario to walk on
3. Add platforms using bricks (B) and question blocks (?)
4. Place enemies (G, K, Y) on solid ground along the likely main path
5. Place coins (C) along likely traversal paths
6. Avoid creating impossible jumps or dead ends

## Response Format:
Respond with a JSON object of one of these types:

### STEP - To execute tools:
```json
{{
  "type": "STEP",
  "rationale": "explanation of your reasoning",
  "plan": "high-level plan for improvement",
  "tool_calls": [
    {{"tool_name": "place_tile", "parameters": {{"mode": "single", "tile_type": "solid", "y": 15, "x": 1}}}}
  ],
  "acceptance_hint": "hint about whether to accept changes"
}}
```

### PROPOSE_SKILL - To propose a new tool:
```json
{{
  "type": "PROPOSE_SKILL",
  "rationale": "why this tool would help",
  "skill_spec": {{
    "name": "tool_name",
    "description": "what it does",
    "parameters": {{}},
    "implementation_hint": "how to implement"
  }}
}}
```

### STOP - To terminate optimization:
```json
{{
  "type": "STOP",
  "rationale": "why stopping now",
  "final_notes": "observations about the result"
}}
```

Respond with ONLY the JSON object, no additional text."""

        prompt = f"""You are an AI agent optimizing a Super Mario Bros level. Your goals are:
1. Create a completable level (Mario can reach the end)
2. Have proper tube/pipe structures (tubes must span 2 tiles properly)
{smb_goal}
4. Minimize horizontal noise for smoother gameplay
5. Add coins and question blocks for rewards

## How Evaluation Works
A Mario AI agent (A* solver) plays your level from left to right. The metrics below reflect the **gameplay outcome**, not the tile layout:
- **enemies_killed**: Number of enemies the player stomped or hit with shells during play. Enemies that fall off cliffs on their own do NOT count. Simply placing enemy tiles does not guarantee kills — enemies must be on the player's path where they cannot be avoided.
- **coins_collected**: Number of coins the player picked up during play. Coins must be on or near the traversal path.
- **jumps**: Number of voluntary jumps the player performed from the ground. Bouncing off enemies (stomp bounces) does NOT count. More gaps and elevated platforms force more jumps.
- **complete**: Whether the player reached the end flag (1.0 = yes).

To increase enemies_killed, place enemies on narrow ground platforms that the player must cross (not on optional platforms). To increase coins_collected, place coins along the main path or on mandatory platforms. To increase jumps, create gaps in the ground and elevated platforms that force the player to jump from the ground — enemy stomps do not count as jumps.

## Current Level (rows are y, columns are x, 0-indexed):
```
{level_str}
```
{_coord_bounds_block(self.current_level)}Legend: '.' = empty, 'X' = solid floor, '#' = solid platform, 'B' = brick, '?' = question block, 'T' = tube, 'C' = coin, 'G' = goomba, 'K' = koopa, 'Y' = spiny

{traj_section}

## {metrics_str}

{sim_section}

## {score_str}

## Termination Status:
{term_status}

## {tools_str}

## Instructions:
1. Ensure the level is completable (Mario can traverse from start to end)
2. Place solid ground (X) for Mario to walk on
3. Add platforms using bricks (B) and question blocks (?)
4. Place enemies (G, K, Y) on solid ground along the player's path — enemies the player can avoid will not count
5. Place coins (C) along the main traversal path — coins the player never reaches will not count
6. Avoid creating impossible jumps or dead ends

## Response Format:
Respond with a JSON object of one of these types:

### STEP - To execute tools:
```json
{{
  "type": "STEP",
  "rationale": "explanation of your reasoning",
  "plan": "high-level plan for improvement",
  "tool_calls": [
    {{"tool_name": "place_tile", "parameters": {{"mode": "single", "tile_type": "solid", "y": 15, "x": 1}}}}
  ],
  "acceptance_hint": "hint about whether to accept changes"
}}
```

### PROPOSE_SKILL - To propose a new tool:
```json
{{
  "type": "PROPOSE_SKILL",
  "rationale": "why this tool would help",
  "skill_spec": {{
    "name": "tool_name",
    "description": "what it does",
    "parameters": {{}},
    "implementation_hint": "how to implement"
  }}
}}
```

### STOP - To terminate optimization:
```json
{{
  "type": "STOP",
  "rationale": "why stopping now",
  "final_notes": "observations about the result"
}}
```

Respond with ONLY the JSON object, no additional text."""

        return prompt

    # Per-problem-type metric keys for feedback.
    # Each entry: (metric_key, display_label, format_func_name_or_None)
    # format_func_name_or_None: "pct" for percentage, None for raw value
    _FEEDBACK_METRICS: dict[str, list[tuple[str, str, str | None]]] = {
        "binary": [
            ("path", "path", None),
            ("num_connected_regions", "regions", None),
        ],
        "binarydoor": [
            ("door_path", "door_path", None),
            ("num_connected_regions", "regions", None),
        ],
        "zelda": [
            ("player_key", "player_key", None),
            ("key_door", "key_door", None),
            ("players", "players", None),
            ("keys", "keys", None),
            ("doors", "doors", None),
            ("enemies", "enemies", None),
            ("solution_length", "solution_length", None),
            ("regions", "regions", None),
        ],
        "sokoban": [
            ("crates", "crates", None),
            ("players", "players", None),
            ("targets", "targets", None),
            ("solution_length", "solution_length", None),
        ],
        "loderunner": [
            ("ladder", "ladder", None),
            ("rope", "rope", None),
            ("player", "player", None),
            ("gold", "gold", None),
            ("enemy", "enemy", None),
            ("collected_gold", "collected_gold", None),
        ],
        "smb": [
            ("enemies", "enemies_killed", None),
            ("coins", "coins_collected", None),
            ("jumps", "jumps", None),
            ("complete", "complete", "pct"),
            ("tube_issues", "tube_issues", None),
            # ("noise", "noise", "f3"),
        ],
        "smb_static": [
            ("pipe_count", "pipe_count", None),
            ("enemy_tile_count", "enemy_tile_count", None),
            ("coin_tile_count", "coin_tile_count", None),
            ("block_tile_count", "block_tile_count", None),
            ("tube_issues", "tube_issues", None),
            ("unsupported_pipe_count", "unsupported_pipes", None),
            ("unsupported_enemy_count", "unsupported_enemies", None),
            ("complete", "complete", "pct"),
            ("coin_tiles_collected", "coin_tiles_collected", None),
        ],
    }

    def _build_environment_step_feedback(
        self,
        new_level: Level,
        new_result: EvalResult,
        tool_results: list[dict],
    ) -> str:
        """Optional compact attribution hook for an evaluated candidate."""
        return ""

    def _build_step_feedback(self, entry: ConversationEntry) -> str:
        """Build a rich feedback summary from a conversation entry's step record.

        Includes accept/reject status, per-metric deltas with targets, and
        tool call summary. Falls back to a simple one-liner when no step_record
        is available or the message type is not STEP.

        Args:
            entry: The conversation entry to build feedback from.

        Returns:
            Multi-line feedback string (no trailing newline).
        """
        # Status line (always present)
        if entry.accepted:
            status_line = f"[Previous step result: ACCEPTED - {entry.accept_reason}]"
        else:
            status_line = f"[Previous step result: REJECTED - {entry.accept_reason}]"

        rec = entry.step_record
        # Non-STEP messages or missing record: just the status line
        if rec is None or entry.message_type != "STEP" or not rec.metrics_after:
            return status_line

        lines = [status_line]

        # Metric deltas
        metric_defs = self._FEEDBACK_METRICS.get(self.problem_type, [])
        if metric_defs:
            lines.append("Metrics change from your last edit:")
            for metric_key, display_label, fmt in metric_defs:
                before = rec.metrics_before.get(metric_key)
                after = rec.metrics_after.get(metric_key)
                if before is None and after is None:
                    continue

                # Format values
                if fmt == "pct":
                    before_str = f"{before*100:.1f}%" if isinstance(before, (int, float)) else "?"
                    after_str = f"{after*100:.1f}%" if isinstance(after, (int, float)) else "?"
                elif fmt == "f3":
                    before_str = f"{before:.3f}" if isinstance(before, (int, float)) else "?"
                    after_str = f"{after:.3f}" if isinstance(after, (int, float)) else "?"
                else:
                    before_str = str(before) if before is not None else "?"
                    after_str = str(after) if after is not None else "?"

                # Target info from config
                target_cfg = getattr(self.config.targets, metric_key, None)
                if target_cfg and target_cfg.mode == "target" and target_cfg.value is not None:
                    target_str = f" (target: exactly {target_cfg.value})"
                elif target_cfg and target_cfg.mode == "maximize":
                    target_str = " (goal: maximize)"
                else:
                    target_str = ""

                lines.append(f"  {display_label}: {before_str} → {after_str}{target_str}")

        # Tool call summary
        if rec.tool_results:
            # Check for truncation marker
            truncated_entry = next(
                (tr for tr in rec.tool_results if tr.get("tool_name") == "_truncated"), None
            )
            # Count only real tool calls (exclude truncation marker)
            real_results = [tr for tr in rec.tool_results if tr.get("tool_name") != "_truncated"]
            num_calls = len(real_results)
            num_success = sum(1 for tr in real_results if tr.get("result", {}).get("success", False))
            lines.append(f"Tool results: {num_calls} calls, {num_success} succeeded, {rec.num_tiles_changed} tiles changed")
            if truncated_entry:
                lines.append(f"WARNING: {truncated_entry['result']['errors'][0]}")

        if rec.environment_feedback_after:
            lines.append(rec.environment_feedback_after)

        return "\n".join(lines)

    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        """Build multi-turn messages array from conversation history + current prompt.

        When conversation_length <= 1, returns a single user message (current behavior).
        Otherwise, includes filtered history as alternating user/assistant pairs,
        and prepends feedback about the last step's outcome to the current prompt.

        Args:
            prompt: The current step's prompt.

        Returns:
            List of message dicts with "role" and "content" keys.
        """
        if (
            self.ablation_mode in {
                "no_historical_feedback",
                "no_metric_observation",
                "no_evaluation_loop",
            }
            or self.conversation_length <= 1
            or not self._conversation_history
        ):
            return [{"role": "user", "content": prompt}]

        # Filter history based on conversation_filter
        if self.conversation_filter == "accepted":
            filtered = [e for e in self._conversation_history if e.accepted]
        else:
            filtered = list(self._conversation_history)

        # Keep last (conversation_length - 1) entries for history pairs
        max_history = self.conversation_length - 1
        history_entries = filtered[-max_history:]

        # Build messages: alternating user/assistant pairs from history
        messages = []
        for entry in history_entries:
            messages.append({"role": "user", "content": entry.prompt})
            messages.append({"role": "assistant", "content": entry.raw_response})

        # Build rich feedback about the most recent step (always from the actual
        # last entry, not the filtered list, so the LLM knows what just happened)
        last_entry = self._conversation_history[-1]
        feedback = self._build_step_feedback(last_entry)

        messages.append({"role": "user", "content": feedback + "\n\n" + prompt})

        return messages

    def _capture_grpo_request_messages(self, record: StepRecord, prompt: str) -> None:
        """Capture the exact request only under R042's default-off gate."""

        if os.environ.get("RPE_PCG_GRPO_TOKEN_CAPTURE") == "on":
            record.request_messages = self._build_messages(prompt)

    def call_llm(self, prompt: str, max_retries: int = 3, timeout: float = 300.0) -> tuple[str, bool, int, int]:
        """Call LLM and get response, with retry and timeout.

        Args:
            prompt: Prompt to send.
            max_retries: Max retry attempts on transient failures.
            timeout: Timeout in seconds for each API call.

        Returns:
            Tuple of (raw response text, truncated flag, input_tokens, output_tokens).
            truncated is True if the response was cut off by max_tokens.
            Token counts are 0 if usage info is unavailable.
        """
        self._init_client()

        messages = self._build_messages(prompt)

        # Debug: log conversation context and feedback prefix
        if self.debug and len(messages) > 1:
            self._write_debug(f"--- CONVERSATION: {len(messages)} messages ({(len(messages)-1)//2} history pairs + current) ---")
            # Log the feedback prefix from the last user message (the full prompt
            # is already logged separately in step(), so only capture the new content)
            last_msg = messages[-1]["content"]
            feedback_end = last_msg.find("\n\nYou are an AI agent")
            if feedback_end > 0:
                self._write_debug(f"--- FEEDBACK PREFIX ---\n{last_msg[:feedback_end]}\n--- END FEEDBACK PREFIX ---\n")
            else:
                self._write_debug(f"--- CURRENT MESSAGE (first 500 chars) ---\n{last_msg[:500]}...\n--- END CURRENT MESSAGE ---\n")

        # OpenAI reasoning models (o1, o3, etc.) require max_completion_tokens
        # instead of max_tokens, and don't support the temperature parameter.
        model_lower = self.config.llm.model.lower()
        is_reasoning_model = any(
            tag in model_lower for tag in ("/o1", "/o3", "/o4", "-o1", "-o3", "-o4")
        )

        if is_reasoning_model:
            api_params = dict(
                model=self.config.llm.model,
                messages=messages,
                max_completion_tokens=self.config.llm.max_tokens,
                timeout=timeout,
            )
        else:
            api_params = dict(
                model=self.config.llm.model,
                messages=messages,
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                timeout=timeout,
            )

        sampling_seed = _openai_sampling_seed(self.config)
        if sampling_seed is not None:
            api_params["seed"] = sampling_seed

        extra_body = self._build_openai_extra_body()
        if extra_body:
            api_params["extra_body"] = extra_body

        last_error = None
        self._last_reasoning_content = ""
        for attempt in range(max_retries):
            try:
                if self.config.llm.provider == "google":
                    content, truncated, input_tokens, output_tokens = self._call_google(messages, timeout)
                elif self.config.llm.provider == "anthropic":
                    content, truncated, input_tokens, output_tokens = self._call_anthropic(messages, timeout)
                else:
                    response = self._client.chat.completions.create(**api_params)
                    message = response.choices[0].message
                    content = message.content
                    reasoning = getattr(message, "reasoning_content", None)
                    if reasoning is None:
                        extra = getattr(message, "model_extra", None) or {}
                        reasoning = extra.get("reasoning_content") or extra.get("reasoning")
                    self._last_reasoning_content = str(reasoning or "")
                    finish_reason = getattr(response.choices[0], "finish_reason", None)
                    # Some OpenAI-compatible reasoning servers can return a
                    # reasoning-only message with ``content=null``.  Treat it
                    # as an incomplete structured response so the existing
                    # thinking-off handoff can recover it instead of crashing
                    # later in ``parse_response`` on ``None.strip()``.
                    missing_content = content is None
                    if missing_content:
                        content = ""
                    elif not isinstance(content, str):
                        raise TypeError(
                            "OpenAI-compatible message content must be text or null"
                        )
                    truncated = finish_reason == "length" or missing_content
                    # Extract token usage
                    input_tokens = 0
                    output_tokens = 0
                    usage = getattr(response, "usage", None)
                    if usage is not None:
                        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                        output_tokens = getattr(usage, "completion_tokens", 0) or 0
                return content, truncated, input_tokens, output_tokens
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    self.llm_retry_count += 1
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    print(f"  LLM call attempt {attempt + 1} failed ({e}), retrying in {wait}s...")
                    if self.debug:
                        import traceback
                        traceback.print_exc()
                    time.sleep(wait)

        raise last_error

    def _build_smb_openai_extra_body(self) -> dict[str, Any] | None:
        """Build vLLM/OpenAI-compatible extra_body for local constrained decoding."""
        if self.config.llm.provider != "openai":
            return None

        extra_body = dict(self.config.llm.extra_body or {})
        grammar_domain = getattr(self.config.llm, "grammar_domain", None)
        if grammar_domain and "structured_outputs" not in extra_body:
            if self._openai_structured_outputs is None:
                from toolusepcg.protocol.grammar import build_step_grammar_pack

                max_tool_calls = int(
                    getattr(self.config.llm, "grammar_max_tool_calls", None)
                    or self.config.optimizer.max_tool_calls_per_step
                )
                # R033: bound coordinate params to the grid at decode time.
                # Grid dims come from config (constant per run), so the
                # one-shot grammar cache below stays valid.
                coord_bounds = None
                if os.environ.get("RPE_PCG_COORD_GRAMMAR", "") == "on":
                    coord_bounds = (int(self.config.env.height), int(self.config.env.width))
                # Build from the actual runtime registry.  Domain-global
                # inventories can drift from experiment-specific tools and
                # previously caused a grammar/registry contract failure.
                pack = build_step_grammar_pack(
                    self.registry,
                    max_tool_calls=max_tool_calls,
                    coord_bounds=coord_bounds,
                )
                # The decoder and parser consume this exact JSON Schema.  A
                # permissive GBNF plus a stricter parser is not an admissible
                # production contract for Zelda Mastery.
                schema = json.loads(json.dumps(pack.json_schema))
                if grammar_domain == "zelda":
                    schema["properties"]["expected_delta"] = {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "player_key": {"type": "integer"},
                            "key_door": {"type": "integer"},
                        },
                        "required": ["player_key", "key_door"],
                    }
                    schema["required"].append("expected_delta")
                self._openai_step_schema = schema
                self._openai_structured_outputs = {"json": schema}
            extra_body["structured_outputs"] = self._openai_structured_outputs

        return extra_body or None


    def _build_openai_extra_body(self) -> dict[str, Any] | None:
        """Build vLLM/OpenAI-compatible extra_body for local constrained decoding."""
        if self.problem_type in {"smb", "smb_static"}:
            return self._build_smb_openai_extra_body()
        if self.config.llm.provider != "openai":
            return None

        extra_body = dict(self.config.llm.extra_body or {})
        grammar_domain = getattr(self.config.llm, "grammar_domain", None)
        if grammar_domain and "structured_outputs" not in extra_body:
            if self._openai_structured_outputs is None:
                from toolusepcg.protocol.grammar import build_grammar_for_domain

                max_tool_calls = int(
                    getattr(self.config.llm, "grammar_max_tool_calls", None)
                    or self.config.optimizer.max_tool_calls_per_step
                )
                # R033: bound coordinate params to the grid at decode time.
                # Grid dims come from config (constant per run), so the
                # one-shot grammar cache below stays valid.
                coord_bounds = None
                if os.environ.get("RPE_PCG_COORD_GRAMMAR", "") == "on":
                    coord_bounds = (int(self.config.env.height), int(self.config.env.width))
                pack = build_grammar_for_domain(
                    grammar_domain,
                    max_tool_calls=max_tool_calls,
                    coord_bounds=coord_bounds,
                )
                # The decoder and parser consume this exact JSON Schema.  A
                # permissive GBNF plus a stricter parser is not an admissible
                # production contract for Zelda Mastery.
                schema = json.loads(json.dumps(pack.json_schema))
                if grammar_domain == "zelda":
                    schema["properties"]["expected_delta"] = {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "player_key": {"type": "integer"},
                            "key_door": {"type": "integer"},
                        },
                        "required": ["player_key", "key_door"],
                    }
                    schema["required"].append("expected_delta")
                self._openai_step_schema = schema
                self._openai_structured_outputs = {"json": schema}
            extra_body["structured_outputs"] = self._openai_structured_outputs

        return extra_body or None

    def _call_google(self, messages: list[dict], timeout: float) -> tuple[str, bool, int, int]:
        """Call Google Genai API and return (content, truncated, input_tokens, output_tokens)."""
        from google.genai import types

        # Convert OpenAI-style messages to Google Genai format
        system_instruction = None
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=msg["content"])]))
            elif msg["role"] == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part.from_text(text=msg["content"])]))

        config = types.GenerateContentConfig(
            max_output_tokens=self.config.llm.max_tokens,
            temperature=self.config.llm.temperature,
            system_instruction=system_instruction,
            http_options=types.HttpOptions(timeout=int(timeout * 1000)),
        )

        response = self._client.models.generate_content(
            model=self.config.llm.model,
            contents=contents,
            config=config,
        )

        content = response.text
        # Check for truncation (MAX_TOKENS finish reason)
        truncated = False
        if response.candidates and response.candidates[0].finish_reason:
            truncated = response.candidates[0].finish_reason.name == "MAX_TOKENS"

        input_tokens = 0
        output_tokens = 0
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            candidate_tokens = getattr(usage, "candidates_token_count", 0) or 0
            thought_tokens = getattr(usage, "thoughts_token_count", 0) or 0
            output_tokens = candidate_tokens + thought_tokens

        return content, truncated, input_tokens, output_tokens

    def _call_anthropic(self, messages: list[dict], timeout: float) -> tuple[str, bool, int, int]:
        """Call Anthropic Messages API and return (content, truncated, input_tokens, output_tokens)."""
        system_text = None
        anth_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                anth_messages.append({"role": msg["role"], "content": msg["content"]})

        kwargs = dict(
            model=self.config.llm.model,
            max_tokens=self.config.llm.max_tokens,
            temperature=self.config.llm.temperature,
            messages=anth_messages,
            timeout=timeout,
        )
        if system_text is not None:
            kwargs["system"] = system_text

        response = self._client.messages.create(**kwargs)
        content = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        truncated = response.stop_reason == "max_tokens"
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        return content, truncated, input_tokens, output_tokens

    def parse_response(self, response: str, truncated: bool = False) -> tuple[StepMessage | ProposeSkillMessage | StopMessage | None, list[str]]:
        """Parse LLM response into message.

        Args:
            response: Raw response text.
            truncated: Whether the response was truncated by max_tokens.

        Returns:
            Tuple of (parsed message or None, list of errors).
        """
        grammar_domain = getattr(self.config.llm, "grammar_domain", None)
        strict_structured = self.config.llm.provider == "openai" and bool(grammar_domain)
        if strict_structured:
            # A length-truncated or half-transported action must never become
            # executable through brace completion or suffix deletion.
            if truncated:
                return None, ["Truncated structured response rejected fail-closed"]
            if self._openai_step_schema is None:
                self._build_openai_extra_body()
            try:
                parsed = json.loads(response.strip())
            except json.JSONDecodeError as exc:
                return None, [f"Strict JSON parse error: {exc}"]
            try:
                from jsonschema import Draft202012Validator

                Draft202012Validator(self._openai_step_schema).validate(parsed)
            except Exception as exc:
                return None, [f"Structured response schema error: {exc}"]
            extract_errors: list[str] = []
        else:
            # Legacy unconstrained providers retain historical repair behavior.
            parsed, extract_errors = parse_with_repair(response, truncated=truncated)
        if parsed is None:
            return None, extract_errors

        if not strict_structured:
            validation_errors = validate_message(parsed)
            if validation_errors:
                return None, validation_errors

        # Parse into typed message
        try:
            message = parse_message(parsed)
            return message, []
        except Exception as e:
            return None, [f"Failed to parse message: {e}"]

    def execute_step(self, message: StepMessage) -> tuple[Level, list[dict], int]:
        """Execute a STEP message using EditManager.

        All edits go through EditManager, which is connected to the PCG benchmark.

        Args:
            message: StepMessage with tool calls.

        Returns:
            Tuple of (new level, tool results, total tiles changed).
        """
        # Create checkpoint before edits
        checkpoint = self.edit_manager.checkpoint()

        tool_results = []
        total_changed = 0

        # Edit tools that modify levels
        edit_tools = {
            "place_wall_segment", "place_tile", "place_single_tile", "place_line", "place_patch",
            "build_smb_stomp_corridor",
            "build_smb_jump_episode",
            "build_smb_behavior_episode",
            "generate_random", "generate_maze", "generate_bsp", "generate_ca",
            "generate_connect", "generate_digger", "generate_wfc",
            "generate_zelda_tile_placing",
            "generate_zelda_random", "generate_zelda_maze", "generate_zelda_bsp",
            "generate_zelda_ca", "generate_zelda_connect", "generate_zelda_digger",
            "generate_zelda_wfc",
        }

        max_calls = self.config.optimizer.max_tool_calls_per_step
        total_requested = len(message.tool_calls)
        for i, tc in enumerate(message.tool_calls[:max_calls]):
            # Defensive: ill-trained LLMs sometimes emit `parameters` as a
            # string / list / null rather than the expected dict. Don't crash
            # the whole optimization — record a failed tool call and continue.
            params = tc.parameters
            if not isinstance(params, dict):
                from toolusepcg.tools.registry import ToolResult
                bad_type = type(params).__name__
                result = ToolResult(
                    success=False,
                    data={},
                    errors=[(
                        f"parameters must be a JSON object (dict), "
                        f"got {bad_type}: {str(params)[:80]}"
                    )],
                )
            else:
                try:
                    result = self.registry.execute(tc.tool_name, **params)
                except Exception as e:
                    from toolusepcg.tools.registry import ToolResult
                    result = ToolResult(
                        success=False,
                        data={},
                        errors=[f"{type(e).__name__}: {e}"],
                    )
            tool_results.append({
                "tool_name": tc.tool_name,
                "parameters": tc.parameters,
                "result": result.to_dict(),
            })

            # Track changes from edit tools
            if tc.tool_name in edit_tools and result.success:
                num_changed = result.data.get("num_tiles_changed", 0)
                total_changed += num_changed

        # Add truncation notice if tool calls were dropped
        if total_requested > max_calls:
            tool_results.append({
                "tool_name": "_truncated",
                "parameters": {},
                "result": {
                    "success": False,
                    "data": {},
                    "errors": [
                        f"Tool calls truncated: {total_requested} requested but limit is {max_calls} per step. "
                        f"Only the first {max_calls} were executed, {total_requested - max_calls} were dropped."
                    ],
                },
            })

        # Get the working level after all edits
        working_level = self.edit_manager.level

        # Rollback to checkpoint (actual acceptance happens in step())
        self.edit_manager.rollback(checkpoint)

        return working_level, tool_results, total_changed

    @staticmethod
    def _clamp_confidence(value: Any) -> float:
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    def _parse_acceptance_triage(self, parsed: Any) -> dict[str, Any]:
        if not isinstance(parsed, dict):
            return {
                "accepted": False,
                "decision": "reject",
                "confidence": 0.0,
                "reason": "malformed_triage_response",
                "rationale": "",
                "risk": "",
                "expected_next_benefit": "",
                "parsed": None,
            }

        decision = str(parsed.get("decision", "reject")).strip().lower()
        confidence = self._clamp_confidence(parsed.get("confidence", 0.0))
        payload = {
            "accepted": False,
            "decision": decision,
            "confidence": confidence,
            "reason": "triage_reject",
            "rationale": str(parsed.get("rationale", "")),
            "risk": str(parsed.get("risk", "")),
            "expected_next_benefit": str(parsed.get("expected_next_benefit", "")),
            "parsed": parsed,
        }
        if decision not in {"accept", "reject"}:
            payload["reason"] = "invalid_triage_decision"
            return payload
        if decision == "accept" and confidence >= self.triage_min_confidence:
            payload["accepted"] = True
            payload["reason"] = "llm_triage_accept"
            return payload
        if decision == "accept":
            payload["reason"] = "triage_low_confidence"
        return payload

    def _format_triage_context(
        self,
        *,
        message: StepMessage | None,
        new_result: EvalResult,
        new_score: float,
        num_changes: int,
        new_level: Level | None,
        tool_results: list[dict] | None,
    ) -> dict[str, Any]:
        step_payload = message.to_dict() if message is not None else {}
        return {
            "problem_type": self.problem_type,
            "step_num": self.step_count,
            "score_before": self.current_score,
            "score_after": new_score,
            "score_delta": new_score - self.current_score,
            "num_tiles_changed": num_changes,
            "metrics_before": self.current_result.metrics if self.current_result else {},
            "metrics_after": new_result.metrics,
            "errors_after": new_result.errors,
            "level_before": self.current_level.to_string() if self.current_level else "",
            "level_after": new_level.to_string() if new_level is not None else "",
            "step_message": {
                "rationale": step_payload.get("rationale", ""),
                "plan": step_payload.get("plan", ""),
                "acceptance_hint": step_payload.get("acceptance_hint", ""),
                "tool_calls": step_payload.get("tool_calls", []),
            },
            "tool_results": tool_results or [],
        }

    def _build_acceptance_triage_prompt(
        self,
        *,
        message: StepMessage | None,
        new_result: EvalResult,
        new_score: float,
        num_changes: int,
        new_level: Level | None,
        tool_results: list[dict] | None,
    ) -> str:
        context = self._format_triage_context(
            message=message,
            new_result=new_result,
            new_score=new_score,
            num_changes=num_changes,
            new_level=new_level,
            tool_results=tool_results,
        )
        return f"""You are an independent acceptance triage agent for a PCG level-editing optimizer.

The normal greedy/annealing acceptance controller did not accept this edit. Decide whether to accept it anyway because the edit is a plausible "temporarily worse, later better" move.

Be conservative:
- Accept only when the STEP rationale and acceptance_hint match the actual tool results and metric deltas.
- Accept when an edit sets up missing structure, required objects, reachability, or future controllability even if the immediate scalar score drops.
- Reject no-op, invalidating, incoherent, duplicate, or speculative edits.
- Do not accept merely because the model asked to accept; ground the decision in the metrics and changed tiles.
- Use only the JSON context below.

JSON context:
{json.dumps(context, indent=2, sort_keys=True, default=str)}

Return exactly this JSON object and no markdown:
{{
  "decision": "accept",
  "confidence": 0.0,
  "rationale": "one short sentence grounded in the context",
  "risk": "one short sentence",
  "expected_next_benefit": "one short sentence describing what later edits could improve"
}}
"""

    def _run_acceptance_triage(
        self,
        *,
        message: StepMessage | None,
        new_result: EvalResult,
        new_score: float,
        num_changes: int,
        new_level: Level | None,
        tool_results: list[dict] | None,
    ) -> dict[str, Any]:
        prompt = self._build_acceptance_triage_prompt(
            message=message,
            new_result=new_result,
            new_score=new_score,
            num_changes=num_changes,
            new_level=new_level,
            tool_results=tool_results,
        )
        try:
            self._init_triage_client()
            response = self._triage_client.chat.completions.create(
                model=self.triage_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.triage_max_tokens,
                temperature=0.0,
                timeout=300,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = response.choices[0].message.content or ""
            parsed, errors = parse_with_repair(raw)
            triage = self._parse_acceptance_triage(parsed)
            triage.update(
                {
                    "model": self.triage_model,
                    "base_url": self.triage_base_url,
                    "min_confidence": self.triage_min_confidence,
                    "raw_response": raw,
                    "parse_errors": errors,
                }
            )
            return triage
        except Exception as exc:
            return {
                "accepted": False,
                "decision": "reject",
                "confidence": 0.0,
                "reason": "triage_call_failed",
                "rationale": "",
                "risk": str(exc),
                "expected_next_benefit": "",
                "model": self.triage_model,
                "base_url": self.triage_base_url,
                "min_confidence": self.triage_min_confidence,
                "raw_response": "",
                "parse_errors": [f"{type(exc).__name__}: {exc}"],
                "parsed": None,
            }

    def decide_accept(
        self,
        new_result: EvalResult,
        num_changes: int,
        *,
        message: StepMessage | None = None,
        new_level: Level | None = None,
        tool_results: list[dict] | None = None,
    ) -> tuple[bool, str]:
        """Decide whether to accept new level.

        Args:
            new_result: Evaluation of new level.
            num_changes: Number of tiles changed.

        Returns:
            Tuple of (accept, reason).
        """
        self._last_acceptance_triage = None
        # Clean acceptance-controller ablation: preserve the ordinary prompt,
        # observations, conversation history, and evaluation logging, but do
        # not let validity or scalar score veto a non-noop transaction.
        if getattr(self, "acceptance_policy", "default") == "always_accept":
            if num_changes == 0:
                return False, "always_accept_no_changes"
            return True, "always_accept_non_noop"
        # Zelda v0 correctness kernel: hard validity is lexicographically above
        # every scalar/style objective and cannot be bypassed by exploration,
        # annealing, or LLM triage.  This is an acceptance contract, not an arm.
        acceptance_problem = getattr(
            self, "problem_type", getattr(new_result, "problem_type", None)
        )
        if acceptance_problem == "zelda" and (
            bool(new_result.valid) != bool(self.current_result.valid)
        ):
            if new_result.valid:
                return True, "accepted_hard_validity_repair"
            return False, "rejected_hard_validity_regression"
        if (
            interaction_protocol_v2_enabled()
            and bool(getattr(self.current_result, "valid", False))
            and not bool(getattr(new_result, "valid", False))
        ):
            return False, "rejected_validity_regression"
        new_score = self.scorer.score(new_result)

        # Never accept no-op steps (all tool calls failed, nothing changed)
        if num_changes == 0:
            return False, f"rejected_no_changes (score: {self.current_score:.2f} -> {new_score:.2f})"

        # Always accept if better
        if new_score > self.current_score:
            return True, f"improved (score: {self.current_score:.2f} -> {new_score:.2f})"

        # Simulated annealing: accept worse moves with decreasing probability
        if self.config.optimizer.strategy == "annealing":
            T = self.sa_initial_temp * (self.sa_decay_rate ** self.step_count)
            if T > 0:
                p = math.exp((new_score - self.current_score) / T)
                if random.random() < p:
                    return True, (f"annealing (T={T:.4f}, p={p:.4f}, "
                                  f"score: {self.current_score:.2f} -> {new_score:.2f})")
            return False, f"rejected (score: {self.current_score:.2f} -> {new_score:.2f})"

        if self.config.optimizer.strategy == "llm_triage":
            if random.random() < self.epsilon:
                return True, f"exploration (epsilon={self.epsilon}, score: {self.current_score:.2f} -> {new_score:.2f})"
            triage = self._run_acceptance_triage(
                message=message,
                new_result=new_result,
                new_score=new_score,
                num_changes=num_changes,
                new_level=new_level,
                tool_results=tool_results,
            )
            self._last_acceptance_triage = triage
            confidence = float(triage.get("confidence", 0.0) or 0.0)
            if triage.get("accepted"):
                return True, (
                    f"llm_triage_accept (confidence={confidence:.2f}, "
                    f"score: {self.current_score:.2f} -> {new_score:.2f})"
                )
            return False, (
                f"llm_triage_reject ({triage.get('reason', 'triage_reject')}, "
                f"confidence={confidence:.2f}, score: {self.current_score:.2f} -> {new_score:.2f})"
            )

        # Greedy: accept worse with probability epsilon
        if random.random() < self.epsilon:
            return True, f"exploration (epsilon={self.epsilon}, score: {self.current_score:.2f} -> {new_score:.2f})"

        return False, f"rejected (score: {self.current_score:.2f} -> {new_score:.2f})"

    def step(self, force_apply: bool = False) -> StepRecord:
        """Execute one optimization step.

        Args:
            force_apply: When True (rollout block mode, R025), the per-step
                accept/reject controller is skipped and the step's edit is
                ALWAYS tentatively applied (committed to the running level) so
                the next step in the block can build on it. The block-level
                accept/reject decision is made later by the caller. The default
                (False) preserves the original per-step behavior byte-for-byte.

        Returns:
            StepRecord with step details.
        """
        step_start = time.time()
        self.step_count += 1
        record = StepRecord(
            step_num=self.step_count,
            message_type="",
            level_before=self.current_level.to_string(),
            metrics_before=self.current_result.metrics.copy(),
            score_before=self.current_score,
        )

        message = None  # Track parsed message for later processing

        # Build and send prompt
        prompt = self.build_prompt()
        # call_llm builds the same deterministic request immediately below;
        # capture it before conversation history can change.
        self._capture_grpo_request_messages(record, prompt)

        # Debug: log the prompt
        self._write_debug(f"{'='*80}")
        self._write_debug(f"STEP {self.step_count}")
        self._write_debug(f"{'='*80}")
        self._write_debug(f"\n--- PROMPT ---\n{prompt}\n--- END PROMPT ---\n")

        truncated = False
        try:
            raw_response, truncated, input_tokens, output_tokens = self.call_llm(prompt)
            record.raw_response = raw_response
            record.reasoning_content = self._last_reasoning_content
            record.input_tokens = input_tokens
            record.output_tokens = output_tokens
        except Exception as e:
            record.errors.append(f"LLM call failed: {e}")
            record.message_type = "ERROR"
            self._write_debug(f"--- LLM ERROR ---\n{e}\n--- END LLM ERROR ---\n")
            # Fall through to logging

        # Parse response (only if LLM call succeeded)
        if not record.errors:
            # Debug: log the raw response
            truncation_note = " [TRUNCATED by max_tokens]" if truncated else ""
            self._write_debug(f"--- RAW RESPONSE{truncation_note} ---\n{raw_response}\n--- END RAW RESPONSE ---\n")

            record.truncated = truncated
            if truncated:
                # Try to salvage truncated response via aggressive repair
                message, parse_errors = self.parse_response(raw_response, truncated=True)
            else:
                message, parse_errors = self.parse_response(raw_response)
            if message is None:
                if truncated:
                    parse_errors.append("Response was truncated by max_tokens limit")
                record.errors.extend(parse_errors)
                record.message_type = "PARSE_ERROR"
                self._write_debug(f"--- PARSE_ERROR ---\nErrors: {parse_errors}\n--- END PARSE_ERROR ---\n")
                # Fall through to logging
            else:
                record.message_type = message.type.value
                record.parsed_message = message.to_dict()
                self._write_debug(f"--- PARSED OK --- type={record.message_type}\n")

        # Handle message type (only if parsing succeeded)
        if message is not None:
            if isinstance(message, StepMessage):
                # Execute tools (with rollback)
                new_level, tool_results, num_changed = self.execute_step(message)
                record.tool_results = tool_results
                record.transaction_tiles_changed = num_changed

                # Evaluate a novel level, or reuse a cached result under the
                # explicit R100 duplicate guard.  A duplicate is rejected
                # below even if its cached score would otherwise be accepted.
                level_digest = _level_digest(new_level)
                duplicate_guard_enabled = bool(
                    getattr(self, "duplicate_level_guard", False)
                )
                cached_evaluation = (
                    getattr(self, "_evaluated_level_cache", {}).get(level_digest)
                    if duplicate_guard_enabled
                    else None
                )
                if cached_evaluation is None:
                    new_result = self.evaluator.evaluate(new_level)
                    if duplicate_guard_enabled:
                        self._evaluated_level_cache[level_digest] = (
                            self.step_count,
                            copy.deepcopy(new_result),
                        )
                else:
                    first_step, cached_result = cached_evaluation
                    new_result = copy.deepcopy(cached_result)
                    record.duplicate_level = True
                    record.duplicate_first_evaluated_step = first_step
                    record.evaluator_skipped = True

                # Inject change_ratio for scoring penalty (all problems)
                import numpy as np
                total = self.current_level.data.size
                net_changed = int(
                    np.sum(new_level.data != self.current_level.data)
                )
                record.net_tiles_changed = net_changed
                effective_num_changed = (
                    net_changed if self.stable_change_accounting else num_changed
                )
                record.num_tiles_changed = effective_num_changed
                if total > 0:
                    new_result.metrics["change_ratio"] = net_changed / total

                record.level_after = new_level.to_string()
                record.metrics_after = new_result.metrics.copy()
                record.environment_feedback_after = (
                    self._build_environment_step_feedback(
                        new_level, new_result, tool_results
                    )
                )
                if record.duplicate_level:
                    duplicate_feedback = (
                        "DUPLICATE_LEVEL_GUARD: this candidate exactly repeats "
                        f"a level first evaluated at step "
                        f"{record.duplicate_first_evaluated_step}; evaluator "
                        "execution was skipped. Propose a materially different "
                        "behavior-changing edit next."
                    )
                    record.environment_feedback_after = "\n".join(
                        value
                        for value in (
                            duplicate_feedback,
                            record.environment_feedback_after,
                        )
                        if value
                    )
                record.score_after = self.scorer.score(new_result)

                # Decide acceptance. The no-evaluation-loop ablation removes the
                # hidden accept/reject controller from the generation loop; scores
                # are still logged for analysis but never determine whether edits
                # are applied.
                if record.duplicate_level:
                    accept = False
                    reason = (
                        "duplicate_level_guard_rejected "
                        f"(first_evaluated_step="
                        f"{record.duplicate_first_evaluated_step})"
                    )
                elif force_apply:
                    # Rollout block mode (R025): always tentatively apply the
                    # edit so the running level evolves across the block; the
                    # accept/reject decision is deferred to the block boundary.
                    accept = effective_num_changed > 0
                    reason = (
                        "rollout_block_tentative_apply" if accept
                        else "rollout_block_no_changes"
                    )
                    # Stash the last STEP context so the block decision can pass
                    # it to decide_accept() (e.g. llm_triage prompt building).
                    self._block_last_message = message
                    self._block_last_tool_results = tool_results
                elif getattr(
                    self,
                    "commit_policy",
                    (
                        "force_apply"
                        if self.ablation_mode == "no_evaluation_loop"
                        else "verified"
                    ),
                ) == "force_apply":
                    accept = effective_num_changed > 0
                    if accept:
                        reason = (
                            "no_evaluation_loop_applied_without_accept_reject"
                            if self.ablation_mode == "no_evaluation_loop"
                            else "force_apply_policy_applied_without_accept_reject"
                        )
                    else:
                        reason = (
                            "no_evaluation_loop_no_changes"
                            if self.ablation_mode == "no_evaluation_loop"
                            else "force_apply_policy_no_changes"
                        )
                else:
                    accept, reason = self.decide_accept(
                        new_result,
                        effective_num_changed,
                        message=message,
                        new_level=new_level,
                        tool_results=tool_results,
                    )
                record.accepted = accept
                record.accept_reason = reason
                if self._last_acceptance_triage:
                    record.acceptance_triage = self._last_acceptance_triage

                if accept:
                    # Apply changes via EditManager
                    self.edit_manager.set_level(new_level)
                    self.current_result = new_result
                    if self.stable_change_accounting:
                        # change_ratio is a transition cost, not persistent
                        # state. Carrying it into the next step creates an
                        # artificial score gain when the same final grid is
                        # rebuilt through many temporary writes.
                        self.current_result.metrics.pop("change_ratio", None)
                        self.current_score = self.scorer.score(self.current_result)
                    else:
                        self.current_score = record.score_after
                    # In rollout block mode the change budget is charged once,
                    # for the whole block, by the block decision (not per step).
                    if not force_apply:
                        self.termination.add_changes(effective_num_changed)

            elif isinstance(message, ProposeSkillMessage):
                # Handle skill proposal
                record.accepted = True
                record.accept_reason = "skill_proposed"
                # Save skill spec to disk
                if self.skill_manager:
                    self.skill_manager.propose(message.skill_spec, message.rationale)
                # Also log to trace
                if self.logger:
                    self.logger.log_skill_proposal(message.skill_spec)

            elif isinstance(message, StopMessage):
                # Agent wants to stop
                record.accepted = True
                record.accept_reason = "agent_stop"

        record.elapsed_time = time.time() - step_start

        # Always log step (success, error, or parse failure)
        if self.logger:
            self.logger.log_step(record)

        # Print step progress if verbose
        if self.verbose:
            self._print_step(record)

        self.step_records.append(record)

        # Store conversation history entry (skip ERROR steps where LLM call
        # failed entirely since there's no response to include)
        if record.message_type != "ERROR" and record.raw_response:
            self._conversation_history.append(ConversationEntry(
                prompt=prompt,
                raw_response=record.raw_response,
                accepted=record.accepted,
                accept_reason=record.accept_reason,
                message_type=record.message_type,
                step_record=record,
            ))

        return record

    def _print_step(self, record: StepRecord) -> None:
        """Print step progress to terminal.

        Args:
            record: StepRecord with step details.
        """
        # Header
        print(f"\n{'='*60}")
        token_info = f" | Tokens: {record.input_tokens}in/{record.output_tokens}out" if record.input_tokens or record.output_tokens else ""
        time_info = f" | Time: {record.elapsed_time:.1f}s" if record.elapsed_time > 0 else ""
        print(f"Step {record.step_num} | Type: {record.message_type}{token_info}{time_info}")
        print(f"{'='*60}")

        # Show rationale from parsed message
        if record.parsed_message:
            rationale = record.parsed_message.get("rationale", "")
            if rationale:
                # Truncate if too long
                if len(rationale) > 200:
                    rationale = rationale[:200] + "..."
                print(f"\nRationale: {rationale}")

        # Show tool calls
        if record.tool_results:
            print(f"\nTool Calls ({len(record.tool_results)}):")
            for i, tr in enumerate(record.tool_results):
                tool_name = tr.get("tool_name", "unknown")
                params = tr.get("parameters", {})
                result = tr.get("result", {})
                success = result.get("success", False)
                status = "OK" if success else "FAIL"

                # Defensive: a malformed LLM tool call may have parameters as
                # a non-dict (string/list/null). Skip the rich pretty-print.
                if not isinstance(params, dict):
                    bad_type = type(params).__name__
                    print(
                        f"  {i+1}. {tool_name}: <bad-params type={bad_type}: "
                        f"{str(params)[:60]!r}> -> {status}"
                    )
                    continue

                # Format params concisely
                if tool_name == "place_wall_segment":
                    mode = params.get("mode", "?")
                    tile = params.get("tile_type", "wall")
                    if mode == "start_dir_len":
                        params_str = f"({params.get('start_y')},{params.get('start_x')}) {params.get('direction')} x{params.get('length')} [{tile}]"
                    else:
                        params_str = f"({params.get('start_y')},{params.get('start_x')}) -> ({params.get('end_y')},{params.get('end_x')}) [{tile}]"
                    num_changed = result.get("data", {}).get("num_tiles_changed", 0)
                    print(f"  {i+1}. {tool_name}: {params_str} -> {status} ({num_changed} tiles)")
                elif tool_name == "place_tile":
                    mode = params.get("mode", "?")
                    tile = params.get("tile_type", "?")
                    if mode == "single":
                        params_str = f"({params.get('y')},{params.get('x')}) [{tile}]"
                    elif mode == "line":
                        if params.get("direction"):
                            params_str = f"({params.get('y')},{params.get('x')}) {params.get('direction')} x{params.get('length')} [{tile}]"
                        else:
                            params_str = f"({params.get('y')},{params.get('x')}) -> ({params.get('end_y')},{params.get('end_x')}) [{tile}]"
                    else:
                        params_str = f"({params.get('y')},{params.get('x')}) -> ({params.get('end_y')},{params.get('end_x')}) [{tile}] rect"
                    num_changed = result.get("data", {}).get("num_tiles_changed", 0)
                    print(f"  {i+1}. {tool_name}: {params_str} -> {status} ({num_changed} tiles)")
                elif tool_name == "place_single_tile":
                    tile = params.get("tile_type", "?")
                    params_str = f"({params.get('y')},{params.get('x')}) [{tile}]"
                    num_changed = result.get("data", {}).get("num_tiles_changed", 0)
                    print(f"  {i+1}. {tool_name}: {params_str} -> {status} ({num_changed} tiles)")
                elif tool_name == "place_line":
                    tile = params.get("tile_type", "?")
                    if params.get("direction"):
                        params_str = f"({params.get('y')},{params.get('x')}) {params.get('direction')} x{params.get('length')} [{tile}]"
                    else:
                        params_str = f"({params.get('y')},{params.get('x')}) -> ({params.get('end_y')},{params.get('end_x')}) [{tile}]"
                    num_changed = result.get("data", {}).get("num_tiles_changed", 0)
                    print(f"  {i+1}. {tool_name}: {params_str} -> {status} ({num_changed} tiles)")
                elif tool_name == "place_patch":
                    tile = params.get("tile_type", "?")
                    filled = "filled" if params.get("filled", True) else "border"
                    params_str = f"({params.get('y')},{params.get('x')}) -> ({params.get('end_y')},{params.get('end_x')}) [{tile}] {filled}"
                    num_changed = result.get("data", {}).get("num_tiles_changed", 0)
                    print(f"  {i+1}. {tool_name}: {params_str} -> {status} ({num_changed} tiles)")
                elif tool_name == "calculate_stats":
                    metrics = result.get("data", {}).get("metrics", {})
                    if self.problem_type == 'zelda':
                        players = metrics.get("players", "?")
                        keys = metrics.get("keys", "?")
                        doors = metrics.get("doors", "?")
                        sol_len = metrics.get("solution_length", "?")
                        print(f"  {i+1}. {tool_name}: P={players}, K={keys}, D={doors}, sol_len={sol_len}")
                    elif self.problem_type == 'sokoban':
                        players = metrics.get("players", "?")
                        crates = metrics.get("crates", "?")
                        targets = metrics.get("targets", "?")
                        heuristic = metrics.get("heuristic", "?")
                        sol_len = metrics.get("solution_length", "?")
                        solvable = "Y" if heuristic is not None and heuristic >= 0 else "N"
                        print(f"  {i+1}. {tool_name}: P={players}, crates={crates}, targets={targets}, solvable={solvable}, sol_len={sol_len}")
                    elif self.problem_type == 'loderunner':
                        player = metrics.get("player", "?")
                        gold = metrics.get("gold", "?")
                        collected = metrics.get("collected_gold", "?")
                        print(f"  {i+1}. {tool_name}: P={player}, gold={gold}, collected={collected}")
                    elif self.problem_type == 'smb':
                        complete = metrics.get("complete", 0)
                        enemies = metrics.get("enemies", "?")
                        complete_pct = f"{complete*100:.0f}%" if isinstance(complete, (int, float)) else "?"
                        print(f"  {i+1}. {tool_name}: complete={complete_pct}, enemies={enemies}")
                    elif self.problem_type == 'binarydoor':
                        door_path = metrics.get("door_path", "?")
                        regions = metrics.get("num_connected_regions", "?")
                        print(f"  {i+1}. {tool_name}: door_path={door_path}, regions={regions}")
                    else:
                        path = metrics.get("path", "?")
                        regions = metrics.get("num_connected_regions", "?")
                        print(f"  {i+1}. {tool_name}: path={path}, regions={regions}")
                else:
                    print(f"  {i+1}. {tool_name}: {status}")

        # Show metrics change
        if record.message_type == "STEP":
            print(f"\nMetrics:")

            if self.problem_type == 'zelda':
                # Zelda metrics
                for metric in ["players", "keys", "doors", "enemies"]:
                    before = record.metrics_before.get(metric, "?")
                    after = record.metrics_after.get(metric, "?")
                    print(f"  {metric}: {before} -> {after}")

                sol_before = record.metrics_before.get("solution_length", "?")
                sol_after = record.metrics_after.get("solution_length", "?")
                sol_delta = ""
                if isinstance(sol_before, (int, float)) and isinstance(sol_after, (int, float)):
                    delta = sol_after - sol_before
                    sol_delta = f" ({'+' if delta >= 0 else ''}{delta})"
                print(f"  solution_length: {sol_before} -> {sol_after}{sol_delta}")

                regions_before = record.metrics_before.get("regions", "?")
                regions_after = record.metrics_after.get("regions", "?")
                print(f"  regions: {regions_before} -> {regions_after}")
            elif self.problem_type == 'sokoban':
                # Sokoban metrics
                for metric in ["players", "crates", "targets"]:
                    before = record.metrics_before.get(metric, "?")
                    after = record.metrics_after.get(metric, "?")
                    print(f"  {metric}: {before} -> {after}")

                heur_before = record.metrics_before.get("heuristic", -1)
                heur_after = record.metrics_after.get("heuristic", -1)
                solv_before = "Y" if heur_before is not None and heur_before >= 0 else "N"
                solv_after = "Y" if heur_after is not None and heur_after >= 0 else "N"
                print(f"  solvable: {solv_before} -> {solv_after}")

                sol_before = record.metrics_before.get("solution_length", "?")
                sol_after = record.metrics_after.get("solution_length", "?")
                sol_delta = ""
                if isinstance(sol_before, (int, float)) and isinstance(sol_after, (int, float)):
                    delta = sol_after - sol_before
                    sol_delta = f" ({'+' if delta >= 0 else ''}{delta})"
                print(f"  solution_length: {sol_before} -> {sol_after}{sol_delta}")
            elif self.problem_type == 'loderunner':
                # Lode Runner metrics
                for metric in ["player", "gold", "enemy"]:
                    before = record.metrics_before.get(metric, "?")
                    after = record.metrics_after.get(metric, "?")
                    print(f"  {metric}: {before} -> {after}")

                col_before = record.metrics_before.get("collected_gold", "?")
                col_after = record.metrics_after.get("collected_gold", "?")
                print(f"  collected_gold: {col_before} -> {col_after}")
            elif self.problem_type == 'smb':
                # SMB metrics
                comp_before = record.metrics_before.get("complete", 0)
                comp_after = record.metrics_after.get("complete", 0)
                comp_before_str = f"{comp_before*100:.0f}%" if isinstance(comp_before, (int, float)) else "?"
                comp_after_str = f"{comp_after*100:.0f}%" if isinstance(comp_after, (int, float)) else "?"
                print(f"  complete: {comp_before_str} -> {comp_after_str}")

                for metric in ["enemies", "coins", "jumps"]:
                    before = record.metrics_before.get(metric, "?")
                    after = record.metrics_after.get(metric, "?")
                    print(f"  {metric}: {before} -> {after}")

                tube_before = record.metrics_before.get("tube_issues", "?")
                tube_after = record.metrics_after.get("tube_issues", "?")
                print(f"  tube_issues: {tube_before} -> {tube_after}")
            elif self.problem_type == 'binarydoor':
                # BinaryDoor metrics
                dp_before = record.metrics_before.get("door_path", "?")
                dp_after = record.metrics_after.get("door_path", "?")
                regions_before = record.metrics_before.get("num_connected_regions", "?")
                regions_after = record.metrics_after.get("num_connected_regions", "?")

                dp_delta = ""
                if isinstance(dp_before, (int, float)) and isinstance(dp_after, (int, float)):
                    delta = dp_after - dp_before
                    dp_delta = f" ({'+' if delta >= 0 else ''}{delta})"

                regions_delta = ""
                if isinstance(regions_before, (int, float)) and isinstance(regions_after, (int, float)):
                    delta = regions_after - regions_before
                    regions_delta = f" ({'+' if delta >= 0 else ''}{delta})"

                print(f"  DoorPath: {dp_before} -> {dp_after}{dp_delta}")
                print(f"  Regions:  {regions_before} -> {regions_after}{regions_delta}")
            else:
                # Binary metrics
                path_before = record.metrics_before.get("path", "?")
                path_after = record.metrics_after.get("path", "?")
                regions_before = record.metrics_before.get("num_connected_regions", "?")
                regions_after = record.metrics_after.get("num_connected_regions", "?")

                path_delta = ""
                if isinstance(path_before, (int, float)) and isinstance(path_after, (int, float)):
                    delta = path_after - path_before
                    path_delta = f" ({'+' if delta >= 0 else ''}{delta})"

                regions_delta = ""
                if isinstance(regions_before, (int, float)) and isinstance(regions_after, (int, float)):
                    delta = regions_after - regions_before
                    regions_delta = f" ({'+' if delta >= 0 else ''}{delta})"

                print(f"  Path:    {path_before} -> {path_after}{path_delta}")
                print(f"  Regions: {regions_before} -> {regions_after}{regions_delta}")

            print(f"  Score:   {record.score_before:.2f} -> {record.score_after:.2f}")

        # Show acceptance
        if record.message_type == "STEP":
            status_icon = "ACCEPTED" if record.accepted else "REJECTED"
            print(f"\nDecision: {status_icon} - {record.accept_reason}")

        # Show termination status
        term_status = self.termination.status()
        pct = term_status['budget_percent_used']
        print(f"\nBudget: {term_status['total_changes']}/{term_status['change_budget']} ({pct:.1f}%)")

        # Show errors if any
        if record.errors:
            print(f"\nErrors: {record.errors}")

        # Debug: show raw LLM response in terminal
        if self.debug and record.raw_response:
            print(f"\n--- DEBUG: Raw LLM Response ---")
            print(record.raw_response[:2000])
            if len(record.raw_response) > 2000:
                print(f"... ({len(record.raw_response)} chars total, truncated)")
            print(f"--- END Raw LLM Response ---")

    def run(self, max_steps: int = 100, max_consecutive_errors: int = 5) -> dict:
        """Run optimization loop.

        Args:
            max_steps: Maximum steps to run.
            max_consecutive_errors: Stop after this many consecutive LLM/parse errors.

        Returns:
            Summary dictionary.
        """
        if self.edit_manager is None:
            raise RuntimeError("Optimizer not initialized. Call initialize() first.")

        # R025 rollout mode: when rollout_steps > 1, group inner LLM calls into
        # blocks of k tentative steps with a single block-level accept/reject.
        # The default (rollout_steps == 1) falls through to the ORIGINAL
        # per-step loop below, completely untouched, so existing experiments are
        # byte-for-byte unaffected.
        if self.rollout_steps > 1:
            self._run_rollout_blocks(max_steps, max_consecutive_errors)
        else:
            self._run_per_step(max_steps, max_consecutive_errors)

        # Close debug log file
        if self._debug_file:
            self._debug_file.close()
            self._debug_file = None

        return self._build_summary()

    def _run_per_step(self, max_steps: int, max_consecutive_errors: int) -> None:
        """Original per-step optimization loop (rollout_steps == 1).

        This is the unmodified pre-R025 control flow: one LLM call per step,
        one accept/reject per step.
        """
        consecutive_errors = 0

        remaining_steps = max_steps - self.step_count
        for _ in range(remaining_steps):
            # Check termination
            if self.termination.should_terminate(self.current_result):
                break

            record = self.step()

            # Track consecutive errors for circuit breaker
            # Truncation-caused parse errors don't count - they indicate
            # output length issues, not API failures
            if record.message_type in ("ERROR", "PARSE_ERROR"):
                if record.message_type == "PARSE_ERROR" and record.truncated:
                    # Truncated response - log but don't count toward circuit breaker
                    if self.verbose:
                        print(f"  (truncated response, not counting toward error limit)")
                else:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"\nStopping: {consecutive_errors} consecutive errors.")
                        break
            else:
                consecutive_errors = 0

            # Check for STOP message
            if record.message_type == "STOP":
                break

    def _build_summary(self) -> dict:
        """Build the run summary dict (shared by per-step and rollout loops)."""
        total_input_tokens = sum(r.input_tokens for r in self.step_records)
        total_output_tokens = sum(r.output_tokens for r in self.step_records)
        summary = {
            "total_steps": self.step_count,
            "final_score": self.current_score,
            "final_metrics": self.current_result.metrics if self.current_result else {},
            "termination_status": self.termination.status(),
            "accepted_steps": sum(1 for r in self.step_records if r.accepted),
            "rejected_steps": sum(1 for r in self.step_records if not r.accepted and r.message_type == "STEP"),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "total_time": sum(r.elapsed_time for r in self.step_records),
            "avg_step_time": sum(r.elapsed_time for r in self.step_records) / len(self.step_records) if self.step_records else 0,
        }
        if getattr(self, "duplicate_level_guard", False):
            summary["duplicate_level_guard"] = {
                "enabled": True,
                "duplicates_rejected": sum(
                    record.duplicate_level for record in self.step_records
                ),
                "evaluator_calls_skipped": sum(
                    record.evaluator_skipped for record in self.step_records
                ),
                "retry_bound": "existing semantic max_steps; no extra LLM call",
            }

        # Log final state
        if self.logger:
            self.logger.log_final_level(self.current_level, self.current_result)
            self.logger.log_summary(summary)

        return summary

    def _run_rollout_blocks(self, max_steps: int, max_consecutive_errors: int) -> None:
        """Rollout optimization loop (rollout_steps > 1, R025).

        Groups the trial's LLM calls into blocks of up to ``rollout_steps``
        steps. Within a block every step's edit is ALWAYS tentatively applied
        (``self.step(force_apply=True)``) so the model builds on its own prior
        edits and the running level evolves freely. A block ends after k steps,
        or earlier if the model emits STOP, the trial budget is exhausted, or
        the consecutive-error circuit-breaker trips. At the block boundary a
        SINGLE accept/reject decision is made over (block-end vs block-start)
        using the same ``decide_accept`` logic; on reject the whole block is
        rolled back to the block-start checkpoint.
        """
        import numpy as np

        k = self.rollout_steps
        consecutive_errors = 0
        steps_done = 0
        remaining_steps = max_steps - self.step_count

        while steps_done < remaining_steps:
            if self.termination.should_terminate(self.current_result):
                break

            # --- Open a block: snapshot the block-start state ---
            block_checkpoint = self.edit_manager.checkpoint()
            block_start_score = self.current_score
            block_start_result = self.current_result
            self._block_last_message = None
            self._block_last_tool_results = None

            block_size = min(k, remaining_steps - steps_done)
            stop_trial = False
            error_stop = False

            for _ in range(block_size):
                record = self.step(force_apply=True)
                steps_done += 1

                # Circuit breaker bookkeeping (mirrors the per-step loop).
                if record.message_type in ("ERROR", "PARSE_ERROR"):
                    if record.message_type == "PARSE_ERROR" and record.truncated:
                        if self.verbose:
                            print(f"  (truncated response, not counting toward error limit)")
                    else:
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            error_stop = True
                            break
                else:
                    consecutive_errors = 0

                if record.message_type == "STOP":
                    stop_trial = True
                    break

            # --- Close the block: single accept/reject over cumulative edits ---
            self._finalize_block(
                block_checkpoint=block_checkpoint,
                block_start_score=block_start_score,
                block_start_result=block_start_result,
                np=np,
            )

            if error_stop:
                print(f"\nStopping: {consecutive_errors} consecutive errors.")
                break
            if stop_trial:
                break

    def _finalize_block(
        self,
        *,
        block_checkpoint,
        block_start_score,
        block_start_result,
        np,
    ) -> tuple[bool, str, int]:
        """Make the single block-level accept/reject decision (R025).

        Compares the cumulative block-end level against the block-start
        checkpoint via the existing ``decide_accept`` controller. On accept the
        block's cumulative edits are kept and the change budget is charged once;
        on reject (including the no-op case where cumulative changes == 0) the
        whole block is rolled back to ``block_checkpoint``.

        Returns ``(accept, reason, num_changes)``.
        """
        # State after the tentative block: current_* now reflect the block END.
        block_end_level = self.current_level.copy()
        block_end_result = self.current_result
        block_end_score = self.current_score

        # Cumulative changes across the whole block (robust to per-step churn).
        diff = int(np.sum(block_end_level.data != block_checkpoint.data))

        # Compare block-end against block-START: decide_accept reads
        # self.current_score, so restore it to the block-start baseline first.
        self.current_score = block_start_score
        self.current_result = block_start_result

        accept, reason = self.decide_accept(
            block_end_result,
            diff,
            message=self._block_last_message,
            new_level=block_end_level,
            tool_results=self._block_last_tool_results,
        )

        if accept:
            # Keep the block's cumulative edits; charge the budget once.
            self.edit_manager.set_level(block_end_level)
            self.current_result = block_end_result
            self.current_score = block_end_score
            self.termination.add_changes(diff)
        else:
            # Reject (or no-op): roll the entire block back to block start.
            self.edit_manager.rollback(block_checkpoint)
            self.current_score = block_start_score
            self.current_result = block_start_result

        return accept, reason, diff
