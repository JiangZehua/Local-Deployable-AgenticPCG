"""Configuration loading with YAML merge support."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import yaml


@dataclass
class MetricRange:
    """Valid range for random sampling of control parameters."""
    min_value: int | float
    max_value: int | float
    step: int | float = 1


@dataclass
class QualityExperimentSettings:
    """Settings specific to quality experiments."""
    vary_seeds: bool = True
    seed_range: tuple[int, int] = (0, 1000)
    pass_threshold: float = 1.0


@dataclass
class DiversityExperimentSettings:
    """Settings specific to diversity experiments."""
    seed_start: int = 0
    min_diversity_threshold: float = 0.4


@dataclass
class ControllabilityExperimentSettings:
    """Settings specific to controllability experiments."""
    ranges: dict[str, dict] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Unified configuration for all experiment types."""
    enabled: bool = False
    experiment_type: Literal["quality", "diversity", "controllability"] = "quality"
    num_trials: int = 10
    sampling_seed: int | None = None

    # Type-specific settings (only one used based on experiment_type)
    quality: QualityExperimentSettings | None = None
    diversity: DiversityExperimentSettings | None = None
    controllability: ControllabilityExperimentSettings | None = None

    def get_type_settings(self):
        """Get the settings for the current experiment type."""
        return getattr(self, self.experiment_type, None)


def get_control_ranges(
    problem_type: str,
    height: int,
    width: int,
    *,
    smb_envelope: str = "native",
) -> dict[str, MetricRange]:
    """Compute valid control parameter ranges from PCG benchmark definitions.

    Args:
        problem_type: Type of problem (binary, zelda, sokoban, loderunner, smb).
        height: Level height.
        width: Level width.

    Returns:
        Dictionary mapping control parameter names to MetricRange objects.
    """
    if problem_type == 'binary':
        # Binary: path ranges from width+height to width*height/2
        return {
            'path': MetricRange(
                min_value=width + height,
                max_value=max(width + height + 1, width * height // 2),
                step=1
            )
        }
    elif problem_type == 'zelda':
        # Zelda: player_key and key_door range from (width+height)/2 to width*height/4
        min_val = (width + height) // 2
        max_val = max(min_val + 1, width * height // 4)
        return {
            'player_key': MetricRange(min_value=min_val, max_value=max_val, step=1),
            'key_door': MetricRange(min_value=min_val, max_value=max_val, step=1)
        }
    elif problem_type == 'sokoban':
        # Sokoban: crates range from 1 to max(width+1, height+1)
        return {
            'crates': MetricRange(
                min_value=1,
                max_value=max(width + 1, height + 1),
                step=1
            )
        }
    elif problem_type == 'loderunner':
        # LodeRunner: ladder and rope range from 0 to 0.2 * width * height
        max_tiles = int(0.2 * width * height)
        return {
            'ladder': MetricRange(min_value=0, max_value=max(1, max_tiles), step=1),
            'rope': MetricRange(min_value=0, max_value=max(1, max_tiles), step=1)
        }
    elif problem_type == 'smb_static':
        cells = height * width
        return {
            'pipe_count': MetricRange(min_value=0, max_value=max(1, width // 3), step=1),
            'enemy_tile_count': MetricRange(min_value=0, max_value=cells, step=1),
            'coin_tile_count': MetricRange(min_value=0, max_value=cells, step=1),
            'block_tile_count': MetricRange(min_value=0, max_value=cells, step=1),
        }
    elif problem_type == 'smb':
        # SMB mastery uses named, frozen envelopes rather than an implicit
        # width-derived approximation.
        from .smb_contract import get_smb_target_envelope

        envelope = get_smb_target_envelope(smb_envelope)
        return {
            name: MetricRange(min_value=bounds[0], max_value=bounds[1], step=1)
            for name, bounds in envelope.items()
        }
    elif problem_type == 'binarydoor':
        # BinaryDoor: door_path ranges from width+height to width*height/2
        return {
            'door_path': MetricRange(
                min_value=width + height,
                max_value=max(width + height + 1, width * height // 2),
                step=1
            )
        }
    return {}


def get_problem_type(env_name: str) -> str:
    """Infer problem type from environment name."""
    if env_name.startswith('zelda'):
        return 'zelda'
    elif env_name.startswith('sokoban'):
        return 'sokoban'
    elif env_name.startswith('loderunner'):
        return 'loderunner'
    elif env_name.startswith('smb-static'):
        return 'smb_static'
    elif env_name.startswith('smb'):
        return 'smb'
    elif env_name.startswith('binarydoor'):
        return 'binarydoor'
    return 'binary'


@dataclass
class EnvConfig:
    name: str = "binary-v0"
    height: int = 16
    width: int = 16
    tiles: dict[str, int] = field(default_factory=lambda: {"EMPTY": 0, "WALL": 1})
    smb_solver: str = "auto"  # "auto" (heuristic then astar) or "astar" (A* only)

    @property
    def problem_type(self) -> str:
        """Infer problem type from environment name."""
        return get_problem_type(self.name)


@dataclass
class TargetConfig:
    mode: str = "maximize"  # "maximize" or "target"
    min_threshold: int | float | None = None
    value: int | float | None = None


@dataclass
class TargetsConfig:
    """Targets configuration - supports Binary, Zelda, and Sokoban metrics."""
    # Binary metrics
    path: TargetConfig | None = None
    num_connected_regions: TargetConfig | None = None

    # Zelda metrics (all optional, only used for Zelda problems)
    players: TargetConfig | None = None
    keys: TargetConfig | None = None
    doors: TargetConfig | None = None
    enemies: TargetConfig | None = None
    player_key: TargetConfig | None = None  # Path from player to key
    key_door: TargetConfig | None = None    # Path from key to door
    solution_length: TargetConfig | None = None  # Total path length (player_key + key_door)

    # Sokoban metrics (all optional, only used for Sokoban problems)
    crates: TargetConfig | None = None      # Number of crates (should match targets)
    targets: TargetConfig | None = None     # Number of target tiles
    heuristic: TargetConfig | None = None   # Heuristic value (0 = solvable)

    # Lode Runner metrics (all optional)
    gold: TargetConfig | None = None        # Number of gold pieces
    ladder: TargetConfig | None = None      # Number of ladder tiles
    rope: TargetConfig | None = None        # Number of rope tiles
    collected_gold: TargetConfig | None = None  # Gold that can be collected

    # SMB metrics (all optional)
    complete: TargetConfig | None = None    # Completion percentage (0-1)
    coins: TargetConfig | None = None       # Coins collected
    jumps: TargetConfig | None = None       # Number of jumps

    # Static SMB tile-grid metrics (separate from gameplay outcomes)
    pipe_count: TargetConfig | None = None
    enemy_tile_count: TargetConfig | None = None
    coin_tile_count: TargetConfig | None = None
    block_tile_count: TargetConfig | None = None

    # BinaryDoor metrics (all optional)
    door_path: TargetConfig | None = None   # Shortest path between two doors

    # R037 reward-v2 structural metrics (env-gated at load time)
    tile_entropy: TargetConfig | None = None
    transition_density: TargetConfig | None = None


@dataclass
class ScoringConfig:
    w_path: float = 1.0
    w_regions: float = 10.0
    # Zelda-specific weights for critical tiles
    w_player: float = 100.0    # Penalty weight for wrong player count
    w_key: float = 100.0       # Penalty weight for wrong key count
    w_door: float = 100.0      # Penalty weight for wrong door count
    w_enemy: float = 5.0       # Penalty weight for wrong enemy count
    w_playable: float = 200.0  # Bonus for playable level (paths exist)
    # Sokoban-specific weights
    w_crate: float = 50.0      # Penalty weight for wrong crate count
    w_target: float = 50.0     # Penalty weight for wrong target count
    w_balance: float = 100.0   # Penalty for crates != targets
    w_solvable: float = 300.0  # Bonus for solvable level (heuristic >= 0)
    w_solution: float = 1.0    # Weight for solution length
    # Lode Runner-specific weights
    w_gold: float = 20.0       # Weight for gold count
    w_ladder: float = 10.0     # Weight for ladder count
    w_rope: float = 10.0       # Weight for rope count
    w_collected: float = 50.0  # Bonus for collectible gold
    # SMB-specific weights
    w_complete: float = 300.0  # Bonus for level completion
    w_coins: float = 5.0       # Weight for coins
    w_jumps: float = 2.0       # Weight for jumps
    # Static SMB-specific weights
    w_pipe_count: float = 20.0
    w_enemy_tile_count: float = 10.0
    w_coin_tile_count: float = 10.0
    w_block_tile_count: float = 1.0
    w_coin_reachability: float = 40.0
    w_support: float = 50.0
    # BinaryDoor-specific weights
    w_door_path: float = 1.0   # Weight for door-to-door path
    w_connected: float = 200.0 # Bonus/penalty for door connectivity
    # Change penalty (all problems)
    w_change_penalty: float = 100.0  # Penalty when >60% of board changed from last accepted level
    # Structural coherence prior (all problems). Default 0 preserves older runs.
    w_coherence: float = 0.0


@dataclass
class OptimizerConfig:
    prompt_profile: str = "current"  # frozen freeform prompt for M7 / scene suites
    strategy: str = "greedy"  # "greedy", "annealing", or "llm_triage"
    # Experimental controller ablation. ``always_accept`` keeps the normal
    # prompt/feedback path but applies every non-noop transaction, including
    # validity regressions. This is intentionally distinct from
    # ``ablation_mode=no_evaluation_loop``, which also changes observations.
    acceptance_policy: str = "default"  # "default" or "always_accept"
    ablation_mode: str = "full"  # "full", "no_historical_feedback", "no_metric_observation", or "no_evaluation_loop"
    # Paper-only controller switches. ``mode_default`` preserves every legacy
    # ablation exactly: no_evaluation_loop force-applies and disables exact
    # stopping; all other modes use verified commit and exact stopping.  The
    # explicit values let the commit/stop factorial change one hidden
    # controller boundary at a time without changing metric visibility.
    commit_policy: str = "mode_default"  # "mode_default" | "force_apply" | "verified"
    evaluator_exact_stop: str = "mode_default"  # "mode_default" | "on" | "off"
    epsilon_accept_worse: float = 0.05
    annealing_initial_temp: float = 10.0   # SA starting temperature (T0)
    annealing_decay_rate: float = 0.95     # Multiplicative cooling per step: T(t) = T0 * rate^t
    triage_model: str | None = None        # llm_triage: defaults to llm.model
    triage_base_url: str | None = None     # llm_triage: defaults to llm.base_url
    triage_max_tokens: int = 512
    triage_min_confidence: float = 0.55
    max_tool_calls_per_step: int = 20
    init_strategy: str = "random"  # "random" | "weighted_random" | "empty" | "filled"
    edit_tools: list[str] | None = None  # None = auto (problem-type-based), or explicit list e.g. ["place_tile"]
    conversation_length: int = 1        # 1 = no history (current behavior)
    conversation_filter: str = "all"    # "all" or "accepted"
    extra_instruction: str | None = None  # Extra high-level instruction appended to the LLM prompt
    rollout_steps: int = 1  # Inner-loop block size: 1 = per-step accept/reject (default); k>1 = roll out k tentative steps then one block accept/reject
    # Opt-in corrected accounting for transactions that rewrite cells multiple
    # times but leave the final grid unchanged. Historical runs keep the legacy
    # cumulative-write behavior unless an experiment enables this flag.
    stable_change_accounting: bool = False


@dataclass
class TerminationConfig:
    change_budget_multiplier: float = 2.0


@dataclass
class LLMConfig:
    provider: str = "openai"  # "openai" for local vLLM by default; "portkey" is opt-in.
    model: str = "Qwen/Qwen3.6-35B-A3B-FP8"
    base_url: str = "http://localhost:8000/v1"
    max_tokens: int = 8192
    temperature: float = 0.0
    api_key: str | None = None
    extra_body: dict[str, Any] | None = field(
        default_factory=lambda: {"chat_template_kwargs": {"enable_thinking": False}}
    )
    # transformers-provider extras (ignored by other providers):
    # - base_model: required for PEFT adapter ckpts if adapter_config.json is
    #   missing base_model_name_or_path, or to override it (e.g. point at a
    #   local SFT full-FT ckpt that the adapter was trained on top of).
    # - chat_template_path: overrides the tokenizer's chat template; use
    #   configs/chat_template_no_think.jinja to drop Qwen3.5's empty
    #   <think>...</think> injection.
    # - load_in_4bit: quantize the base for inference (saves VRAM on 2B+).
    base_model: str | None = None
    chat_template_path: str | None = None
    load_in_4bit: bool = False
    # If set, build an xgrammar STEP-message grammar for this domain and pass
    # it as a logits processor at generation time → token-level schema lock.
    # Use this to kill failure modes like {"parameters": "...."} where the
    # model emits a non-dict for `parameters`.
    grammar_domain: str | None = None
    grammar_max_tool_calls: int | None = 20


def sanitize_model_name(model: str) -> str:
    """Convert a model identifier to a filesystem-safe string.

    Examples:
        "@vertexai/gemini-2.5-pro" -> "gemini-2.5-pro"
        "Qwen/Qwen3-30B-A3B-Thinking-2507" -> "Qwen3-30B-A3B-Thinking-2507"
    """
    # Take the last path component (after the final /)
    name = model.rsplit("/", 1)[-1]
    # Strip leading @ if present
    name = name.lstrip("@")
    # Replace any remaining filesystem-unsafe characters
    name = re.sub(r'[^\w\-.]', '_', name)
    return name


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    targets: TargetsConfig = field(default_factory=TargetsConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    termination: TerminationConfig = field(default_factory=TerminationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    seed: int = 42


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load configuration from YAML file with optional overrides.

    Args:
        config_path: Path to YAML config file. If None, uses default config.
        overrides: Dictionary of overrides to merge on top of loaded config.

    Returns:
        Fully resolved Config object.
    """
    # Start with default config
    default_path = Path(__file__).parent.parent / "configs" / "default.yaml"

    if default_path.exists():
        with open(default_path) as f:
            config_dict = yaml.safe_load(f) or {}
    else:
        config_dict = {}

    # Merge user config if provided
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path) as f:
                user_config = yaml.safe_load(f) or {}
            config_dict = deep_merge(config_dict, user_config)
            # A config file's targets describe the complete metric set for its
            # problem type. Do not leak Binary defaults such as `path` into
            # LodeRunner/Sokoban/SMB/Zelda overlays and saved configs.
            if "targets" in user_config:
                config_dict["targets"] = user_config["targets"] or {}
            # Controllability ranges define the complete set of controllable
            # parameters for a problem type, so the user config's ranges must
            # replace (not merge with) the default's ranges.
            user_ctrl = (user_config.get("experiment", {})
                         .get("controllability", {})
                         .get("ranges"))
            if user_ctrl is not None:
                config_dict["experiment"]["controllability"]["ranges"] = user_ctrl

    # Apply overrides
    if overrides:
        # Validate target overrides against the problem type before merging
        override_targets = overrides.get("targets", {})
        if override_targets:
            env_dict = deep_merge(
                config_dict.get("env", {}), overrides.get("env", {})
            )
            problem_type = get_problem_type(env_dict.get("name", "binary-v0"))
            valid_targets = _VALID_TARGETS_BY_PROBLEM.get(problem_type)
            if valid_targets is not None:
                valid_targets = set(valid_targets)
                if os.environ.get("RPE_PCG_REWARD_V2") == "on":
                    valid_targets |= _REWARD_V2_STRUCTURAL_TARGETS
                unknown = set(override_targets.keys()) - valid_targets
                if unknown:
                    raise ValueError(
                        f"Unknown target(s) {unknown} for problem type "
                        f"'{problem_type}'. "
                        f"Valid targets: {sorted(valid_targets)}"
                    )
            # A Zelda path override selects the complete path-objective family.
            # This prevents CLI exact targets from inheriting maximize targets
            # (or thresholds) from configs/zelda.yaml.
            if problem_type == "zelda" and (
                set(override_targets) & _ZELDA_PATH_TARGETS
            ):
                base_targets = config_dict.setdefault("targets", {})
                for metric in _ZELDA_PATH_TARGETS:
                    base_targets.pop(metric, None)
        config_dict = deep_merge(config_dict, overrides)

    # Build Config object
    return _dict_to_config(config_dict)


_VALID_TARGETS_BY_PROBLEM: dict[str, set[str]] = {
    'binary': {'path', 'num_connected_regions'},
    'binarydoor': {'door_path', 'num_connected_regions'},
    'zelda': {'players', 'keys', 'doors', 'enemies', 'player_key', 'key_door', 'solution_length'},
    'sokoban': {'crates', 'targets', 'heuristic', 'solution_length'},
    'loderunner': {'gold', 'ladder', 'rope', 'collected_gold'},
    'smb': {'complete', 'coins', 'jumps', 'enemies'},
    'smb_static': {
        'pipe_count', 'enemy_tile_count', 'coin_tile_count', 'block_tile_count'
    },
}
_REWARD_V2_STRUCTURAL_TARGETS = {'tile_entropy', 'transition_density'}
_ZELDA_PATH_TARGETS = {'player_key', 'key_door', 'solution_length'}


def _validate_zelda_path_contract(targets_dict: dict[str, Any]) -> None:
    """Reject ambiguous Zelda path objectives before they reach the harness.

    A run may target the two component distances *or* their total, never both.
    Exact targets cannot retain a maximize threshold/value from a merged config,
    and component objectives must use one mode consistently.
    """
    active = {
        name: targets_dict[name]
        for name in _ZELDA_PATH_TARGETS
        if name in targets_dict and targets_dict[name] is not None
    }
    for name, spec in active.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Zelda target '{name}' must be a mapping")
        mode = spec.get("mode", "maximize")
        if mode not in {"target", "maximize"}:
            raise ValueError(
                f"Zelda target '{name}' has invalid mode {mode!r}; "
                "expected 'target' or 'maximize'"
            )
        if mode == "target":
            if spec.get("value") is None:
                raise ValueError(f"Exact Zelda target '{name}' requires value")
            if spec.get("min_threshold") is not None:
                raise ValueError(
                    f"Exact Zelda target '{name}' cannot also define a "
                    "maximize min_threshold"
                )
        elif spec.get("value") is not None:
            raise ValueError(
                f"Maximize Zelda target '{name}' cannot also define an exact value"
            )

    components = [name for name in ("player_key", "key_door") if name in active]
    if len(components) == 1:
        raise ValueError(
            "Zelda v0 component targets must specify player_key and key_door "
            "together; partial component targets are a different contract"
        )
    if "solution_length" in active and components:
        raise ValueError(
            "Zelda solution_length objective cannot coexist with player_key/"
            "key_door objectives"
        )
    component_modes = {active[name].get("mode", "maximize") for name in components}
    if len(component_modes) > 1:
        raise ValueError(
            "Exact and maximize Zelda component objectives cannot coexist"
        )

    for name in components:
        spec = active[name]
        if spec.get("mode", "maximize") == "target":
            value = spec.get("value")
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"Zelda v0 target '{name}' must be an integer in [10, 40]"
                )
            if not 10 <= value <= 40:
                raise ValueError(
                    f"Zelda v0 target '{name}'={value} is outside [10, 40]"
                )

    solution = active.get("solution_length")
    if solution and solution.get("mode", "maximize") == "target":
        value = solution.get("value")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Zelda solution_length target must be an integer")

    enemies = targets_dict.get("enemies")
    if isinstance(enemies, dict) and enemies.get("mode", "maximize") == "target":
        value = enemies.get("value")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Zelda enemies target must be a non-negative integer")


def _validate_smb_target_contract(targets_dict: dict[str, Any]) -> None:
    """Validate exact SMB values against the frozen expanded envelope.

    The native envelope remains the default controllability sampling domain;
    the expanded envelope is the accepted mastery input contract so historical
    coin=4/5 specifications remain explicit and valid.
    """
    from .smb_contract import validate_smb_target_value

    for name in ("complete", "enemies", "coins", "jumps"):
        spec = targets_dict.get(name)
        if spec is None:
            continue
        if not isinstance(spec, dict):
            raise ValueError(f"SMB target '{name}' must be a mapping")
        mode = spec.get("mode", "maximize")
        if mode not in {"target", "maximize"}:
            raise ValueError(
                f"SMB target '{name}' has invalid mode {mode!r}; "
                "expected 'target' or 'maximize'"
            )
        if mode == "target":
            if spec.get("value") is None:
                raise ValueError(f"Exact SMB target '{name}' requires value")
            if spec.get("min_threshold") is not None:
                raise ValueError(
                    f"Exact SMB target '{name}' cannot define min_threshold"
                )
            validate_smb_target_value(name, spec["value"], envelope="expanded")
        elif spec.get("value") is not None:
            raise ValueError(
                f"Maximize SMB target '{name}' cannot also define an exact value"
            )


def _validate_smb_static_target_contract(targets_dict: dict[str, Any]) -> None:
    """Validate unambiguous non-negative integer tile-count objectives."""
    for name in (
        "pipe_count", "enemy_tile_count", "coin_tile_count", "block_tile_count"
    ):
        spec = targets_dict.get(name)
        if spec is None:
            continue
        if not isinstance(spec, dict):
            raise ValueError(f"SMB static target '{name}' must be a mapping")
        mode = spec.get("mode", "maximize")
        if mode not in {"target", "maximize"}:
            raise ValueError(
                f"SMB static target '{name}' has invalid mode {mode!r}"
            )
        value = spec.get("value")
        if mode == "target":
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"Exact SMB static target '{name}' must be a non-negative integer"
                )
            if spec.get("min_threshold") is not None:
                raise ValueError(
                    f"Exact SMB static target '{name}' cannot define min_threshold"
                )
        elif value is not None:
            raise ValueError(
                f"Maximize SMB static target '{name}' cannot define an exact value"
            )


def config_from_dict(d: dict) -> Config:
    """Convert a fully-resolved config dictionary to a Config object.

    Public wrapper for _dict_to_config(). Used when loading config from
    a saved run (e.g., resume) instead of from YAML files.

    Args:
        d: Configuration dictionary (as produced by config_to_dict()).

    Returns:
        Fully resolved Config object.
    """
    return _dict_to_config(d)


def _dict_to_config(d: dict) -> Config:
    """Convert dictionary to Config dataclass."""
    env = EnvConfig(**d.get("env", {}))

    targets_dict = d.get("targets", {})
    if env.problem_type == "zelda":
        _validate_zelda_path_contract(targets_dict)
    elif env.problem_type == "smb":
        _validate_smb_target_contract(targets_dict)
    elif env.problem_type == "smb_static":
        _validate_smb_static_target_contract(targets_dict)

    # Build targets - handle both Binary and Zelda metrics
    targets_kwargs = {}

    # Binary metrics (always present)
    if "path" in targets_dict:
        targets_kwargs["path"] = TargetConfig(**targets_dict["path"])
    if "num_connected_regions" in targets_dict:
        targets_kwargs["num_connected_regions"] = TargetConfig(**targets_dict["num_connected_regions"])

    # Zelda metrics (optional)
    zelda_metrics = ["players", "keys", "doors", "enemies", "player_key", "key_door", "solution_length"]
    for metric in zelda_metrics:
        if metric in targets_dict:
            targets_kwargs[metric] = TargetConfig(**targets_dict[metric])

    # Sokoban metrics (optional)
    sokoban_metrics = ["crates", "targets", "heuristic"]
    for metric in sokoban_metrics:
        if metric in targets_dict:
            targets_kwargs[metric] = TargetConfig(**targets_dict[metric])

    # Lode Runner metrics (optional)
    loderunner_metrics = ["gold", "ladder", "rope", "collected_gold"]
    for metric in loderunner_metrics:
        if metric in targets_dict:
            targets_kwargs[metric] = TargetConfig(**targets_dict[metric])

    # SMB metrics (optional)
    smb_metrics = ["complete", "coins", "jumps"]
    for metric in smb_metrics:
        if metric in targets_dict:
            targets_kwargs[metric] = TargetConfig(**targets_dict[metric])

    # Static SMB metrics (independent tile-count domain)
    for metric in (
        "pipe_count", "enemy_tile_count", "coin_tile_count", "block_tile_count"
    ):
        if metric in targets_dict:
            targets_kwargs[metric] = TargetConfig(**targets_dict[metric])

    # BinaryDoor metrics (optional)
    if "door_path" in targets_dict:
        targets_kwargs["door_path"] = TargetConfig(**targets_dict["door_path"])

    # R037 reward-v2 structural metrics (optional when allowed by loader validation)
    for metric in ["tile_entropy", "transition_density"]:
        if metric in targets_dict:
            targets_kwargs[metric] = TargetConfig(**targets_dict[metric])

    targets = TargetsConfig(**targets_kwargs)

    scoring = ScoringConfig(**d.get("scoring", {}))
    optimizer = OptimizerConfig(**d.get("optimizer", {}))
    termination = TerminationConfig(**d.get("termination", {}))
    llm = LLMConfig(**d.get("llm", {}))

    # Parse experiment config
    exp_dict = d.get("experiment", {})
    experiment = _parse_experiment_config(exp_dict)

    seed = d.get("seed", 42)

    return Config(
        env=env,
        targets=targets,
        scoring=scoring,
        optimizer=optimizer,
        termination=termination,
        llm=llm,
        experiment=experiment,
        seed=seed,
    )


def _parse_experiment_config(exp_dict: dict) -> ExperimentConfig:
    """Parse experiment configuration from dictionary.

    Args:
        exp_dict: Experiment configuration dictionary.

    Returns:
        ExperimentConfig object.
    """
    if not exp_dict:
        return ExperimentConfig()

    # Parse type-specific settings
    quality_settings = None
    diversity_settings = None
    controllability_settings = None

    if "quality" in exp_dict and exp_dict["quality"]:
        q = exp_dict["quality"]
        quality_settings = QualityExperimentSettings(
            vary_seeds=q.get("vary_seeds", True),
            seed_range=tuple(q.get("seed_range", [0, 1000])),
            pass_threshold=q.get("pass_threshold", 1.0),
        )

    if "diversity" in exp_dict and exp_dict["diversity"]:
        d = exp_dict["diversity"]
        diversity_settings = DiversityExperimentSettings(
            seed_start=d.get("seed_start", 0),
            min_diversity_threshold=d.get("min_diversity_threshold", 0.4),
        )

    if "controllability" in exp_dict and exp_dict["controllability"]:
        c = exp_dict["controllability"]
        controllability_settings = ControllabilityExperimentSettings(
            ranges=c.get("ranges", {}),
        )

    return ExperimentConfig(
        enabled=exp_dict.get("enabled", False),
        experiment_type=exp_dict.get("experiment_type", "quality"),
        num_trials=exp_dict.get("num_trials", 10),
        sampling_seed=exp_dict.get("sampling_seed"),
        quality=quality_settings,
        diversity=diversity_settings,
        controllability=controllability_settings,
    )


def config_to_dict(config: Config) -> dict:
    """Convert Config dataclass to dictionary for serialization."""
    from .protocol import effective_interaction_protocol

    # Build targets dict
    targets = {}

    # Add Binary targets if present
    for metric in ["path", "num_connected_regions"]:
        target = getattr(config.targets, metric, None)
        if target is not None:
            targets[metric] = {
                "mode": target.mode,
                "min_threshold": target.min_threshold,
                "value": target.value,
            }

    # Add Zelda targets if present
    zelda_metrics = ["players", "keys", "doors", "enemies", "player_key", "key_door", "solution_length"]
    for metric in zelda_metrics:
        target = getattr(config.targets, metric, None)
        if target is not None:
            targets[metric] = {
                "mode": target.mode,
                "min_threshold": target.min_threshold,
                "value": target.value,
            }

    # Add Sokoban targets if present
    sokoban_metrics = ["crates", "targets", "heuristic"]
    for metric in sokoban_metrics:
        target = getattr(config.targets, metric, None)
        if target is not None:
            targets[metric] = {
                "mode": target.mode,
                "min_threshold": target.min_threshold,
                "value": target.value,
            }

    # Add Lode Runner targets if present
    loderunner_metrics = ["gold", "ladder", "rope", "collected_gold"]
    for metric in loderunner_metrics:
        target = getattr(config.targets, metric, None)
        if target is not None:
            targets[metric] = {
                "mode": target.mode,
                "min_threshold": target.min_threshold,
                "value": target.value,
            }

    # Add SMB targets if present
    smb_metrics = ["complete", "coins", "jumps"]
    for metric in smb_metrics:
        target = getattr(config.targets, metric, None)
        if target is not None:
            targets[metric] = {
                "mode": target.mode,
                "min_threshold": target.min_threshold,
                "value": target.value,
            }

    # Add BinaryDoor targets if present
    if config.targets.door_path is not None:
        targets["door_path"] = {
            "mode": config.targets.door_path.mode,
            "min_threshold": config.targets.door_path.min_threshold,
            "value": config.targets.door_path.value,
        }

    # Add reward-v2 structural targets if present
    for metric in ["tile_entropy", "transition_density"]:
        target = getattr(config.targets, metric, None)
        if target is not None:
            targets[metric] = {
                "mode": target.mode,
                "min_threshold": target.min_threshold,
                "value": target.value,
            }

    return {
        "interaction_protocol": effective_interaction_protocol(),
        "env": {
            "name": config.env.name,
            "height": config.env.height,
            "width": config.env.width,
            "tiles": config.env.tiles,
            "smb_solver": config.env.smb_solver,
        },
        "targets": targets,
        "scoring": {
            "w_path": config.scoring.w_path,
            "w_regions": config.scoring.w_regions,
            "w_player": config.scoring.w_player,
            "w_key": config.scoring.w_key,
            "w_door": config.scoring.w_door,
            "w_enemy": config.scoring.w_enemy,
            "w_playable": config.scoring.w_playable,
            "w_crate": config.scoring.w_crate,
            "w_target": config.scoring.w_target,
            "w_balance": config.scoring.w_balance,
            "w_solvable": config.scoring.w_solvable,
            "w_solution": config.scoring.w_solution,
            "w_gold": config.scoring.w_gold,
            "w_ladder": config.scoring.w_ladder,
            "w_rope": config.scoring.w_rope,
            "w_collected": config.scoring.w_collected,
            "w_complete": config.scoring.w_complete,
            "w_coins": config.scoring.w_coins,
            "w_jumps": config.scoring.w_jumps,
            "w_coin_reachability": config.scoring.w_coin_reachability,
            "w_support": config.scoring.w_support,
            "w_door_path": config.scoring.w_door_path,
            "w_connected": config.scoring.w_connected,
            "w_change_penalty": config.scoring.w_change_penalty,
            "w_coherence": config.scoring.w_coherence,
        },
        "optimizer": {
            "strategy": config.optimizer.strategy,
            "prompt_profile": config.optimizer.prompt_profile,
            "acceptance_policy": config.optimizer.acceptance_policy,
            "ablation_mode": config.optimizer.ablation_mode,
            "commit_policy": config.optimizer.commit_policy,
            "evaluator_exact_stop": config.optimizer.evaluator_exact_stop,
            "epsilon_accept_worse": config.optimizer.epsilon_accept_worse,
            "annealing_initial_temp": config.optimizer.annealing_initial_temp,
            "annealing_decay_rate": config.optimizer.annealing_decay_rate,
            "triage_model": config.optimizer.triage_model,
            "triage_base_url": config.optimizer.triage_base_url,
            "triage_max_tokens": config.optimizer.triage_max_tokens,
            "triage_min_confidence": config.optimizer.triage_min_confidence,
            "max_tool_calls_per_step": config.optimizer.max_tool_calls_per_step,
            "edit_tools": config.optimizer.edit_tools,
            "conversation_length": config.optimizer.conversation_length,
            "conversation_filter": config.optimizer.conversation_filter,
            "extra_instruction": config.optimizer.extra_instruction,
            "rollout_steps": config.optimizer.rollout_steps,
        },
        "termination": {
            "change_budget_multiplier": config.termination.change_budget_multiplier,
        },
        "llm": {
            "provider": config.llm.provider,
            "model": config.llm.model,
            "base_url": config.llm.base_url,
            "max_tokens": config.llm.max_tokens,
            "temperature": config.llm.temperature,
            "api_key": config.llm.api_key,
            "extra_body": config.llm.extra_body,
            "base_model": config.llm.base_model,
            "chat_template_path": config.llm.chat_template_path,
            "load_in_4bit": config.llm.load_in_4bit,
            "grammar_domain": config.llm.grammar_domain,
            "grammar_max_tool_calls": config.llm.grammar_max_tool_calls,
        },
        "experiment": _experiment_config_to_dict(config.experiment),
        "seed": config.seed,
    }


def _experiment_config_to_dict(experiment: ExperimentConfig) -> dict:
    """Convert ExperimentConfig to dictionary for serialization.

    Args:
        experiment: ExperimentConfig object.

    Returns:
        Dictionary representation.
    """
    result = {
        "enabled": experiment.enabled,
        "experiment_type": experiment.experiment_type,
        "num_trials": experiment.num_trials,
        "sampling_seed": experiment.sampling_seed,
    }

    # Add type-specific settings
    if experiment.quality:
        result["quality"] = {
            "vary_seeds": experiment.quality.vary_seeds,
            "seed_range": list(experiment.quality.seed_range),
            "pass_threshold": experiment.quality.pass_threshold,
        }

    if experiment.diversity:
        result["diversity"] = {
            "seed_start": experiment.diversity.seed_start,
            "min_diversity_threshold": experiment.diversity.min_diversity_threshold,
        }

    if experiment.controllability:
        result["controllability"] = {
            "ranges": experiment.controllability.ranges,
        }

    return result
